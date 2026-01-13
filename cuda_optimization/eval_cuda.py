"""
Evaluation script for CUDA Optimization task.

This script evaluates custom CUDA kernels by:
1. Compiling and loading both original and custom models with timeout
2. Defending against thread injection and lazy evaluation attacks
3. Checking correctness against the reference implementation
4. Measuring performance (warmup 5s, run 10s)

Returns: (reward, error_msg) where reward = original_time / custom_time (speedup)
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
import torch.nn as nn
import threading
import random
import time
import traceback
from typing import Tuple, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def execute_model_with_timeout(
    model_src: str,
    context: dict,
    timeout: float = 300.0,
    build_directory: str = None,
    info_string: str = ""
) -> Tuple[bool, str, Optional[float]]:
    """
    Execute model source code with a time limit.
    
    Args:
        model_src: Source code to execute
        context: Dictionary to execute the code in
        timeout: Maximum time in seconds
        build_directory: Optional build directory for CUDA extensions
        info_string: Prefix for logging
        
    Returns:
        (success, error_message, execution_time)
    """
    info_prefix = f"[{info_string}] " if info_string else ""
    
    # Add build directory to source if provided
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_src = (
            "import os\n"
            f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_src
    
    # Static analysis for problematic patterns
    blocking_patterns = [
        ('time.sleep(', 'time.sleep() calls'),
        ('input(', 'user input'),
        ('while True:', 'infinite loops'),
    ]
    
    for pattern, description in blocking_patterns:
        if pattern in model_src:
            return False, f"Code contains blocking pattern: {description}", None
    
    def _execute_code():
        try:
            compile(model_src, "<string>", "exec")
            exec(model_src, context)
            return True
        except Exception as e:
            raise e
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute_code)
            try:
                t1 = time.time()
                future.result(timeout=timeout)
                t2 = time.time()
                execution_time = t2 - t1
                print(f"{info_prefix}Model code execution completed in {execution_time:.2f}s")
                return True, "", execution_time
                
            except FuturesTimeoutError:
                future.cancel()
                elapsed_time = time.time() - t1
                error_msg = f"Execution timeout after {elapsed_time:.2f}s"
                print(f"{info_prefix}{error_msg}")
                return False, error_msg, None
                
    except SyntaxError as e:
        error_msg = f"Syntax Error: {e}"
        return False, error_msg, None
        
    except Exception as e:
        error_msg = f"Runtime Error: {e}"
        return False, error_msg, None


def load_original_model_and_inputs(
    model_src: str,
    context: dict,
    timeout: float = 300.0,
    info_string: str = ""
) -> Tuple[Optional[type], Optional[Callable], Optional[Callable]]:
    """Load original model class and input generation functions."""
    success, error_msg, _ = execute_model_with_timeout(
        model_src=model_src,
        context=context,
        timeout=timeout,
        info_string=info_string
    )
    
    if not success:
        print(f"[{info_string}] Failed to load original model: {error_msg}")
        return None, None, None
    
    Model = context.get("Model")
    get_init_inputs = context.get("get_init_inputs")
    get_inputs = context.get("get_inputs")
    
    return Model, get_init_inputs, get_inputs


def load_custom_model(
    model_src: str,
    context: dict,
    build_directory: str = None,
    timeout: float = 300.0,
    info_string: str = ""
) -> Optional[type]:
    """Load custom model class (ModelNew)."""
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    success, error_msg, _ = execute_model_with_timeout(
        model_src=model_src,
        context=context,
        timeout=timeout,
        build_directory=build_directory,
        info_string=info_string
    )
    
    if not success:
        print(f"[{info_string}] Failed to load custom model: {error_msg}")
        return None
    
    ModelNew = context.get("ModelNew")
    
    if ModelNew is None:
        print(f"[{info_string}] Error: ModelNew was not defined in the custom model source code")
        return None
    
    if not callable(ModelNew) or not isinstance(ModelNew, type):
        print(f"[{info_string}] Error: ModelNew should be a class, got {type(ModelNew)}")
        return None
    
    return ModelNew


# =============================================================================
# Defense Functions
# =============================================================================

def defend_against_thread_injection(
    kernel: Callable,
    *args,
    **kwargs
) -> Tuple[bool, str, Any]:
    """
    Defense against thread injection attack.
    
    Thread injection spawns a background thread to do computation while
    returning an empty tensor immediately. This cheats timing but passes
    correctness checks since the thread finishes before verification.
    
    Defense: Compare thread count before and after kernel execution.
    
    Returns:
        (passed, message, output)
    """
    before = threading.active_count()
    output = kernel(*args, **kwargs)
    torch.cuda.synchronize()
    after = threading.active_count()
    
    if after > before:
        return False, f"Kernel spawned background thread (before={before}, after={after})", output
    else:
        return True, "Thread injection check passed", output


def defend_against_stream_injection(
    kernel: Callable,
    *args,
    ratio_threshold: float = 1.5,
    device: torch.device = None,
    **kwargs
) -> Tuple[bool, str, Any, Optional[float]]:
    """
    Defense against stream injection attack using hybrid approach.
    
    Stream injection runs the kernel on a separate CUDA stream, causing
    events recorded on the default stream to miss the actual computation time.
    
    Returns:
        (passed, message, output, timing_ms)
    """
    # Warmup
    _ = kernel(*args, **kwargs)
    torch.cuda.synchronize(device=device)
    
    num_iterations = 10
    original_times = []
    defense_times = []
    output = None
    
    for i in range(num_iterations):
        run_original_first = random.choice([True, False])
        
        def time_original():
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = kernel(*args, **kwargs)
            e.record()
            torch.cuda.synchronize(device=device)
            return s.elapsed_time(e), out
        
        def time_defense():
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = kernel(*args, **kwargs)
            torch.cuda.synchronize(device=device)  # Force sync ALL streams
            e.record()
            torch.cuda.synchronize(device=device)
            return s.elapsed_time(e), out
        
        if run_original_first:
            t, output = time_original()
            original_times.append(t)
            t, output = time_defense()
            defense_times.append(t)
        else:
            t, output = time_defense()
            defense_times.append(t)
            t, output = time_original()
            original_times.append(t)
    
    # Use median to reduce noise
    original_time_ms = sorted(original_times)[num_iterations // 2]
    defense_time_ms = sorted(defense_times)[num_iterations // 2]
    
    # Compare: Large ratio indicates stream injection
    if original_time_ms < 0.001:  # Less than 1µs
        ratio = float('inf')
    else:
        ratio = defense_time_ms / original_time_ms
    
    if ratio > ratio_threshold:
        return (
            False,
            f"Stream injection detected! Original: {original_time_ms:.3f}ms, "
            f"Defense: {defense_time_ms:.3f}ms, Ratio: {ratio:.1f}x",
            output,
            defense_time_ms
        )
    else:
        return (
            True,
            f"Stream injection check passed. Ratio: {ratio:.2f}x",
            output,
            original_time_ms
        )


def defend_against_lazy_evaluation(
    kernel: Callable,
    *args,
    expected_device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[bool, str, Any]:
    """
    Defense against lazy evaluation attack.
    
    Lazy evaluation returns a tensor subclass that defers computation until
    certain methods (like __eq__) are called.
    
    Returns:
        (passed, message, output)
    """
    output = kernel(*args, **kwargs)
    
    # Infer expected device from first input argument
    if expected_device is None:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                expected_device = arg.device
                break
    
    if expected_device is None:
        expected_device = torch.device('cuda')
    
    # Handle multiple outputs
    outputs_to_check = [output] if isinstance(output, torch.Tensor) else output
    if isinstance(output, (tuple, list)):
        outputs_to_check = output
    
    for idx, out in enumerate(outputs_to_check):
        prefix = f"Output[{idx}]" if len(outputs_to_check) > 1 else "Output"
        
        # Check 1: Must be a tensor
        if not isinstance(out, torch.Tensor):
            return False, f"{prefix} is not a tensor: {type(out)}", output
        
        # Check 2: Must be standard torch.Tensor, not a subclass
        if type(out).__name__ not in ['Tensor', 'Parameter']:
            return False, f"{prefix} is {type(out).__name__}, not standard torch.Tensor", output
        
        # Check 3: Must be on correct device
        if out.device != expected_device:
            return False, f"{prefix} on wrong device: {out.device} (expected {expected_device})", output
        
        # Check 4: Must have allocated storage
        try:
            storage_size = out.untyped_storage().size()
            if storage_size == 0:
                return False, f"{prefix} has no allocated storage (likely lazy)", output
        except Exception as e:
            return False, f"{prefix} cannot access storage: {e}", output
        
        # Check 5: Storage pointer must be valid
        try:
            ptr = out.data_ptr()
            if ptr == 0:
                return False, f"{prefix} storage pointer is null (likely lazy)", output
        except Exception as e:
            return False, f"{prefix} cannot access data pointer: {e}", output
    
    return True, "Lazy evaluation check passed", output


# =============================================================================
# Correctness Check
# =============================================================================

def check_correctness(
    original_model: nn.Module,
    custom_model: nn.Module,
    get_inputs: Callable,
    device: torch.device,
    num_trials: int = 5,
    seed: int = 42,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    info_string: str = ""
) -> Tuple[bool, str, dict]:
    """
    Check correctness of custom model against reference implementation.
    
    Returns:
        (success, error_message, metadata)
    """
    info_prefix = f"[{info_string}] " if info_string else ""
    
    metadata = {
        "num_trials": num_trials,
        "trials_passed": 0,
        "max_difference": [],
        "avg_difference": []
    }
    
    # Generate trial seeds deterministically
    torch.manual_seed(seed)
    trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_trials)]
    
    with torch.no_grad():
        for trial in range(num_trials):
            trial_seed = trial_seeds[trial]
            
            try:
                # Generate inputs
                set_seed(trial_seed)
                inputs = get_inputs()
                inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
                
                # Run original model
                set_seed(trial_seed)
                original_model.eval()
                original_output = original_model(*inputs)
                torch.cuda.synchronize(device=device)
                
                # Run custom model
                set_seed(trial_seed)
                custom_model.eval()
                custom_output = custom_model(*inputs)
                torch.cuda.synchronize(device=device)
                
                # Check shapes
                if original_output.shape != custom_output.shape:
                    return False, f"Shape mismatch: expected {original_output.shape}, got {custom_output.shape}", metadata
                
                # Check values
                if not torch.allclose(original_output, custom_output, atol=atol, rtol=rtol):
                    max_diff = torch.max(torch.abs(original_output - custom_output)).item()
                    avg_diff = torch.mean(torch.abs(original_output - custom_output)).item()
                    metadata["max_difference"].append(max_diff)
                    metadata["avg_difference"].append(avg_diff)
                    return False, f"Value mismatch: max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}", metadata
                
                metadata["trials_passed"] += 1
                
            except Exception as e:
                return False, f"Runtime error in trial {trial + 1}: {e}", metadata
    
    return True, "", metadata


# =============================================================================
# Performance Evaluation
# =============================================================================

def eval_performance(
    original_model: nn.Module,
    custom_model: nn.Module,
    get_inputs: Callable,
    device: torch.device,
    warmup_time: float = 5.0,
    run_time: float = 10.0,
    seed: int = 42,
    info_string: str = ""
) -> Tuple[Optional[float], str, dict]:
    """
    Measure performance of custom model vs original.
    
    Warmup for warmup_time seconds, then run for run_time seconds.
    
    Returns:
        (score, message, metadata) where score = original_time / custom_time (speedup)
    """
    info_prefix = f"[{info_string}] " if info_string else ""
    
    metadata = {
        "warmup_time": warmup_time,
        "run_time": run_time,
        "original_times_ms": [],
        "custom_times_ms": [],
        "original_avg_ms": 0.0,
        "custom_avg_ms": 0.0,
        "num_iterations": 0
    }
    
    # Generate inputs
    set_seed(seed)
    inputs = get_inputs()
    inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
    
    # Warmup phase
    print(f"{info_prefix}Warming up for {warmup_time}s...")
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_time:
        with torch.no_grad():
            _ = original_model(*inputs)
            _ = custom_model(*inputs)
            torch.cuda.synchronize(device=device)
    
    # Performance measurement phase
    print(f"{info_prefix}Running performance measurement for {run_time}s...")
    original_times = []
    custom_times = []
    
    run_start = time.time()
    iteration = 0
    
    with torch.no_grad():
        while time.time() - run_start < run_time:
            # Generate fresh inputs for each iteration
            set_seed(seed + iteration)
            inputs = get_inputs()
            inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
            
            # Randomize order to eliminate systematic bias
            run_original_first = random.choice([True, False])
            
            if run_original_first:
                # Time original
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = original_model(*inputs)
                end_event.record()
                torch.cuda.synchronize(device=device)
                original_time = start_event.elapsed_time(end_event)
                
                # Time custom
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = custom_model(*inputs)
                end_event.record()
                torch.cuda.synchronize(device=device)
                custom_time = start_event.elapsed_time(end_event)
            else:
                # Time custom first
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = custom_model(*inputs)
                end_event.record()
                torch.cuda.synchronize(device=device)
                custom_time = start_event.elapsed_time(end_event)
                
                # Time original
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = original_model(*inputs)
                end_event.record()
                torch.cuda.synchronize(device=device)
                original_time = start_event.elapsed_time(end_event)
            
            original_times.append(original_time)
            custom_times.append(custom_time)
            iteration += 1
    
    if not original_times or not custom_times:
        return None, "No performance measurements collected", metadata
    
    # Calculate averages
    avg_original = sum(original_times) / len(original_times)
    avg_custom = sum(custom_times) / len(custom_times)
    
    metadata["original_times_ms"] = original_times
    metadata["custom_times_ms"] = custom_times
    metadata["original_avg_ms"] = avg_original
    metadata["custom_avg_ms"] = avg_custom
    metadata["num_iterations"] = iteration
    
    # Calculate score (speedup)
    if avg_custom <= 0:
        return None, "Invalid custom model timing", metadata
    
    score = avg_original / avg_custom
    
    message = f"Original: {avg_original:.3f}ms, Custom: {avg_custom:.3f}ms, Score: {score:.3f}x"
    if score > 1.0:
        message += f" ({score:.2f}x faster)"
    elif score < 1.0:
        message += f" ({1.0/score:.2f}x slower)"
    
    print(f"{info_prefix}{message}")
    
    return score, message, metadata


# =============================================================================
# Main get_reward Function
# =============================================================================

def get_reward(
    code: str,
    work_dir: str,
    device_index: int = 0,
    warmup_time: float = 5.0,
    run_time: float = 10.0,
    num_correct_trials: int = 5,
    info_string: str = ""
) -> Tuple[float, str, str]:
    """
    Evaluate a custom CUDA kernel against the reference implementation.
    
    Pipeline:
    1. Load original model from initial_code.py and custom model from code string
    2. Defense against thread injection, stream injection, and lazy evaluation
    3. Correctness check
    4. Performance evaluation (warmup + timed run)
    
    Args:
        code: Python code string containing the custom model (ModelNew class)
        work_dir: Working directory for CUDA extension builds
        device_index: GPU device index to use
        warmup_time: Warmup duration in seconds (default: 5.0)
        run_time: Performance measurement duration in seconds (default: 10.0)
        num_correct_trials: Number of correctness trials (default: 5)
        info_string: Prefix for logging
        
    Returns:
        Tuple of (reward, error_msg, details)
        - reward: float representing speedup (original_time / custom_time), 0 if failed
        - error_msg: "" if successful, error description otherwise
        - details: "" (reserved for future use)
    """
    from pathlib import Path
    
    info_prefix = f"[{info_string}] " if info_string else ""
    
    if not torch.cuda.is_available():
        return 0.0, "CUDA is not available", ""
    
    # Load original model source from initial_code.py in the same directory
    script_dir = Path(__file__).parent.resolve()
    original_model_path = script_dir / "initial_code.py"
    
    if not original_model_path.exists():
        return 0.0, f"Original model file not found: {original_model_path}", ""
    
    original_model_src = load_src_from_file(str(original_model_path))
    print(f"{info_prefix}Loaded original model from: {original_model_path}")
    
    # Use code string directly
    custom_model_src = code
    print(f"{info_prefix}Loaded custom model from code string")
    
    # If ModelNew is not defined but Model is, rename Model to ModelNew
    import re
    if 'class ModelNew' not in custom_model_src and 'class Model' in custom_model_src:
        print(f"{info_prefix}Renaming 'Model' to 'ModelNew' in custom code...")
        # Replace class definition
        custom_model_src = re.sub(r'\bclass Model\b', 'class ModelNew', custom_model_src)
        # Replace Model references (but not ModelNew)
        custom_model_src = re.sub(r'\bModel\b(?!New)', 'ModelNew', custom_model_src)
    
    device = torch.device(f'cuda:{device_index}')
    torch.cuda.set_device(device)
    
    # Create work directory for CUDA builds
    os.makedirs(work_dir, exist_ok=True)
    build_dir = os.path.join(work_dir, "cuda_build")
    os.makedirs(build_dir, exist_ok=True)
    
    seed = 42
    
    try:
        # =====================================================================
        # Step 1: Load original model
        # =====================================================================
        print(f"{info_prefix}Step 1: Loading original model...")
        context_original = {}
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
            original_model_src, context_original, timeout=60.0, info_string=info_string
        )
        
        if Model is None or get_init_inputs is None or get_inputs is None:
            return 0.0, "Failed to load original model"
        
        # Initialize original model
        set_seed(seed)
        init_inputs = get_init_inputs()
        init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]
        
        with torch.no_grad():
            set_seed(seed)
            original_model = Model(*init_inputs).to(device)
        
        # =====================================================================
        # Step 2: Load custom model
        # =====================================================================
        print(f"{info_prefix}Step 2: Loading custom model...")
        context_custom = {}
        ModelNew = load_custom_model(
            custom_model_src, context_custom, build_directory=build_dir, 
            timeout=120.0, info_string=info_string
        )
        
        if ModelNew is None:
            return 0.0, "Failed to load custom model (ModelNew not defined)"
        
        # Initialize custom model
        with torch.no_grad():
            set_seed(seed)
            custom_model = ModelNew(*init_inputs).to(device)
        
        torch.cuda.synchronize(device=device)
        
        # =====================================================================
        # Step 3: Defense checks
        # =====================================================================
        print(f"{info_prefix}Step 3: Running defense checks...")
        
        # Generate test inputs
        set_seed(seed)
        test_inputs = get_inputs()
        test_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in test_inputs]
        
        # Defense against lazy evaluation
        passed, msg, _ = defend_against_lazy_evaluation(
            lambda: custom_model(*test_inputs),
            expected_device=device
        )
        if not passed:
            return 0.0, f"Lazy evaluation detected: {msg}", ""
        print(f"{info_prefix}  Lazy evaluation check: PASSED")
        
        # Defense against thread injection
        passed, msg, _ = defend_against_thread_injection(
            lambda: custom_model(*test_inputs)
        )
        if not passed:
            return 0.0, f"Thread injection detected: {msg}", ""
        print(f"{info_prefix}  Thread injection check: PASSED")
        
        # Defense against stream injection
        passed, msg, _, _ = defend_against_stream_injection(
            lambda: custom_model(*test_inputs),
            device=device
        )
        if not passed:
            return 0.0, f"Stream injection detected: {msg}", ""
        print(f"{info_prefix}  Stream injection check: PASSED")
        
        # =====================================================================
        # Step 4: Correctness check
        # =====================================================================
        print(f"{info_prefix}Step 4: Checking correctness...")
        correct, error_msg, metadata = check_correctness(
            original_model=original_model,
            custom_model=custom_model,
            get_inputs=get_inputs,
            device=device,
            num_trials=num_correct_trials,
            seed=seed,
            info_string=info_string
        )
        
        if not correct:
            return 0.0, f"Correctness check failed: {error_msg}", ""
        print(f"{info_prefix}  Correctness check: PASSED ({metadata['trials_passed']}/{num_correct_trials} trials)")
        
        # =====================================================================
        # Step 5: Performance evaluation
        # =====================================================================
        print(f"{info_prefix}Step 5: Evaluating performance...")
        score, perf_msg, perf_metadata = eval_performance(
            original_model=original_model,
            custom_model=custom_model,
            get_inputs=get_inputs,
            device=device,
            warmup_time=warmup_time,
            run_time=run_time,
            seed=seed,
            info_string=info_string
        )
        
        if score is None:
            return 0.0, f"Performance evaluation failed: {perf_msg}", ""
        
        print(f"{info_prefix}Final score: {score:.4f}x speedup")
        print(f"{info_prefix}  Iterations: {perf_metadata['num_iterations']}")
        print(f"{info_prefix}  Original avg: {perf_metadata['original_avg_ms']:.3f}ms")
        print(f"{info_prefix}  Custom avg: {perf_metadata['custom_avg_ms']:.3f}ms")
        
        return score, "", ""
        
    except Exception as e:
        error_msg = f"Exception during evaluation: {e}\n{traceback.format_exc()}"
        print(f"{info_prefix}{error_msg}")
        return 0.0, error_msg, ""
    
    finally:
        # Cleanup GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=device)


# =============================================================================
# Helper: Load source from file
# =============================================================================

def load_src_from_file(filepath: str) -> str:
    """Load source code from a file."""
    with open(filepath, 'r') as f:
        return f.read()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    
    # Use custom_code.py which defines ModelNew
    custom_model_path = os.path.join(script_dir, "custom_code.py")
    work_dir = os.path.join(script_dir, "cuda_build")
    
    with open(custom_model_path, 'r') as f:
        code = f.read()
    
    reward, error_msg, details = get_reward(code=code, work_dir=work_dir)
    print(f"\nFinal result: reward={reward:.4f}, error={error_msg}")

