"""
CUDA Optimization Flask Evaluation Server

Flask server that evaluates custom CUDA kernels by:
1. Compiling and loading both original and custom models
2. Defending against thread injection, stream injection, and lazy evaluation attacks
3. Checking correctness against the reference implementation
4. Measuring performance (speedup = original_time / custom_time)

Server writes progress to a log file (JSON lines) for polling by the client.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

app = Flask(__name__)
CORS(app)


def write_log(log_path: str, data: dict):
    """Append a JSON line to the log file"""
    if log_path and log_path != "/dev/null":
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    import torch
    
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
            })
    
    return jsonify({
        'status': 'healthy',
        'service': 'CUDA Optimization Evaluation Server',
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpus': gpu_info
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate a custom CUDA kernel against the original implementation.
    
    Expected JSON payload:
    {
        "code_path": "/path/to/custom_model.py",
        "work_dir": "/path/to/work_dir",
        "log_path": "/path/to/result.json",
        "info_string": "your_info",
        "device_index": 0,
        "warmup_time": 5.0,
        "run_time": 10.0
    }
    """
    data = request.json
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Extract parameters
    code_path = data.get('code_path')
    work_dir = data.get('work_dir')
    log_path = data.get('log_path', '/dev/null')
    info_string = data.get('info_string', '')
    device_index = data.get('device_index', 0)
    warmup_time = data.get('warmup_time', 5.0)
    run_time = data.get('run_time', 10.0)
    num_correct_trials = data.get('num_correct_trials', 5)
    
    # Validate required fields
    if not code_path:
        return jsonify({'error': 'Missing code_path'}), 400
    if not work_dir:
        return jsonify({'error': 'Missing work_dir'}), 400
    
    # Ensure directories exist
    os.makedirs(work_dir, exist_ok=True)
    
    # Clear existing log file
    if log_path and log_path != "/dev/null" and os.path.exists(log_path):
        os.remove(log_path)
    
    info_prefix = f"[{info_string}] " if info_string else ""
    
    try:
        # Write start time
        write_log(log_path, {
            'start_time': time.time(),
            'info_string': info_string,
            'code_path': code_path
        })
        
        # Check if code file exists
        if not os.path.exists(code_path):
            write_log(log_path, {
                'done': True,
                'error': True,
                'error_msg': f'Code file not found: {code_path}'
            })
            return jsonify({
                'success': False,
                'error': f'Code file not found: {code_path}'
            }), 400
        
        # Import evaluation functions
        import torch
        import torch.nn as nn
        import threading
        import random
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        
        if not torch.cuda.is_available():
            write_log(log_path, {
                'done': True,
                'error': True,
                'error_msg': 'CUDA not available'
            })
            return jsonify({
                'success': False,
                'error': 'CUDA not available'
            }), 500
        
        device = torch.device(f'cuda:{device_index}')
        torch.cuda.set_device(device)
        
        # Setup build directory
        build_dir = os.path.join(work_dir, "cuda_build")
        os.makedirs(build_dir, exist_ok=True)
        
        seed = 42
        
        write_log(log_path, {'evaluation_started': time.time()})
        
        # =====================================================================
        # Helper functions (inline to keep server self-contained)
        # =====================================================================
        
        def set_seed(s):
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
        
        def load_src_from_file(filepath):
            with open(filepath, 'r') as f:
                return f.read()
        
        def execute_model_with_timeout(model_src, context, timeout=300.0, build_directory=None):
            if build_directory:
                context["BUILD_DIRECTORY"] = build_directory
                model_src = (
                    "import os\n"
                    f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
                ) + model_src
            
            # Static analysis for blocking patterns
            blocking_patterns = [
                ('time.sleep(', 'time.sleep() calls'),
                ('input(', 'user input'),
                ('while True:', 'infinite loops'),
            ]
            
            for pattern, description in blocking_patterns:
                if pattern in model_src:
                    return False, f"Code contains blocking pattern: {description}", None
            
            def _execute_code():
                compile(model_src, "<string>", "exec")
                exec(model_src, context)
                return True
            
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_execute_code)
                    t1 = time.time()
                    future.result(timeout=timeout)
                    t2 = time.time()
                    return True, "", t2 - t1
            except FuturesTimeoutError:
                return False, f"Execution timeout", None
            except SyntaxError as e:
                return False, f"Syntax Error: {e}", None
            except Exception as e:
                return False, f"Runtime Error: {e}", None
        
        # =====================================================================
        # Step 1: Load original model
        # =====================================================================
        print(f"{info_prefix}Step 1: Loading original model...")
        
        # Load original model from initial_code.py in the same directory
        script_dir = Path(__file__).parent.resolve()
        original_model_path = script_dir / "initial_code.py"
        
        if not original_model_path.exists():
            error_msg = f"Original model file not found: {original_model_path}"
            write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
            return jsonify({'success': False, 'error': error_msg}), 500
        
        original_model_src = load_src_from_file(str(original_model_path))
        context_original = {}
        success, error_msg, _ = execute_model_with_timeout(
            original_model_src, context_original, timeout=60.0
        )
        
        if not success:
            write_log(log_path, {'done': True, 'error': True, 'error_msg': f"Failed to load original model: {error_msg}"})
            return jsonify({'success': False, 'error': f"Failed to load original model: {error_msg}"}), 500
        
        Model = context_original.get("Model")
        get_init_inputs = context_original.get("get_init_inputs")
        get_inputs = context_original.get("get_inputs")
        
        if not all([Model, get_init_inputs, get_inputs]):
            error_msg = "Original model missing required definitions (Model, get_init_inputs, get_inputs)"
            write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
            return jsonify({'success': False, 'error': error_msg}), 500
        
        # Initialize original model
        set_seed(seed)
        init_inputs = get_init_inputs()
        init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]
        
        with torch.no_grad():
            set_seed(seed)
            original_model = Model(*init_inputs).to(device)
        
        write_log(log_path, {'model_loading_done': time.time(), 'stage': 'original_loaded'})
        
        # =====================================================================
        # Step 2: Load custom model
        # =====================================================================
        print(f"{info_prefix}Step 2: Loading custom model...")
        
        custom_model_src = load_src_from_file(code_path)
        
        # If ModelNew is not defined but Model is, rename Model to ModelNew
        import re
        if 'class ModelNew' not in custom_model_src and 'class Model' in custom_model_src:
            print(f"{info_prefix}  Renaming 'Model' to 'ModelNew' in custom code...")
            # Replace class definition
            custom_model_src = re.sub(r'\bclass Model\b', 'class ModelNew', custom_model_src)
            # Replace Model references (but not ModelNew)
            custom_model_src = re.sub(r'\bModel\b(?!New)', 'ModelNew', custom_model_src)
        
        context_custom = {}
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        
        success, error_msg, _ = execute_model_with_timeout(
            custom_model_src, context_custom, timeout=120.0, build_directory=build_dir
        )
        
        if not success:
            write_log(log_path, {'done': True, 'error': True, 'error_msg': f"Failed to load custom model: {error_msg}"})
            return jsonify({'success': False, 'error': f"Failed to load custom model: {error_msg}"}), 500
        
        ModelNew = context_custom.get("ModelNew")
        
        if ModelNew is None:
            error_msg = "ModelNew was not defined in custom model"
            write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
            return jsonify({'success': False, 'error': error_msg}), 500
        
        # Initialize custom model
        with torch.no_grad():
            set_seed(seed)
            custom_model = ModelNew(*init_inputs).to(device)
        
        torch.cuda.synchronize(device=device)
        
        write_log(log_path, {'model_loading_done': time.time(), 'stage': 'custom_loaded'})
        
        # =====================================================================
        # Step 3: Defense checks
        # =====================================================================
        print(f"{info_prefix}Step 3: Running defense checks...")
        
        set_seed(seed)
        test_inputs = get_inputs()
        test_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in test_inputs]
        
        # Defense: Lazy evaluation check
        with torch.no_grad():
            output = custom_model(*test_inputs)
        
        outputs_to_check = [output] if isinstance(output, torch.Tensor) else output
        if isinstance(output, (tuple, list)):
            outputs_to_check = output
        
        for idx, out in enumerate(outputs_to_check):
            if not isinstance(out, torch.Tensor):
                error_msg = f"Output[{idx}] is not a tensor: {type(out)}"
                write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
                return jsonify({'success': False, 'error': error_msg}), 500
            
            if type(out).__name__ not in ['Tensor', 'Parameter']:
                error_msg = f"Output[{idx}] is {type(out).__name__}, not standard torch.Tensor (lazy evaluation)"
                write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
                return jsonify({'success': False, 'error': error_msg}), 500
            
            try:
                if out.untyped_storage().size() == 0:
                    error_msg = f"Output[{idx}] has no allocated storage (lazy evaluation)"
                    write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
                    return jsonify({'success': False, 'error': error_msg}), 500
            except Exception as e:
                error_msg = f"Cannot access storage: {e}"
                write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
                return jsonify({'success': False, 'error': error_msg}), 500
        
        print(f"{info_prefix}  Lazy evaluation check: PASSED")
        
        # Defense: Thread injection check
        before = threading.active_count()
        with torch.no_grad():
            _ = custom_model(*test_inputs)
        torch.cuda.synchronize()
        after = threading.active_count()
        
        if after > before:
            error_msg = f"Thread injection detected (before={before}, after={after})"
            write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
            return jsonify({'success': False, 'error': error_msg}), 500
        
        print(f"{info_prefix}  Thread injection check: PASSED")
        
        # Defense: Stream injection check
        with torch.no_grad():
            _ = custom_model(*test_inputs)
        torch.cuda.synchronize(device=device)
        
        num_iterations = 10
        original_times = []
        defense_times = []
        
        for _ in range(num_iterations):
            # Original timing
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                _ = custom_model(*test_inputs)
            e.record()
            torch.cuda.synchronize(device=device)
            original_times.append(s.elapsed_time(e))
            
            # Defense timing (sync before end event)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                _ = custom_model(*test_inputs)
            torch.cuda.synchronize(device=device)
            e.record()
            torch.cuda.synchronize(device=device)
            defense_times.append(s.elapsed_time(e))
        
        original_time_ms = sorted(original_times)[num_iterations // 2]
        defense_time_ms = sorted(defense_times)[num_iterations // 2]
        
        if original_time_ms < 0.001:
            ratio = float('inf')
        else:
            ratio = defense_time_ms / original_time_ms
        
        if ratio > 1.5:
            error_msg = f"Stream injection detected (ratio={ratio:.2f})"
            write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
            return jsonify({'success': False, 'error': error_msg}), 500
        
        print(f"{info_prefix}  Stream injection check: PASSED (ratio={ratio:.2f})")
        
        write_log(log_path, {'defense_checks_done': time.time()})
        
        # =====================================================================
        # Step 4: Correctness check
        # =====================================================================
        print(f"{info_prefix}Step 4: Checking correctness...")
        
        torch.manual_seed(seed)
        trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)]
        
        with torch.no_grad():
            for trial in range(num_correct_trials):
                trial_seed = trial_seeds[trial]
                
                set_seed(trial_seed)
                inputs = get_inputs()
                inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
                
                set_seed(trial_seed)
                original_model.eval()
                original_output = original_model(*inputs)
                torch.cuda.synchronize(device=device)
                
                set_seed(trial_seed)
                custom_model.eval()
                custom_output = custom_model(*inputs)
                torch.cuda.synchronize(device=device)
                
                if original_output.shape != custom_output.shape:
                    error_msg = f"Shape mismatch: expected {original_output.shape}, got {custom_output.shape}"
                    write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
                    return jsonify({'success': False, 'error': error_msg}), 500
                
                if not torch.allclose(original_output, custom_output, atol=1e-2, rtol=1e-2):
                    max_diff = torch.max(torch.abs(original_output - custom_output)).item()
                    error_msg = f"Value mismatch in trial {trial+1}: max_diff={max_diff:.6f}"
                    write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
                    return jsonify({'success': False, 'error': error_msg}), 500
        
        print(f"{info_prefix}  Correctness check: PASSED ({num_correct_trials} trials)")
        
        write_log(log_path, {'correctness_tests_done': time.time()})
        
        # =====================================================================
        # Step 5: Performance evaluation
        # =====================================================================
        print(f"{info_prefix}Step 5: Evaluating performance...")
        
        set_seed(seed)
        inputs = get_inputs()
        inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        # Warmup
        print(f"{info_prefix}Warming up for {warmup_time}s...")
        warmup_start = time.time()
        while time.time() - warmup_start < warmup_time:
            with torch.no_grad():
                _ = original_model(*inputs)
                _ = custom_model(*inputs)
                torch.cuda.synchronize(device=device)
        
        # Performance measurement
        print(f"{info_prefix}Running performance measurement for {run_time}s...")
        original_times = []
        custom_times = []
        
        run_start = time.time()
        iteration = 0
        
        with torch.no_grad():
            while time.time() - run_start < run_time:
                set_seed(seed + iteration)
                inputs = get_inputs()
                inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
                
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
            error_msg = "No performance measurements collected"
            write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
            return jsonify({'success': False, 'error': error_msg}), 500
        
        avg_original = sum(original_times) / len(original_times)
        avg_custom = sum(custom_times) / len(custom_times)
        
        if avg_custom <= 0:
            error_msg = "Invalid custom model timing"
            write_log(log_path, {'done': True, 'error': True, 'error_msg': error_msg})
            return jsonify({'success': False, 'error': error_msg}), 500
        
        score = avg_original / avg_custom
        
        print(f"{info_prefix}Final score: {score:.4f}x speedup")
        print(f"{info_prefix}  Iterations: {iteration}")
        print(f"{info_prefix}  Original avg: {avg_original:.3f}ms")
        print(f"{info_prefix}  Custom avg: {avg_custom:.3f}ms")
        
        # Write final result
        write_log(log_path, {
            'done': True,
            'error': False,
            'reward': float(score),
            'original_avg_ms': avg_original,
            'custom_avg_ms': avg_custom,
            'num_iterations': iteration
        })
        
        return jsonify({
            'success': True,
            'reward': float(score),
            'original_avg_ms': avg_original,
            'custom_avg_ms': avg_custom,
            'num_iterations': iteration
        })
        
    except Exception as e:
        error_msg = f'Exception: {type(e).__name__} {str(e)}\n{traceback.format_exc()}'
        print(f"{info_prefix}{error_msg}")
        
        write_log(log_path, {
            'done': True,
            'error': True,
            'error_msg': error_msg,
            'reward': 0.0
        })
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'reward': 0.0
        }), 500
    
    finally:
        # Cleanup GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=device)


@app.route('/api/gpus', methods=['GET'])
def get_gpus():
    """Get available GPU information"""
    import torch
    
    if not torch.cuda.is_available():
        return jsonify({
            'available': False,
            'count': 0,
            'gpus': []
        })
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_info.append({
            'id': i,
            'name': torch.cuda.get_device_name(i),
            'memory_total_gb': round(props.total_memory / (1024**3), 2),
            'memory_allocated_gb': round(torch.cuda.memory_allocated(i) / (1024**3), 2),
            'memory_reserved_gb': round(torch.cuda.memory_reserved(i) / (1024**3), 2),
            'compute_capability': f'{props.major}.{props.minor}'
        })
    
    return jsonify({
        'available': True,
        'count': len(gpu_info),
        'gpus': gpu_info
    })


@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'service': 'CUDA Optimization Evaluation Server',
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check with GPU info',
            'GET /api/gpus': 'Detailed GPU information',
            'POST /api/evaluate': {
                'description': 'Evaluate a custom CUDA kernel',
                'request_json': {
                    'code_path': 'Path to custom model Python file (required)',
                    'work_dir': 'Working directory for CUDA builds (required)',
                    'log_path': 'Path to write progress log (optional)',
                    'info_string': 'Identifier for logging (optional)',
                    'device_index': 'GPU index to use (default: 0)',
                    'warmup_time': 'Warmup duration in seconds (default: 5.0)',
                    'run_time': 'Benchmark duration in seconds (default: 10.0)'
                }
            }
        }
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CUDA Optimization Evaluation Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args, unknown = parser.parse_known_args()
    
    # Also support positional port argument for compatibility
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        args.port = int(sys.argv[1])
    
    print("=" * 60)
    print("  CUDA Optimization Evaluation Server")
    print("=" * 60)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print()
    
    # Print GPU info
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA Available: Yes")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    [{i}] {torch.cuda.get_device_name(i)}")
        else:
            print("  CUDA Available: No")
    except ImportError:
        print("  PyTorch not available")
    
    print()
    print("  Endpoints:")
    print("    GET  /         - API documentation")
    print("    GET  /health   - Health check")
    print("    GET  /api/gpus - GPU information")
    print("    POST /api/evaluate - Evaluate custom CUDA kernel")
    print("=" * 60)
    print()
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
        threaded=True
    )

