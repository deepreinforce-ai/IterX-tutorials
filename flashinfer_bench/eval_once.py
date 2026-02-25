import atexit
import os
import threading

import modal
from loguru import logger

# --- Modal setup (mirrors agent/modal_eval.py) ---
MODAL_APP_NAME = "flashinfer-bench-eval-once"
VOLUME_NAME = "flashinfer-trace"
DATASET_PATH = "/data"
DATASET_NAME = "mlsys26-contest"

# Default task ID (same as your eval_script.py default)
DEFAULT_TASK_ID = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

# Modal GPU type
GPU_TYPE = os.environ.get("MODAL_GPU", "B200")

app = modal.App(MODAL_APP_NAME)

image = modal.Image.from_registry(
    "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04", add_python="3.13"
).pip_install("flashinfer-bench", "torch", "triton", "pydantic")

dataset_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Global app context management for concurrent get_reward calls
_app_context = None
_app_lock = threading.Lock()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={DATASET_PATH: dataset_vol},
    timeout=600,
    serialized=True,
)
def remote_evaluate(
    kernel_code: str,
    task_id: str,
    dataset_name: str = DATASET_NAME,
    warmup_runs: int = 10,
    iterations: int = 50,
    timeout: int = 300,
) -> dict:
    """Run flashinfer-bench evaluation on Modal GPU."""
    import uuid

    from flashinfer_bench.bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import (BuildSpec, EvaluationStatus, Solution,
                                       SourceFile, SupportedLanguages,
                                       TraceSet)

    dataset_root = os.path.join(DATASET_PATH, dataset_name)
    trace_set = TraceSet.from_path(dataset_root)

    solution_name = f"eval_{uuid.uuid4().hex[:8]}"
    solution = Solution(
        name=solution_name,
        definition=task_id,
        author="eval_once",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=["cuda"],
            entry_point="main.py::run",
            dependencies=[],
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=kernel_code)],
    )

    trace_set.solutions.setdefault(task_id, []).append(solution)
    trace_set._solution_by_name[solution_name] = solution

    config = BenchmarkConfig(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=1,
        definitions=[task_id],
        solutions=[solution_name],
        timeout_seconds=timeout,
    )

    benchmark = Benchmark(trace_set, config)
    try:
        result_ts = benchmark.run_all(dump_traces=False)
    finally:
        benchmark.close()

    traces = result_ts.traces.get(task_id, [])

    error_statuses = {
        EvaluationStatus.COMPILE_ERROR,
        EvaluationStatus.RUNTIME_ERROR,
        EvaluationStatus.INCORRECT_SHAPE,
        EvaluationStatus.INCORRECT_NUMERICAL,
        EvaluationStatus.INCORRECT_DTYPE,
        EvaluationStatus.TIMEOUT,
    }
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status in error_statuses:
            return {
                "compiled": ev.status != EvaluationStatus.COMPILE_ERROR,
                "correct": False,
                "speedup": 0.0,
                "error": f"{ev.status.value}: {ev.log}",
                "task_id": task_id,
            }

    latencies, ref_latencies, speedups = [], [], []
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status == EvaluationStatus.PASSED:
            latencies.append(ev.performance.latency_ms)
            ref_latencies.append(ev.performance.reference_latency_ms)
            speedups.append(ev.performance.speedup_factor)

    if not latencies:
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "error": "No evaluation results",
            "task_id": task_id,
        }

    n = len(latencies)
    return {
        "compiled": True,
        "correct": True,
        "speedup": sum(speedups) / n,
        "latency_ms": sum(latencies) / n,
        "reference_latency_ms": sum(ref_latencies) / n,
        "error": None,
        "task_id": task_id,
    }


def ensure_dataset_synced(local_dataset_root: str):
    """Upload local dataset to Modal Volume if not already present."""
    try:
        if dataset_vol.listdir(f"/{DATASET_NAME}"):
            logger.info(f"Dataset '{DATASET_NAME}' found in Modal Volume, skipping upload")
            return
    except Exception:
        pass

    logger.info(f"Uploading dataset '{DATASET_NAME}' to Modal Volume...")
    with dataset_vol.batch_upload() as batch:
        batch.put_directory(local_dataset_root, f"/{DATASET_NAME}")
    logger.info("Dataset uploaded")


def start_app():
    """Start the Modal app context. Must be called once before any get_reward calls."""
    global _app_context
    with _app_lock:
        if _app_context is not None:
            return
        ctx = app.run()
        _app_context = ctx
        ctx.__enter__()
        atexit.register(stop_app)
        logger.info("Modal app context started")


def stop_app():
    """Stop the Modal app context. Call when all evaluations are done."""
    global _app_context
    with _app_lock:
        if _app_context is None:
            return
        _app_context.__exit__(None, None, None)
        _app_context = None
        logger.info("Modal app context stopped")


def get_reward(
    code: str,
    task_id: str = DEFAULT_TASK_ID,
    warmup_runs: int = 10,
    iterations: int = 50,
) -> tuple[float, bool, str, str]:
    """
    Evaluate kernel code on Modal GPU.
    Automatically starts the Modal app context on first call.
    Safe for concurrent calls — each call gets its own Modal container/GPU.

    Returns:
        (speedup, success, error_msg, details)
    """
    start_app()

    logger.info(f"Sending code to Modal for evaluation (task={task_id})...")
    try:
        eval_result = remote_evaluate.remote(
            kernel_code=code,
            task_id=task_id,
            warmup_runs=warmup_runs,
            iterations=iterations,
        )
    except Exception as e:
        logger.error(f"Modal call failed: {e}")
        return 0.0, False, "Modal call failed", str(e)

    if not eval_result.get("compiled"):
        return 0.0, False, "Code failed to compile", str(eval_result.get("error", ""))
    if not eval_result.get("correct"):
        return 0.0, False, "Code produced incorrect results", str(eval_result.get("error", ""))

    reward = eval_result["speedup"]
    logger.info(f"Code evaluated successfully with reward: {reward}")
    return reward, True, "", ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a Triton kernel on Modal GPU")
    parser.add_argument("--code", type=str, default="init_code.py", help="Path to the code file to evaluate")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_path = os.path.join(script_dir, args.code)
    with open(code_path) as f:
        code_content = f.read()

    local_dataset = os.path.join(script_dir, "datasets", "mlsys26-contest")
    ensure_dataset_synced(local_dataset)
    try:
        reward, success, error_msg, details = get_reward(code_content)
        print(f"Reward: {reward}, Success: {success}, Error: {error_msg}, Details: {details}")
    finally:
        stop_app()
