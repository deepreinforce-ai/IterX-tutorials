"""
Iterx API - Parallel evaluation using ProcessPoolExecutor
Task: Anthropic Performance Engineering Take-Home

Uses isolated temp directories for each evaluation to allow parallel processing.
"""

import sys
import os
import time
import tempfile
import shutil
import uuid
import subprocess
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional

import requests

# Import BASE_URL and API_KEY from anthropic/iterx/run_iterx.py
import importlib.util
_iterx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "anthropic", "iterx", "run_iterx.py")
_spec = importlib.util.spec_from_file_location("iterx_config", _iterx_path)
_iterx_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_iterx_config)
BASE_URL = _iterx_config.BASE_URL
API_KEY = _iterx_config.API_KEY

# ============================================================================
# Configuration
# ============================================================================
task_id = "YOUR_TASK_ID_FROM_CREATE_TASK"
POLL_INTERVAL = 30
REQUEST_TIMEOUT = 30
MAX_WORKERS = 8  # Number of parallel workers

# Source directory (where this script lives)
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

headers = {
    "Authorization": API_KEY,
    "Content-Type": "application/json"
}


def step_1_fetch_all_unevaluated_code_ids(task_id: str) -> tuple[bool, list[str]]:
    try:
        response = requests.post(
            f"{BASE_URL}/api/fetch_unevaluated_code_ids",
            headers=headers,
            json={"task_id": task_id},
            timeout=REQUEST_TIMEOUT
        )
        result = response.json()
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return False, []

    if result["code"] != 200:
        print(f"fetch_unevaluated_code_ids Error: {result.get('message', 'Unknown error')}")
        return False, []

    unevaluated_code_ids = result["data"]["code_id_list"]
    task_is_finished = result["data"]["task_is_finished"]
    return task_is_finished, unevaluated_code_ids


def step_2_fetch_code_content_by_id(code_id: str) -> Optional[str]:
    """Fetch code content by ID"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/get_code_by_id",
            headers=headers,
            json={"code_id": code_id},
            timeout=REQUEST_TIMEOUT
        )
        result = response.json()
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return None

    if result["code"] != 200:
        print(f"get_code_by_id Error: {result.get('message', 'Unknown error')}")
        return None

    code_content = result["data"]["code"]
    return code_content


def step_3_get_reward_isolated(code: str, source_dir: str) -> Tuple[float, str, bool, str]:
    """
    Evaluate code in an isolated temp directory.
    
    This function creates a unique temp directory, copies the test infrastructure,
    writes the code as perf_takehome.py, runs tests, and cleans up.
    
    Args:
        code: The code to evaluate
        source_dir: Path to the source directory containing problem.py and tests/
    
    Returns:
        tuple of (reward, code_error_msg, correctness_check, details)
    """
    BASELINE = 147734
    
    reward = 0.0
    code_error_msg = ""
    correctness_check = True
    details = ""
    achieved_cycles = None
    
    # Create unique temp directory
    temp_dir = tempfile.mkdtemp(prefix=f"iterx_eval_{uuid.uuid4().hex[:8]}_")
    
    try:
        # Create tests subdirectory
        tests_dir = os.path.join(temp_dir, "tests")
        os.makedirs(tests_dir, exist_ok=True)
        
        # Copy required files to temp directory
        # 1. problem.py
        shutil.copy2(
            os.path.join(source_dir, "problem.py"),
            os.path.join(temp_dir, "problem.py")
        )
        
        # 2. tests/frozen_problem.py
        shutil.copy2(
            os.path.join(source_dir, "tests", "frozen_problem.py"),
            os.path.join(tests_dir, "frozen_problem.py")
        )
        
        # 3. tests/submission_tests.py
        shutil.copy2(
            os.path.join(source_dir, "tests", "submission_tests.py"),
            os.path.join(tests_dir, "submission_tests.py")
        )
        
        # Write the code as perf_takehome.py
        perf_takehome_path = os.path.join(temp_dir, "perf_takehome.py")
        with open(perf_takehome_path, "w") as f:
            f.write(code)
        
        # Run only CorrectnessTests (skip SpeedTests to avoid unnecessary failures)
        submission_tests_path = os.path.join(tests_dir, "submission_tests.py")
        result = subprocess.run(
            ["python3", submission_tests_path, "CorrectnessTests"],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for safety
            cwd=temp_dir
        )
        
        stdout = result.stdout
        stderr = result.stderr
        combined_output = stdout + "\n" + stderr
        
        # Try to extract cycle count from output
        # Look for "CYCLES:  <number>" pattern
        cycles_match = re.search(r"CYCLES:\s*(\d+)", combined_output)
        if cycles_match:
            achieved_cycles = int(cycles_match.group(1))
            reward = BASELINE / achieved_cycles
        
        # Check for errors
        if result.returncode != 0:
            # Include full error output for debugging
            code_error_msg = combined_output.strip()
            # Limit to last 2000 chars to avoid overly long messages
            if len(code_error_msg) > 2000:
                code_error_msg = "...\n" + code_error_msg[-2000:]
            
            # If cycles weren't found and there was an error, use penalty
            if achieved_cycles is None:
                reward = 0.0
        
        # Build details string
        if achieved_cycles is not None:
            speedup = BASELINE / achieved_cycles
            details = f"Cycles: {achieved_cycles}, Speedup: {speedup:.2f}x"
        
    except subprocess.TimeoutExpired:
        code_error_msg = "Execution timed out (>120 seconds)"
        reward = 0.0
        
    except Exception as e:
        code_error_msg = f"Execution error: {str(e)}"
        reward = 0.0
        
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_err:
            print(f"Warning: Failed to clean up temp dir {temp_dir}: {cleanup_err}")
    
    return reward, code_error_msg, correctness_check, details


def fetch_and_evaluate_single_code(args: Tuple[str, str, str, str]) -> Tuple[str, float, str, bool, str]:
    """
    Fetch code content and evaluate in one step (for parallel execution).
    
    Args:
        args: tuple of (code_id, base_url, api_key, source_dir)
    
    Returns:
        tuple of (code_id, reward, code_error_msg, correctness_check, details)
    """
    code_id, base_url, api_key, source_dir = args
    
    # Fetch code content
    try:
        response = requests.post(
            f"{base_url}/api/get_code_by_id",
            headers={
                "Authorization": api_key,
                "Content-Type": "application/json"
            },
            json={"code_id": code_id},
            timeout=30
        )
        result = response.json()
    except Exception as e:
        return code_id, 0.0, f"Fetch failed: {e}", True, ""

    if result["code"] != 200:
        return code_id, 0.0, f"Fetch error: {result.get('message', 'Unknown error')}", True, ""

    code_content = result["data"]["code"]
    
    # Evaluate code
    reward, code_error_msg, correctness_check, details = step_3_get_reward_isolated(
        code_content, source_dir
    )
    return code_id, reward, code_error_msg, correctness_check, details


def step_4_submit_score(task_id: str, code_id: str, reward: float, code_error_msg: str, correctness_check: bool, details: str) -> None:
    try:
        response = requests.post(
            f"{BASE_URL}/api/push_code_reward_by_id",
            headers=headers,
            json={
                "task_id": task_id,
                "code_id": code_id,
                "reward": reward,
                "code_error_msg": code_error_msg,
                "correctness_check": correctness_check,
                "details": details
            },
            timeout=REQUEST_TIMEOUT
        )
        result = response.json()
    except Exception as e:
        print(f"Score Submission Error: {e}")
        return

    if result["code"] == 200:
        print(f"  ✓ Score submitted successfully for {code_id}")
    else:
        print(f"push_code_reward_by_id failed: {result.get('message', 'Unknown error')}")


def step_5_get_training_status(task_id: str) -> None:
    try:
        response = requests.post(
            f"{BASE_URL}/api/get_training_status",
            headers=headers,
            json={"task_id": task_id},
            timeout=REQUEST_TIMEOUT
        )
        result = response.json()
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return

    if result["code"] == 200:
        data = result["data"]
        progress = data.get("progress_percentage", 0)
        total_evaluated = data.get("total_evaluated", 0)
        best = data.get("best", {})

        print("\n" + "=" * 70)
        print(f"[Training Status] Progress: {progress:.1f}% | Evaluated: {total_evaluated}")
        if best:
            print(f"[Training Status] Best: code_id={best.get('code_id', 'N/A')}, reward={best.get('reward', 0):.4f}")
        print("=" * 70)
    else:
        print(f" get_training_status failed: {result.get('message', 'Unknown error')}")


def main(task_id: str) -> None:
    print(f"  task_id: {task_id}", flush=True)
    print(f"  max_workers: {MAX_WORKERS}", flush=True)
    print("=" * 70, flush=True)

    while True:
        task_is_finished, unevaluated_code_ids = step_1_fetch_all_unevaluated_code_ids(task_id)
        if task_is_finished:
            print("  [Task is finished]", flush=True)
            break
        print(f"  ✓ Found {len(unevaluated_code_ids)} unevaluated code(s)", flush=True)
        print("-" * 70, flush=True)
        if len(unevaluated_code_ids) == 0:
            print(f"\n[Waiting {POLL_INTERVAL} seconds before next poll...]", flush=True)
            time.sleep(POLL_INTERVAL)
            continue
        
        # Prepare tasks for parallel fetch + evaluate
        print(f"\n[Processing {len(unevaluated_code_ids)} codes in parallel (fetch + evaluate) with {MAX_WORKERS} workers...]", flush=True)
        
        code_tasks = [
            (code_id, BASE_URL, API_KEY, SOURCE_DIR)
            for code_id in unevaluated_code_ids
        ]
        
        # Process codes in parallel using ProcessPoolExecutor
        completed_count = 0
        total_count = len(code_tasks)
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            print(f"  Submitting {total_count} tasks to executor...", flush=True)
            future_to_code_id = {
                executor.submit(fetch_and_evaluate_single_code, task): task[0]
                for task in code_tasks
            }
            print(f"  All tasks submitted. Waiting for results...", flush=True)
            
            # Process results as they complete
            for future in as_completed(future_to_code_id):
                code_id = future_to_code_id[future]
                completed_count += 1
                try:
                    result_code_id, reward, code_error_msg, correctness_check, details = future.result()
                    print(f"\n[{completed_count}/{total_count}] Eval Complete: {result_code_id}", flush=True)
                    print(f"  Reward: {reward:.4f}", flush=True)
                    if code_error_msg:
                        print(f"  Error:\n{code_error_msg}", flush=True)
                    if details:
                        print(f"  Details: {details}", flush=True)
                    
                    # Submit score
                    step_4_submit_score(task_id, result_code_id, reward, code_error_msg, correctness_check, details)
                    
                except Exception as e:
                    print(f"\n[{completed_count}/{total_count}] Eval Error: {code_id}", flush=True)
                    print(f"  Exception: {e}", flush=True)
                    # Submit error score
                    step_4_submit_score(task_id, code_id, 0.0, str(e), True, "")

        # Get training status
        step_5_get_training_status(task_id)


if __name__ == "__main__":
    # Create new task using api if task_id is not set
    if task_id == "YOUR_TASK_ID_FROM_CREATE_TASK":
        # Import create_task from run_iterx.py
        from run_iterx import create_task
        task_id = create_task()
    main(task_id)
