"""
Iterx API - Create task and run evaluation loop
Task: Fast Sort - Optimize sorting for unknown data distribution
"""

import os
import time

import requests
import yaml
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================

# Load config from tutorials/config.yaml
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

BASE_URL = config["BASE_URL"]
API_KEY = config["API_KEY"]
# task_id = "3dda6e35-493c-4729-8e64-38ae1cd318be"
task_id = "YOUR_TASK_ID_FROM_CREATE_TASK"
POLL_INTERVAL = 30
REQUEST_TIMEOUT = 30

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
    # Get code content
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


def step_3_get_reward(code: str) -> tuple[float, str, bool, str]:
    """
    Evaluate a sorting function.
    
    Args:
        code: Python code string defining a function named 'mysort'

    def mysort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return mysort(left) + middle + mysort(right)
        
    Returns:
        (reward, code_error_msg, correctness_check, details)
        - reward: inverse of execution time (0 if failed)
        - code_error_msg: exception message if failed
        - correctness_check: True if result matches numpy's sorted
        - details: additional info about the evaluation
    """
    import numpy as np
    import os
    
    # Load test data from compressed numpy format
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(script_dir, "sort_test_data.npz")
    
    test_data = np.load(test_data_path)['data']
    
    # Execute the code to get the mysort function
    try:
        namespace = {'np': np, 'numpy': np}  # Provide numpy in namespace
        exec(code, namespace)
        
        if 'mysort' not in namespace:
            code_error_msg = "Code must define a function named 'mysort'"
            reward = 0
            correctness_check = False
            return reward, code_error_msg, correctness_check, ""
        
        mysort = namespace['mysort']
        
        if not callable(mysort):
            code_error_msg = "'mysort' must be a callable function"
            reward = 0
            correctness_check = False
            return reward, code_error_msg, correctness_check, ""
    except Exception as e:
        code_error_msg = f"Failed to execute code: {e}"
        reward = 0
        correctness_check = False
        return reward, code_error_msg, correctness_check, ""
    
    # First run: check correctness
    test_copy = test_data.copy()
    try:
        result = mysort(test_copy)
    except Exception as e:
        code_error_msg = f"Runtime error: {e}"
        correctness_check = False
        return 0.0, code_error_msg, correctness_check, ""
    
    # Extract actual values from result
    if result is None:
        result_values = test_copy
    else:
        result_values = np.asarray(result)
    
    # Compare with numpy's sorted
    expected = np.sort(test_data)
    
    if not np.array_equal(result_values, expected):
        reward = 0
        code_error_msg = ""
        correctness_check = False
        details = f"Sorting incorrect. First mismatch at index {np.argmax(result_values != expected)}"
        return reward, code_error_msg, correctness_check, details
    
    # Run 20 times and compute average time
    NUM_RUNS = 20
    times = []
    for _ in range(NUM_RUNS):
        test_copy = test_data.copy()
        start_time = time.perf_counter()
        try:
            result = mysort(test_copy)
        except Exception as e:
            code_error_msg = f"Runtime error: {e}"
            correctness_check = False
            return 0.0, code_error_msg, correctness_check, ""
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate reward: inverse of average execution time (faster = higher reward)
    reward = 1.0 / avg_time
    code_error_msg = ""
    correctness_check = True
    details = f"Sorted {len(test_data):,} items in avg {avg_time*1000:.2f} ms (min: {min_time*1000:.2f}, max: {max_time*1000:.2f}, runs: {NUM_RUNS})"
    
    return reward, code_error_msg, correctness_check, details


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
        print(f"  ✓ Score submitted successfully.")
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


# ============================================================================
# Task Creation Configuration (if you already have a task_id, ignore this)
# ============================================================================

def create_task() -> str:
    """Create a new task and return the task_id"""
    TASK_NAME = "fast sort"
    TASK_DESCRIPTION = """Implement a fast sorting function called 'mysort' that sorts a numpy array of integers.
The input array has 500,000 items with an unknown distribution that has exploitable structure.
The data may have clusters, partial ordering, or other patterns that can be leveraged.
You have access to numpy (as 'np' or 'numpy') in your function. Feel free to use numba for performance.
The function should return the sorted array."""
    REWARD_DESCRIPTION = "The inverse of execution time in seconds (faster = higher reward)."
    INITIAL_CODE = """import numpy as np
def mysort(arr):
    # Basic quicksort implementation - can you beat numpy's built-in sort?
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = arr[arr < pivot]
    middle = arr[arr == pivot]
    right = arr[arr > pivot]
    return np.concatenate([mysort(left), middle, mysort(right)])"""
    SAMPLE_SIZE = 2
    MODEL = "Qwen3-235B-A22B"

    print("\n[Creating New Task]")
    print("=" * 70)

    try:
        response = requests.post(
            f"{BASE_URL}/api/task/create",
            headers=headers,
            json={
                "task_name": TASK_NAME,
                "task_description": TASK_DESCRIPTION,
                "reward_description": REWARD_DESCRIPTION,
                "initial_code": INITIAL_CODE,
                "sample_size": SAMPLE_SIZE,
                "model": MODEL
            },
            timeout=REQUEST_TIMEOUT
        )
        result = response.json()
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        raise

    if result.get("code") != 200:
        print(f"  ✗ Error creating task: {result.get('message', result)}")
        raise Exception(f"Failed to create task: {result.get('message', result)}")

    new_task_id = result["data"]["task_id"]
    print(f"  ✓ Task created successfully!")
    print(f"  Task ID: {new_task_id}")
    print("=" * 70)
    return new_task_id


def main(task_id: str) -> None:
    print(f"  task_id: {task_id}")
    print("=" * 70)

    while True:
        task_is_finished, unevaluated_code_ids = step_1_fetch_all_unevaluated_code_ids(task_id)
        if task_is_finished:
            print("  [Task is finished]")
            break
        print(f"  ✓ Found {len(unevaluated_code_ids)} unevaluated code(s)")
        print("-" * 70)
        if len(unevaluated_code_ids) == 0:
            print(f"\n[Waiting {POLL_INTERVAL} seconds before next poll...]")
            time.sleep(POLL_INTERVAL)
            continue
        # Process each unevaluated code
        # Feel free to switch to a process pool or thread pool for parallel processing.
        for index, code_id in enumerate(unevaluated_code_ids, 1):
            print(f"\n[{index}/{len(unevaluated_code_ids)}] Evaluating code: {code_id}")
            print("-" * 70)
            code_content = step_2_fetch_code_content_by_id(code_id)
            if code_content is None:
                continue
            # Evaluate code locally
            reward, code_error_msg, correctness_check, details = step_3_get_reward(code_content)
            # Submit score
            step_4_submit_score(task_id, code_id, reward, code_error_msg, correctness_check, details)

        # Get training status
        step_5_get_training_status(task_id)



if __name__ == "__main__":
    # Create new task using api if task_id is not set
    if task_id == "YOUR_TASK_ID_FROM_CREATE_TASK":
        task_id = create_task()
    main(task_id)

