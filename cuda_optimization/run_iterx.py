"""
Iterx API - Create task and run evaluation loop
Task: CUDA Kernel Optimization
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
import yaml
import requests

from eval_cuda_server import get_reward

# ============================================================================
# Configuration
# ============================================================================

config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

BASE_URL = config["BASE_URL"]
API_KEY = config["API_KEY"]
POLL_INTERVAL = 30
REQUEST_TIMEOUT = 30

headers = {
    "Authorization": API_KEY,
    "Content-Type": "application/json"
}


def create_task():
    """Create task and return task_id"""
    # Load initial code from file
    initial_code_path = os.path.join(os.path.dirname(__file__), "initial_code.py")
    with open(initial_code_path, "r") as f:
        initial_code = f.read()
    
    response = requests.post(
        f"{BASE_URL}/api/task/create",
        headers=headers,
        json={
            "task_name": "CUDA Kernel Optimization",
            "task_description": """**Goal:** Design a CUDA kernel that accelerates the reference PyTorch operation. 

**Input:** 
- Reference model in `initial_code.py` defining `Model`, `get_inputs()`, `get_init_inputs()`
- Your optimized implementation defining `ModelNew`

**Output:**
- Numerically equivalent results to the reference
- Faster execution time

**Requirements:**
1. Define a `ModelNew` class with the same interface as the reference `Model`
2. Implement `forward()` method with identical input/output signature
3. Keep `get_inputs()` and `get_init_inputs()` functions exactly as in the reference
4. Maintain all hyperparameters (e.g., `M = 4096`) exactly as in the reference
5. Output must match reference within tolerance (atol=1e-2, rtol=1e-2)
6. No timing cheats (thread/stream/lazy injection)
7. Your code must be in Python.

**Things to Avoid:**
1. Creating new CUDA streams for computation
2. Spawning background threads
3. Returning lazy tensor subclasses
4. Infinite loops or blocking operations
5. Hardcoded test cases (must work for general inputs)

**Writing CUDA in Python:**
You can write CUDA kernels directly in Python files using `torch.utils.cpp_extension.load_inline`. This allows you to embed CUDA C++ code as a string and compile it at runtime.""",
            "reward_description": "Speedup ratio: reward = original_time / custom_time. Higher is better. Returns 0 if correctness tests fail or security checks are violated.",
            "initial_code": initial_code,
            "sample_size": 2,
            "model": "Qwen3-235B-A22B"
        },
        timeout=REQUEST_TIMEOUT
    )
    return response.json()["data"]["task_id"]


def step_1_fetch_all_unevaluated_code_ids(task_id):
    try:
        print(f"{BASE_URL}/api/fetch_unevaluated_code_ids")
        response = requests.post(
            f"{BASE_URL}/api/fetch_unevaluated_code_ids",
            headers=headers,
            json={"task_id": task_id},
            timeout=REQUEST_TIMEOUT
        )
        result = response.json()
    except Exception as e:
        print(f"error {e}")
        return False, []
    
    if result["code"] != 200:
        print(f"fetch_unevaluated_code_ids Error: {result['message']}")
        return False, []
    
    unevaluated_code_ids = result["data"]["code_id_list"]
    task_is_finished = result["data"]["task_is_finished"]
    return task_is_finished, unevaluated_code_ids


def step_2_fetch_code_content_by_id(code_id):
    # Get code content
    try:
        response = requests.post(
            f"{BASE_URL}/api/get_code_by_id",
            headers=headers,
            json={"code_id": code_id},
            timeout=REQUEST_TIMEOUT
        )
        result = response.json()
    except:
        return None
    
    if result["code"] != 200:
        print(f"get_code_by_id Error: {result['message']}")
        return None
    
    code_content = result["data"]["code"]
    return code_content


def step_4_submit_score(task_id, code_id, reward, code_error_msg, correctness_check, details):
    try:
        print(f"details {details}")
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
        print(f"push_code_reward_by_id failed: {result['message']}")


def step_5_get_training_status(task_id):
    try:
        response = requests.post(
            f"{BASE_URL}/api/get_training_status",
            headers=headers,
            json={"task_id": task_id},
            timeout=REQUEST_TIMEOUT
        )
        result = response.json()
    except:
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
        print(f" get_training_status failed: {result['message']}")


def main(task_id):
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
        
        # Process each unevaluated code using thread pool
        def evaluate_single_code(args):
            index, code_id = args
            print(f"\n[{index}/{len(unevaluated_code_ids)}] Evaluating code: {code_id}")
            print("-" * 70)
            code_content = step_2_fetch_code_content_by_id(code_id)
            if code_content is None:
                return
            # Evaluate code locally
            try:
                work_dir = f"/data/iterx/cuda_optimization/{code_id}"
                eval_url = f"http://localhost:{5000+index}"
                reward, code_error_msg, details = get_reward(code_content, work_dir=work_dir, eval_url=eval_url, device_index=index%8)
            except Exception as e:
                print(f"error {e}")
                reward, code_error_msg, details = 0, str(e), ""
            correctness_check = not code_error_msg
            # Submit score
            step_4_submit_score(task_id, code_id, reward, code_error_msg, correctness_check, details)
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(evaluate_single_code, enumerate(unevaluated_code_ids, 1))
        
        # Get training status
        step_5_get_training_status(task_id)
            

def get_or_create_task_id():
    """Load task_id from task_id.txt if exists, otherwise create new task."""
    task_id_path = os.path.join(os.path.dirname(__file__), "task_id.txt")
    
    if os.path.exists(task_id_path):
        try:
            with open(task_id_path, "r") as f:
                task_id = f.read().strip()
                if task_id:
                    print(f"  Loaded existing task_id from {task_id_path}")
                    return task_id
        except Exception as e:
            print(f"  Failed to load task_id: {e}")
    
    # Create new task
    print("  Creating new task...")
    task_id = create_task()
    
    # Save task_id
    with open(task_id_path, "w") as f:
        f.write(task_id)
    print(f"  Saved task_id to {task_id_path}")
    
    return task_id

if __name__ == "__main__":
    task_id = get_or_create_task_id()
    # task_id = "2c8c0e79-4158-47fb-a566-cbd9c7c19c11"
    main(task_id)
