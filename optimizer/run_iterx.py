"""
Iterx API - Create task and run evaluation loop
Task: Design an Optimizer Better Than Adam
"""

import os
import time

import requests
import yaml

from eval_optimizer import get_reward

# ============================================================================
# Configuration
# ============================================================================

# Load config from tutorials/config.yaml
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
            "task_name": "Design an Optimizer Better Than Adam",
            "task_description": """**Goal:** Design an optimization algorithm that outperforms Adam on CIFAR-10 with ResNet-18.

**Interface:**
- Your optimizer must be a Python class with `zero_grad()` and `step()` methods
- Constructor should accept `params` (model parameters) and `lr` (learning rate)

**Requirements:**
1. Must implement standard PyTorch optimizer interface
2. Must handle gradient updates without causing NaN/Inf values
3. Should work with default learning rate of 0.001

**Things to Avoid:**
- Excessive memory usage
- Non-finite gradient values""",
            "reward_description": "Combined score: reward = (1/3) * inverted_auc + (2/3) * final_test_accuracy. Range [0.0, 1.0], higher is better. Inverted AUC rewards fast convergence, final accuracy rewards generalization.",
            "initial_code": initial_code,
            "sample_size": 2,
            "model": "Qwen3-235B-A22B"
        },
        timeout=REQUEST_TIMEOUT
    )
    return response.json()["data"]["task_id"]


def step_1_fetch_all_unevaluated_code_ids(task_id):
    try:
        response = requests.post(
            f"{BASE_URL}/api/fetch_unevaluated_code_ids",
            headers=headers,
            json={"task_id": task_id},
            timeout=REQUEST_TIMEOUT
        )
        result = response.json()
    except Exception as e:
        print(f"Error: {e}")
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
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    if result["code"] != 200:
        print(f"get_code_by_id Error: {result['message']}")
        return None
    
    code_content = result["data"]["code"]
    return code_content


def step_3_get_reward(code):
    # Use the get_reward function from eval_optimizer.py
    try:
        reward, error_msg, details = get_reward(code)
        correctness_check = not error_msg
        
        # API requires: if correctness_check is false, reward must be 0
        if not correctness_check:
            reward = 0
        
        return reward, error_msg, correctness_check, details
    except Exception as e:
        return 0, str(e), False, ""


def step_4_submit_score(task_id, code_id, reward, code_error_msg, correctness_check, details):
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
    except Exception as e:
        print(f"Error: {e}")
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
            print(f"[Eval] Score: {reward} for code with ID {code_id}")
            if code_error_msg:
                print(f"[Eval] Error: {code_error_msg}")
            # Submit score
            step_4_submit_score(task_id, code_id, reward, code_error_msg, correctness_check, details)
        
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
    print(f"task_id {task_id}")
    # Save task_id
    with open(task_id_path, "w") as f:
        f.write(task_id)
    print(f"  Saved task_id to {task_id_path}")
    
    return task_id


if __name__ == "__main__":
    task_id = get_or_create_task_id()
    main(task_id)