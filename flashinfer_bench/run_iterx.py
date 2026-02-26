"""
Iterx API - Create task and run evaluation loop
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from loguru import logger

from eval_once import get_reward

# ============================================================================
# Configuration
# ============================================================================

# BASE_URL = "http://10.30.16.111:18085"
BASE_URL = "https://iterx.deep-reinforce.com"
API_KEY = "YOUR-API-KEY"
POLL_INTERVAL = 10
REQUEST_TIMEOUT = 30

headers = {
    "Authorization": API_KEY,
    "Content-Type": "application/json"
}
GPU_TYPE = os.environ.get("MODAL_GPU", "B200")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="DeepSeek-V3-thinking or Qwen3-235B-A22B")
parser.add_argument("--sample_size", type=int, required=True, help="Sample size for each evaluation")

args = parser.parse_args()


def create_task(initial_code_file_name):
    """Create task and return task_id"""
    # Load initial code from file
    initial_code_path = os.path.join(os.path.dirname(__file__), initial_code_file_name)
    with open(initial_code_path, "r") as f:
        initial_code = f.read()
    with open("definition.json", "r") as f:
        definition = json.loads(f.read())
    definition_str = json.dumps(definition, indent=4)
    task_description = f"""
## Problem Statement

You write custom kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace.
You may make the decision to replace some operators with custom kernels and leave others unchanged.
You may replace multiple operators with custom implementations,
consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu),
or algorithmic changes (such as online softmax). You are only limited by your imagination.

The goal is to get the best performance for the given architecture.

Generate a Triton kernel optimized for {GPU_TYPE} GPU for the problem defined below:

```json
{definition_str}
```

Triton Version: 3.3.1

Requirements:
- Write clean, efficient Triton code optimized for {GPU_TYPE} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your Triton implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {GPU_TYPE} GPU characteristics (memory hierarchy, compute units, etc.)

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- All code must be valid Python that passes ast.parse()

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the code, no explanations or markdown formatting

Generate complete, runnable code only - no framework will add device handling wrapper code.

            """

    response = requests.post(
        f"{BASE_URL}/api/task/create",
        headers=headers,
        json={
            "task_name": f"flashinfer_bench_moe",
            "task_description": task_description,
            "reward_description": """The reward is the Speedup Ratio, which means `reward = BASELINE_LATENCY / CODE_LATENCY`""",
            "initial_code": initial_code,
            "sample_size": args.sample_size,
            "model": "DeepSeek-V3-thinking"
        },
        timeout=REQUEST_TIMEOUT
    )
    print(response.json())
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
    except:
        return False, []

    if result["code"] != 200:
        print(f"  ✗ Error: {result['message']}")
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
        print(f"  ✗ Error: {result['message']}")
        return None

    code_content = result["data"]["code"]
    return code_content


def step_3_get_reward(code):
    try:
        reward, correctness_check, error_msg, details = get_reward(code)
        if not correctness_check:
            assert reward == 0
        return reward, error_msg, correctness_check, details
    except Exception as e:
        logger.error(f"Exception during reward evaluation: {e}")
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
    except:
        print("Submit score failed due to network error.")
        return

    if result["code"] == 200:
        print(f"  ✓ Score submitted successfully.")
    else:
        print(f"  ✗ Error submitting score")


def fetch_eval_and_submit(task_id, code_id, index, total, model, sample_size):
    """Fetch code, evaluate it, and submit the score."""
    print(f"\n[{index}/{total}] Evaluating code: {code_id}")
    print("-" * 70)

    # Step 2: Fetch code content
    code_content = step_2_fetch_code_content_by_id(code_id)
    if code_content is None:
        return

    # Step 3: Evaluate code
    reward, code_error_msg, correctness_check, details = step_3_get_reward(
        code_content
    )
    print(f"[Eval] Score: {reward} for code with ID {code_id}")
    if code_error_msg:
        print(f"[Eval] Error: {code_error_msg}")

    # Step 4: Submit score
    step_4_submit_score(task_id, code_id, reward, code_error_msg, correctness_check, details)


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


def main(task_id):
    print(f"  task_id: {task_id}")
    print("=" * 70)

    while True:
        task_is_finished, unevaluated_code_ids = step_1_fetch_all_unevaluated_code_ids(task_id)
        if task_is_finished:
            print("  [Task is finished]")
            print(len(unevaluated_code_ids))
            break
        print(f"  ✓ Found {len(unevaluated_code_ids)} unevaluated code(s)")
        print("-" * 70)
        if not unevaluated_code_ids:
            # Get training status
            step_5_get_training_status(task_id)

            print(f"\n[Waiting {POLL_INTERVAL} seconds before next poll...]")
            time.sleep(POLL_INTERVAL)
            continue

        # Process each unevaluated code in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(unevaluated_code_ids), 8)) as executor:
            futures = [
                executor.submit(
                    fetch_eval_and_submit,
                    task_id,
                    code_id,
                    index,
                    len(unevaluated_code_ids),
                    args.model,
                    args.sample_size
                )
                for index, code_id in enumerate(unevaluated_code_ids, 1)
            ]

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[Error] Exception during evaluation: {e}")

        # Get training status
        step_5_get_training_status(task_id)

        # print(f"\n[Waiting {POLL_INTERVAL} seconds before next poll...]")
        # time.sleep(POLL_INTERVAL)


def get_or_create_task_id():
    """Load task_id from task_id.txt if exists, otherwise create new task."""
    task_id_path = os.path.join(os.path.dirname(__file__), f"task_id_model_{args.model}_sample_{args.sample_size}.txt")

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
    task_id = create_task("./init_code.py")

    # Save task_id
    with open(task_id_path, "w") as f:
        f.write(task_id)
    print(f"  Saved task_id to {task_id_path}")

    return task_id


if __name__ == "__main__":
    task_id = get_or_create_task_id()
    main(task_id)