"""
Iterx API - Create task and run evaluation loop
Task: MEV Arbitrage
"""

import os
import time

import requests
import yaml

from eval_mev_arbitrage import get_reward

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
    initial_code_path = os.path.join(os.path.dirname(__file__), "initial_code.sol")
    with open(initial_code_path, "r") as f:
        initial_code = f.read()
    
    response = requests.post(
        f"{BASE_URL}/api/task/create",
        headers=headers,
        json={
            "task_name": "MEV Arbitrage",
            "task_description": """**Goal:** Design a Solidity contract named `Arbitrage` that extracts maximum profit from price discrepancies across multiple DEX pools.

**Setup:**
- 4 DEX pools with different token pairs and deep liquidity
- Large price discrepancies between pools (~17.6% spread)
- Flash loan provider (10,000 WETH available, 0.05% fee)
- Starting capital: 10 WETH

**Pool Configuration:**
| Pool | Pair | Reserves | Implied Price |
|------|------|----------|---------------|
| Pool A | WETH/USDC | 5,000 WETH / 8,500,000 USDC | 1 WETH = **1,700** USDC (CHEAP) |
| Pool B | WETH/USDC | 4,000 WETH / 8,000,000 USDC | 1 WETH = **2,000** USDC (EXPENSIVE) |
| Pool C | WETH/DAI | 3,000 WETH / 5,550,000 DAI | 1 WETH = 1,850 DAI |
| Pool D | USDC/DAI | 5,000,000 USDC / 5,100,000 DAI | 1 USDC = 1.02 DAI |

**Arbitrage Opportunities:**
1. **Simple two-pool**: Buy WETH cheap in Pool A (1,700), sell expensive in Pool B (2,000) - **17.6% spread!**
2. **Triangular**: WETH → USDC → DAI → WETH through multiple pools
3. **Flash loan amplified**: Borrow up to 10,000 WETH to maximize extraction

**Requirements:**
1. Contract must be named `Arbitrage`
2. Must have `executeArbitrage()` function (main entry point)
3. Must have `onFlashLoan()` callback for flash loans (optional but recommended)
4. Must have `getProfit()` view function returning profit in WETH
5. Must have `receive()` function for ETH
6. Constructor accepts: flashLender, pools array, tokens array
7. Must compile with Solidity ^0.8.0
8. **Flash loan repayment**: `onFlashLoan()` must use `transfer()` (NOT `approve()`) to repay

**Things to Avoid:**
1. Running out of gas during complex multi-hop swaps
2. Incorrect slippage calculations leading to losses
3. Failing to repay flash loans (will revert)
4. Hardcoding values that only work for specific pool states

**Key Insights:**
- Pools use UniswapV2 constant-product formula (x * y = k) with 0.3% swap fee
- **Critical**: Use batched trades to minimize slippage - single large trades lose money!
- Flash loans available at 0.05% fee (5 basis points)
- Deep liquidity pools (5,000+ WETH) allow large-scale extraction
- Multiple arbitrage paths may exist - find the optimal combination""",
            "reward_description": "Normalized profit ratio: reward = profit / 20.0 (max extractable). Range [0.0, 1.0], higher is better. Returns 0 if contract fails or no profit generated.",
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
    # Use the get_reward function from eval_mev_arbitrage.py
    try:
        work_dir = "/data/iterx/mev_arbitrage"
        reward, error_msg, details = get_reward(code, work_dir=work_dir)
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
    
    # Save task_id
    with open(task_id_path, "w") as f:
        f.write(task_id)
    print(f"  Saved task_id to {task_id_path}")
    
    return task_id


if __name__ == "__main__":
    task_id = get_or_create_task_id()
    main(task_id)