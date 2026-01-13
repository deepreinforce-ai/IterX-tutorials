"""
Evaluation script for the MEV Arbitrage task.

This script evaluates arbitrage contracts by:
1. Deploying mock tokens (WETH, USDC, DAI)
2. Deploying 4 DEX pools with price discrepancies
3. Deploying a flash lender with WETH liquidity
4. Giving the arbitrage contract 10 WETH starting capital
5. Calling executeArbitrage()
6. Measuring profit in WETH

Reward = profit / max_extractable
"""

import os
import json
import shutil
import subprocess
import re
from pathlib import Path
from typing import Tuple


# Configuration
INITIAL_CAPITAL = 10  # WETH
MAX_EXTRACTABLE = 20.0  # Maximum profit for normalization (huge pools allow more extraction)


def setup_hardhat_project(submission_path: str, work_dir: str, force_update_test: bool = False) -> str:
    """
    Set up a Hardhat project with mock contracts and the arbitrage submission.
    
    Args:
        submission_path: Path to the submitted Arbitrage.sol
        work_dir: Working directory for the Hardhat project
        force_update_test: If True, always copy the latest test file
        
    Returns:
        Path to the project directory
    """
    project_dir = Path(work_dir) / "hardhat_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    base_dir = Path(__file__).parent
    
    # Create contracts directory
    contracts_dir = project_dir / "contracts"
    contracts_dir.mkdir(exist_ok=True)
    
    # Copy mock contracts (static, only if not exists)
    mock_contracts = ["MockERC20.sol", "MockDEXPool.sol", "MockFlashLender.sol"]
    for contract in mock_contracts:
        dest = contracts_dir / contract
        if not dest.exists():
            src = base_dir / "contracts" / contract
            if src.exists():
                shutil.copy(src, dest)
    
    # Always copy the submission (it changes each time)
    shutil.copy(submission_path, contracts_dir / "Arbitrage.sol")
    
    # Copy hardhat config (only if not exists)
    if not (project_dir / "hardhat.config.js").exists():
        src = base_dir / "hardhat" / "hardhat.config.js"
        if src.exists():
            shutil.copy(src, project_dir / "hardhat.config.js")
    
    # Copy test script (always update to get latest version with detailed output)
    test_dir = project_dir / "test"
    test_dir.mkdir(exist_ok=True)
    src = base_dir / "hardhat" / "arbitrage_test.js"
    if src.exists():
        shutil.copy(src, test_dir / "arbitrage_test.js")
    
    # Create package.json (only if not exists)
    if not (project_dir / "package.json").exists():
        package_json = {
            "name": "mev-arbitrage-eval",
            "version": "1.0.0",
            "devDependencies": {
                "hardhat": "^2.19.0",
                "@nomicfoundation/hardhat-toolbox": "^4.0.0"
            }
        }
        with open(project_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=4)
    
    return str(project_dir)


def parse_test_output(output: str) -> dict:
    """
    Parse detailed information from the test output.
    
    Returns:
        Dictionary with parsed values
    """
    result = {
        'initial_balance': None,
        'final_balance': None,
        'profit': None,
        'loss': None,
        'loss_percentage': None,
        'execution_error': None,
        'pool_a_price_before': None,
        'pool_b_price_before': None,
        'pool_c_price_before': None,
        'pool_a_price_after': None,
        'pool_b_price_after': None,
        'pool_c_price_after': None,
        'pool_a_impact': None,
        'pool_b_impact': None,
        'pool_c_impact': None,
        'spread': None,
        'hints': []
    }
    
    # Parse values using regex
    patterns = {
        'initial_balance': r'Initial Balance:\s*([\d.]+)\s*WETH',
        'final_balance': r'Final Balance:\s*([\d.]+)\s*WETH',
        'profit': r'Profit:\s*([\d.]+)\s*WETH',
        'loss': r'LOSS:\s*([\d.]+)\s*WETH',
        'loss_percentage': r'Loss percentage:\s*([\d.]+)%',
        'execution_error': r'Execution Error:\s*(.+)',
        'pool_a_price_before': r'Pool A price:\s*([\d.]+)',
        'pool_b_price_before': r'Pool B price:\s*([\d.]+)',
        'pool_c_price_before': r'Pool C price:\s*([\d.]+)',
        'pool_a_price_after': r'Pool A price after:\s*([\d.]+)',
        'pool_b_price_after': r'Pool B price after:\s*([\d.]+)',
        'pool_c_price_after': r'Pool C price after:\s*([\d.]+)',
        'pool_a_impact': r'Pool A price impact:\s*([-\d.]+)%',
        'pool_b_impact': r'Pool B price impact:\s*([-\d.]+)%',
        'pool_c_impact': r'Pool C price impact:\s*([-\d.]+)%',
        'spread': r'Arbitrage spread:\s*([\d.]+)%',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            # Convert numeric values
            if key not in ['execution_error']:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
            else:
                result[key] = value
    
    # Parse hints
    hint_matches = re.findall(r'HINT:\s*(.+)', output)
    result['hints'] = hint_matches
    
    return result


def format_details(parsed: dict) -> str:
    """
    Format parsed test output into a human-readable details string.
    """
    lines = []
    
    # Execution result
    if parsed['execution_error']:
        lines.append(f"❌ EXECUTION FAILED: {parsed['execution_error']}")
    elif parsed['loss'] is not None and parsed['loss'] > 0:
        lines.append(f"❌ TRADE RESULTED IN LOSS: -{parsed['loss']:.4f} WETH ({parsed['loss_percentage']:.1f}% of capital)")
    elif parsed['profit'] is not None and parsed['profit'] > 0:
        lines.append(f"✓ Profitable: +{parsed['profit']:.4f} WETH")
    else:
        lines.append(f"⚠ No profit generated")
    
    # Balances
    if parsed['initial_balance'] is not None and parsed['final_balance'] is not None:
        lines.append(f"Balance: {parsed['initial_balance']:.2f} → {parsed['final_balance']:.4f} WETH")
    
    # Pool prices and impacts
    if parsed['pool_a_price_before'] is not None:
        lines.append(f"Pool A (cheap): {parsed['pool_a_price_before']:.0f} → {parsed['pool_a_price_after']:.0f} USDC/WETH (impact: {parsed['pool_a_impact']:+.1f}%)")
    if parsed['pool_b_price_before'] is not None:
        lines.append(f"Pool B (expensive): {parsed['pool_b_price_before']:.0f} → {parsed['pool_b_price_after']:.0f} USDC/WETH (impact: {parsed['pool_b_impact']:+.1f}%)")
    
    # Spread info
    if parsed['spread'] is not None:
        lines.append(f"Initial arbitrage spread: {parsed['spread']:.2f}%")
    
    # Price impacts summary
    if parsed['pool_a_impact'] is not None and parsed['pool_b_impact'] is not None:
        total_impact = abs(parsed['pool_a_impact']) + abs(parsed['pool_b_impact'])
        if total_impact > 5:
            lines.append(f"⚠ HIGH SLIPPAGE: Total price impact {total_impact:.1f}% - trade size too large!")
    
    # Hints from test
    for hint in parsed.get('hints', []):
        lines.append(f"💡 {hint}")
    
    # Add general guidance for common failure modes
    if parsed['execution_error'] and 'Insufficient balance' in parsed['execution_error']:
        lines.append("💡 TIP: Flash loan failed because arbitrage didn't generate enough profit to repay. Start with simple 2-pool arbitrage without flash loans.")
    
    if parsed['loss'] is not None and parsed['loss'] > 0:
        if parsed['pool_a_impact'] is not None and abs(parsed['pool_a_impact']) > 10:
            lines.append("💡 TIP: Use SMALL trade batches (0.05-0.2 WETH) in a LOOP instead of one large trade. Large trades cause excessive slippage.")
    
    return "\n".join(lines)


def run_arbitrage_test(project_dir: str) -> Tuple[bool, float, str, str]:
    """
    Run the arbitrage test and measure profit.
    
    Args:
        project_dir: Path to the Hardhat project
        
    Returns:
        Tuple of (success, profit_weth, error_msg, details)
    """
    try:
        # Install dependencies if needed
        node_modules_path = Path(project_dir) / "node_modules"
        if not node_modules_path.exists():
            subprocess.run(
                ["npm", "install"],
                cwd=project_dir,
                capture_output=True,
                timeout=120
            )
        
        # Run the arbitrage test
        result = subprocess.run(
            ["npx", "hardhat", "test", "--grep", "Arbitrage"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Parse output for profit
        output = result.stdout + result.stderr
        
        # Parse detailed information
        parsed = parse_test_output(output)
        details = format_details(parsed)
        
        # Look for the profit amount in the output
        profit = parsed.get('profit', 0.0) or 0.0
        
        # If we found a final balance but no profit, calculate it
        if profit == 0.0 and parsed.get('final_balance') is not None:
            profit = max(0, parsed['final_balance'] - INITIAL_CAPITAL)
        
        # Determine if there was an execution error
        if parsed.get('execution_error'):
            # Test ran but contract execution failed
            return True, 0.0, parsed['execution_error'], details
        
        # If test passed but no profit found, check return code
        if result.returncode == 0:
            return True, profit, "", details
        
        # Test failed - extract error message
        error_msg = output[-2000:] if len(output) > 2000 else output
        return False, 0.0, error_msg, details
        
    except subprocess.TimeoutExpired:
        return False, 0.0, "Arbitrage test timed out", ""
    except Exception as e:
        return False, 0.0, f"Error running arbitrage test: {e}", ""


def compute_reward(profit_weth: float) -> float:
    """
    Compute reward based on profit extracted.
    
    Reward = profit / max_extractable
    
    Args:
        profit_weth: Profit in WETH
        
    Returns:
        Reward in range [0.0, 1.0]
    """
    reward = profit_weth / MAX_EXTRACTABLE
    return max(0.0, min(1.0, reward))


def get_reward(code: str, work_dir: str = None) -> Tuple[float, str, str]:
    """
    Evaluate an arbitrage contract and return the reward.
    
    Args:
        code: Solidity code string for the Arbitrage contract
        work_dir: Optional working directory for Hardhat project (reuses node_modules).
                  If None, creates a temp directory that is deleted after evaluation.
        
    Returns:
        Tuple of (reward, error_msg, details)
        - reward: float in range [0.0, 1.0]
        - error_msg: "" if successful, error description otherwise
        - details: Comprehensive feedback about the execution
    """
    import tempfile
    import shutil
    
    # If work_dir not provided, create temp dir (will be deleted at end)
    cleanup_work_dir = work_dir is None
    if work_dir is None:
        work_dir = tempfile.mkdtemp()
    else:
        os.makedirs(work_dir, exist_ok=True)
    
    arbitrage_path = os.path.join(work_dir, "Arbitrage.sol")
    
    try:
        # Write code to file
        with open(arbitrage_path, 'w') as f:
            f.write(code)
        
        if 'contract' not in code:
            return 0.0, "Invalid Solidity file: no contract found", ""
        
        if 'Arbitrage' not in code:
            return 0.0, "Contract must be named 'Arbitrage'", ""
        
        # Setup project
        project_dir = setup_hardhat_project(arbitrage_path, work_dir)
        
        # Run arbitrage test
        success, profit, error_msg, details = run_arbitrage_test(project_dir)
        
        if not success:
            print("Arbitrage test failed")
            print(f"Error: {error_msg}")
            return 0.0, error_msg, details
        
        # Compute reward
        reward = compute_reward(profit)
        
        print(f"Initial: {INITIAL_CAPITAL} WETH")
        print(f"Profit: {profit} WETH")
        print(f"Reward: {reward}")
        if details:
            print(f"Details:\n{details}")
        
        return reward, error_msg, details
        
    except Exception as e:
        error_msg = f"Error evaluating arbitrage: {e}"
        print(error_msg)
        return 0.0, error_msg, ""
    finally:
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    # Test with the initial code
    base_dir = Path(__file__).parent
    initial_code_path = base_dir / "working_arbitrage.sol"
    
    with open(initial_code_path) as f:
        code = f.read()
    
    reward, error_msg, details = get_reward(code)
    print(f"\nFinal reward: {reward}")
    if error_msg:
        print(f"Error: {error_msg}")
    if details:
        print(f"\nDetailed feedback:\n{details}")
