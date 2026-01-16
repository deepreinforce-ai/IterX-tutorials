#!/usr/bin/env python3
"""
Gas-Efficient Batch Token Transfer - Evaluation Script
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import traceback


@dataclass
class GasResult:
    """Results from a single gas measurement"""
    batch_size: int
    gas_used: int
    success: bool
    error: Optional[str] = None


def load_config() -> dict:
    """Load evaluation configuration"""
    config_path = Path(__file__).parent / "hardhat" / "config.json"
    with open(config_path) as f:
        return json.load(f)


def validate_solidity_file(filepath: str) -> Tuple[bool, str]:
    """Validate that the submitted file is a valid Solidity contract"""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    if not filepath.endswith('.sol'):
        return False, "File must have .sol extension"
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'function batchTransfer' not in content:
        return False, "Contract must contain 'batchTransfer' function"
    
    if 'address[] calldata recipients' not in content and 'address[] memory recipients' not in content:
        return False, "batchTransfer must accept recipients array"
    
    if 'uint256[] calldata amounts' not in content and 'uint256[] memory amounts' not in content:
        return False, "batchTransfer must accept amounts array"
    
    return True, ""


def setup_hardhat_project(submission_path: str, temp_dir: str) -> str:
    """Set up a temporary Hardhat project with the submission"""
    project_dir = Path(temp_dir) / "hardhat_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    base_dir = Path(__file__).parent
    
    # Copy contracts directory structure
    contracts_dir = project_dir / "contracts"
    contracts_dir.mkdir(exist_ok=True)
    (contracts_dir / "interfaces").mkdir(exist_ok=True)
    
    # Static files - only copy if not exist (avoid NFS overhead)
    ierc20_dest = contracts_dir / "interfaces" / "IERC20.sol"
    if not ierc20_dest.exists():
        shutil.copy(base_dir / "contracts" / "interfaces" / "IERC20.sol", ierc20_dest)
    
    mockerc20_dest = contracts_dir / "MockERC20.sol"
    if not mockerc20_dest.exists():
        shutil.copy(base_dir / "contracts" / "MockERC20.sol", mockerc20_dest)
    
    # Submission - always copy (this is the new code being evaluated)
    shutil.copy(submission_path, contracts_dir / "BatchTransfer.sol")
    
    # Copy test files - only if not exist
    test_dir = project_dir / "test"
    test_dir.mkdir(exist_ok=True)
    test_dest = test_dir / "BatchTransfer.test.js"
    if not test_dest.exists():
        shutil.copy(base_dir / "correctness_check" / "BatchTransfer.test.js", test_dest)
    
    # Copy Hardhat config - only if not exist
    config_dest = project_dir / "hardhat.config.js"
    if not config_dest.exists():
        shutil.copy(base_dir / "hardhat" / "hardhat.config.js", config_dest)
    
    # Write package.json - only if not exist
    package_dest = project_dir / "package.json"
    if not package_dest.exists():
        package_json = {
            "devDependencies": {
                "@nomicfoundation/hardhat-toolbox": "^4.0.0",
                "hardhat": "^2.19.0"
            },
            "dependencies": {
                "@openzeppelin/contracts": "^5.0.0"
            }
        }
        with open(package_dest, "w") as f:
            json.dump(package_json, f, indent=4)
    
    # Copy gas benchmark script - only if not exist
    eval_dir = project_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    benchmark_dest = eval_dir / "gas_benchmark.js"
    if not benchmark_dest.exists():
        shutil.copy(base_dir / "hardhat" / "gas_benchmark.js", benchmark_dest)
    
    return str(project_dir)


def run_correctness_tests(project_dir: str) -> Tuple[bool, Dict[str, bool]]:
    """Run correctness tests using Hardhat"""
    try:
        # Install dependencies
        # Only install if node_modules doesn't exist
        node_modules_path = Path(project_dir) / "node_modules"
        if not node_modules_path.exists():
            subprocess.run(
                ["npm", "install"],
                cwd=project_dir,
                capture_output=True,
                timeout=120
            )
        
        # Run tests
        result = subprocess.run(
            ["npx", "hardhat", "test"],
            cwd=project_dir,
            capture_output=True,
            timeout=300,
            text=True
        )
        
        if result.returncode == 0:
            return True, {"all_tests": True}
        else:
            return False, {"all_tests": False, "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        return False, {"error": "Test execution timed out"}
    except Exception as e:
        return False, {"error": str(e)}


def measure_gas(project_dir: str, batch_sizes: List[int]) -> Dict[int, GasResult]:
    """Measure gas consumption for different batch sizes"""
    results = {}
    
    try:
        result = subprocess.run(
            ["npx", "hardhat", "run", "evaluation/gas_benchmark.js", "--network", "hardhat"],
            cwd=project_dir,
            capture_output=True,
            timeout=600,
            text=True
        )
        
        if result.returncode != 0:
            for size in batch_sizes:
                results[size] = GasResult(size, 0, False, result.stderr)
            return results
        
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            if line.startswith('GAS_RESULT:'):
                data = json.loads(line[11:])
                for size_str, gas in data.items():
                    size = int(size_str)
                    results[size] = GasResult(size, gas, True)
        
        for size in batch_sizes:
            if size not in results:
                results[size] = GasResult(size, 0, False, "No result returned")
                
    except subprocess.TimeoutExpired:
        for size in batch_sizes:
            results[size] = GasResult(size, 0, False, "Benchmark timed out")
    except Exception as e:
        for size in batch_sizes:
            results[size] = GasResult(size, 0, False, str(e))
    
    return results


def get_reward(code: str, work_dir: str = None) -> Tuple[float, str, str]:
    """
    Main entry point for IterX integration
    
    Args:
        code: Solidity code string for the BatchTransfer contract
        work_dir: Optional working directory for Hardhat project (reuses node_modules).
                  If None, creates a temp directory that is deleted after evaluation.
        
    Returns:
        Tuple of (reward, error_msg, details)
        - reward: Percentage of gas saved compared to baseline (1 - total_gas / 19399102), 
                  clamped to 0 if negative (higher = better), 0 if validation or tests fail
        - error_msg: "" if successful, error description otherwise
        - details: "" (reserved for future use)
    """
    import tempfile
    
    # If work_dir not provided, create temp dir (will be deleted at end)
    cleanup_work_dir = work_dir is None
    if work_dir is None:
        work_dir = tempfile.mkdtemp()
    else:
        os.makedirs(work_dir, exist_ok=True)
    
    code_path = os.path.join(work_dir, "BatchTransfer.sol")
    
    # Write code to file
    with open(code_path, 'w') as f:
        f.write(code)
    
    config = load_config()
    
    is_valid, error = validate_solidity_file(code_path)
    if not is_valid:
        print(f"Validation failed: {error}")
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        return 0.0, f"Validation failed: {error}", ""
    
    try:
        project_dir = setup_hardhat_project(code_path, work_dir)
        
        correctness_passed, test_results = run_correctness_tests(project_dir)
        if not correctness_passed:
            error_msg = f"Correctness tests failed: {test_results}"
            print(error_msg)
            if cleanup_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
            return 0.0, error_msg, ""
        
        batch_sizes = config["evaluation"]["batch_sizes"]
        gas_results_raw = measure_gas(project_dir, batch_sizes)
        
        total_gas = sum(
            result.gas_used 
            for result in gas_results_raw.values() 
            if result.success and result.gas_used > 0
        )
        
        # Print gas results for debugging
        gas_dict = {size: result.gas_used for size, result in gas_results_raw.items() if result.success}
        
        print(f"total_gas {total_gas}")
        baseline_gas = 19399102
        if total_gas > 0:
            save_percent = 1 - total_gas / baseline_gas
            save_percent = max(0.0, min(1.0, save_percent))
            return save_percent, "", ""
        return 0.0, "No gas measurements returned", ""
        
    except Exception as e:
        error_msg = f"Exception in get_reward: {e}"
        print(error_msg)
        print(traceback.format_exc())
        return 0.0, error_msg, ""
    finally:
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "initial_code.sol")) as f:
        code = f.read()
    reward, error_msg, details = get_reward(code)
    print(f"Reward: {reward}, Error: {error_msg}")
