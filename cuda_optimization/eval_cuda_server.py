"""
CUDA Optimization Evaluation Client

This module provides a get_reward function that:
1. Starts a CUDA evaluation server if not running
2. Sends evaluation request to the server
3. Polls the log file for results
4. Kills the server before returning

Completely independent - does not depend on eval_cuda.py
"""

import os
import sys
import json
import time
import requests
import traceback
import subprocess
import psutil
from pathlib import Path
from typing import Tuple, Optional
from urllib.parse import urlparse


# Server script path (absolute path to cuda_eval_flask_server.py)
CUDA_SERVER_SCRIPT_PATH = str(Path(__file__).parent.resolve() / "cuda_eval_flask_server.py")


# =============================================================================
# Server Management Functions
# =============================================================================

def kill_process_on_port(url: str):
    """Kill any process running on the specified port"""
    try:
        parsed = urlparse(url)
        port = parsed.port
        if port is None:
            return
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.net_connections():
                    if conn.laddr.port == port:
                        proc.kill()
                        print(f"Killed process {proc.pid} on port {port}")
                        return
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
    except Exception as e:
        print(f"Error killing process on port: {e}")


def start_eval_server(url: str, log_path: str, script_path: str):
    """Start an evaluation server as a background process"""
    parsed = urlparse(url)
    port = parsed.port
    
    # Kill any existing process on this port
    kill_process_on_port(url)
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Start the server by running the script directly
    with open(log_path, 'w') as log_file:
        subprocess.Popen(
            [sys.executable, script_path, "--port", str(port)],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True
        )
    
    print(f"Started server on port {port}, log: {log_path}")


def check_server_health(url: str, timeout: float = 5.0) -> bool:
    """Check if the server is responding"""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def wait_for_server_ready(url: str, max_wait: float = 180.0, check_interval: float = 2.0) -> bool:
    """Wait for the server to become ready"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if check_server_health(url, timeout=2.0):
            print(f"Server at {url} is ready")
            return True
        time.sleep(check_interval)
    
    print(f"Server at {url} failed to start within {max_wait}s")
    return False


def get_reward(
    code_path: str,
    work_dir: str,
    eval_url: str = "http://localhost:5001",
    log_folder: str = "",
    info_string: str = "",
    device_index: int = 0,
    warmup_time: float = 5.0,
    run_time: float = 10.0,
    timeout: float = 600.0,
    server_script_path: str = CUDA_SERVER_SCRIPT_PATH
) -> Tuple[float, str, str]:
    """
    Evaluate a custom CUDA kernel by managing server lifecycle and polling for results.
    
    Args:
        code_path: Path to the custom model Python file
        work_dir: Working directory for CUDA builds
        eval_url: URL of the evaluation server (e.g., "http://localhost:5001")
        log_folder: Folder to store logs
        info_string: Identifier for logging
        device_index: GPU index to use (default: 0)
        warmup_time: Warmup duration in seconds (default: 5.0)
        run_time: Benchmark duration in seconds (default: 10.0)
        timeout: Maximum time to wait for evaluation (default: 600.0)
        server_script_path: Path to the server Python script
        
    Returns:
        Tuple of (reward, error_msg, details)
        - reward: float representing speedup (original_time / custom_time), 0 if failed
        - error_msg: "" if successful, error description otherwise
        - details: "" (reserved for future use)
    """
    t1 = time.time()
    if log_folder == "":
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{int((time.time()%60)*100):02d}"
        log_folder = os.path.join(work_dir, f"log_{time_str}")
    # Setup directories
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    
    # Server log path
    server_log_path = os.path.join(log_folder, "server_output.log")
    
    # Result log path where server writes evaluation results
    result_log_path = os.path.join(log_folder, "eval_result.json")
    
    # Clear existing result log
    if os.path.exists(result_log_path):
        os.remove(result_log_path)
    
    try:
        # Start evaluation server
        print(f"[{info_string}] Starting eval server at {eval_url}")
        start_eval_server(eval_url, log_path=server_log_path, script_path=server_script_path)
        
        # Wait for server to be ready
        server_health = wait_for_server_ready(eval_url, max_wait=120.0)
        
        if not server_health:
            print(f"[{info_string}] Server failed to start")
            kill_process_on_port(eval_url)
            return 0.0, "Server failed to start", ""
        
        # Send evaluation request to server
        payload = {
            "code_path": code_path,
            "work_dir": work_dir,
            "log_path": result_log_path,
            "info_string": info_string,
            "device_index": device_index,
            "warmup_time": warmup_time,
            "run_time": run_time
        }
        
        try:
            response = requests.post(
                f"{eval_url}/api/evaluate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # Request timeout, not evaluation timeout
            )
            print(f"[{info_string}] Evaluation request sent, status: {response.status_code}")
        except requests.exceptions.Timeout:
            # Server might still be processing async
            print(f"[{info_string}] Request timeout, server may still be processing")
        except requests.exceptions.RequestException as e:
            print(f"[{info_string}] Request exception: {type(e).__name__} {str(e)}")
        
        # Poll for results
        polling_start_time = time.time()
        sleep_time = 5
        prev_line = ""
        prev_time = time.time()
        
        while time.time() - polling_start_time < timeout:
            # Check if result file exists
            if not os.path.exists(result_log_path):
                print(f"[{info_string}] Waiting for result file...")
                time.sleep(sleep_time)
                continue
            
            # Read the log file
            with open(result_log_path, 'r') as f:
                log_content = f.read().strip()
            
            if not log_content:
                print(f"[{info_string}] Result file is empty, waiting...")
                time.sleep(sleep_time)
                continue
            
            # Parse each line to find the final result
            log_lines = log_content.split('\n')
            log_line = log_lines[-1]
            
            # Check for timeout on same line
            if log_line == prev_line:
                # Determine timeout threshold based on stage
                time_bar = 60
                
                if time.time() - prev_time > time_bar:
                    print(f"[{info_string}] Timeout at stage: {prev_line[:100]}...")
                    
                    # Determine error message based on stage
                    if "start_time" in log_line:
                        error_msg = "Startup timeout"
                    elif "evaluation_started" in log_line:
                        error_msg = "Model loading timeout"
                    elif "model_loading_done" in log_line:
                        error_msg = "Defense check timeout"
                    elif "defense_checks_done" in log_line:
                        error_msg = "Correctness test timeout"
                    elif "correctness_tests_done" in log_line:
                        error_msg = "Performance benchmark timeout"
                    else:
                        error_msg = "Unknown stage timeout"
                    
                    kill_process_on_port(eval_url)
                    t2 = time.time()
                    print(f"[{info_string}] Time cost: {t2 - t1:.2f}s")
                    return 0.0, error_msg, ""
                else:
                    time.sleep(sleep_time)
                    continue
            
            # Try to parse the log line as JSON
            try:
                final_result = json.loads(log_line)
            except json.JSONDecodeError:
                print(f"[{info_string}] Failed to parse log line, waiting...")
                time.sleep(sleep_time)
                prev_line = log_line
                prev_time = time.time()
                continue
            
            # Check if evaluation is complete
            if final_result.get("done"):
                # Evaluation complete - kill server
                kill_process_on_port(eval_url)
                
                if final_result.get("error"):
                    error_msg = final_result.get("error_msg", "Unknown error")
                    print(f"[{info_string}] Error: {error_msg}")
                    t2 = time.time()
                    print(f"[{info_string}] Time cost: {t2 - t1:.2f}s")
                    return 0.0, error_msg, ""
                
                # Success - return the reward
                reward = final_result.get("reward", 0.0)
                t2 = time.time()
                print(f"[{info_string}] Reward: {reward:.4f} (time: {t2 - t1:.2f}s)")
                return float(reward), "", ""
            else:
                # Still in progress
                print(f"[{info_string}] In progress: {log_line[:100]}...")
                time.sleep(sleep_time)
                prev_line = log_line
                prev_time = time.time()
                continue
        
        # Overall timeout reached
        print(f"[{info_string}] Overall timeout reached ({timeout}s)")
        kill_process_on_port(eval_url)
        t2 = time.time()
        print(f"[{info_string}] Time cost: {t2 - t1:.2f}s")
        return 0.0, "Overall timeout", ""
        
    except Exception as e:
        error_msg = f"Exception: {type(e).__name__} {str(e)}\n{traceback.format_exc()}"
        print(f"[{info_string}] {error_msg}")
        kill_process_on_port(eval_url)
        return 0.0, error_msg, ""


def get_reward_direct(
    code_path: str,
    work_dir: str,
    eval_url: str,
    info_string: str = "",
    device_index: int = 0,
    warmup_time: float = 5.0,
    run_time: float = 10.0,
    timeout: float = 600.0
) -> Tuple[float, str]:
    """
    Simplified get_reward that assumes server is already running.
    Sends request and waits for response directly (synchronous).
    
    Args:
        code_path: Path to the custom model Python file
        work_dir: Working directory for CUDA builds
        eval_url: URL of the evaluation server (e.g., "http://localhost:5001")
        info_string: Identifier for logging
        device_index: GPU index to use (default: 0)
        warmup_time: Warmup duration in seconds (default: 5.0)
        run_time: Benchmark duration in seconds (default: 10.0)
        timeout: Request timeout in seconds (default: 600.0)
        
    Returns:
        Tuple of (reward, error_msg)
    """
    payload = {
        "code_path": code_path,
        "work_dir": work_dir,
        "log_path": "/dev/null",  # Not using log polling
        "info_string": info_string,
        "device_index": device_index,
        "warmup_time": warmup_time,
        "run_time": run_time
    }
    
    try:
        response = requests.post(
            f"{eval_url}/api/evaluate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        
        result = response.json()
        
        if result.get("success"):
            return float(result.get("reward", 0.0)), ""
        else:
            return 0.0, result.get("error", "Unknown error")
            
    except requests.exceptions.Timeout:
        return 0.0, f"Request timeout after {timeout}s"
    except requests.exceptions.RequestException as e:
        return 0.0, f"Request error: {type(e).__name__} {str(e)}"
    except Exception as e:
        return 0.0, f"Exception: {type(e).__name__} {str(e)}"


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":    
    
    reward, error_msg = get_reward(
        code_path=os.path.join(os.path.dirname(__file__), "custom_code.py"),
        work_dir=os.path.join(os.getcwd(), "cuda_build"),
    )
    
    print()
    print(f"Result: reward={reward:.4f}, error={error_msg}")
