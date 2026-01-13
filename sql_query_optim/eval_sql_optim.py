"""
Evaluation Function for SQL Query Optimization Task

Evaluates the performance and correctness of SQL query transformations.
Measures execution time improvement while ensuring result correctness.
"""

import time
import hashlib
import sqlite3
from typing import Tuple, Optional, Callable, List, Any, Dict
from statistics import median
from tqdm import tqdm

try:
    from .fake_database import get_connection, get_schema_info
    from .test_queries import get_test_queries
except ImportError:
    from fake_database import get_connection, get_schema_info
    from test_queries import get_test_queries


# =============================================================================
# Configuration
# =============================================================================

# Number of warmup runs before timing (for both original and transformed)
WARMUP_RUNS = 1

# Number of timed runs for median calculation
TIMED_RUNS = 7

# Maximum time (seconds) for a single query execution
QUERY_TIMEOUT = 10.0

# Minimum speedup to get positive reward
MIN_SPEEDUP_THRESHOLD = 1.0


# =============================================================================
# Query Execution
# =============================================================================

def normalize_results(results: List[Tuple]) -> str:
    """
    Normalize query results for comparison.
    Sorts rows and converts to a canonical string representation.
    """
    if not results:
        return "EMPTY"
    
    # Sort rows (convert each row to tuple for sorting)
    sorted_results = sorted([tuple(str(x) for x in row) for row in results])
    
    # Create hash of results for comparison
    result_str = str(sorted_results)
    return hashlib.md5(result_str.encode()).hexdigest()


def execute_query_with_timeout(conn: sqlite3.Connection, sql: str, 
                                timeout: float = QUERY_TIMEOUT) -> Tuple[List[Tuple], float, Optional[str]]:
    """
    Execute a query with timeout and return results, execution time, and error.
    
    Args:
        conn: Database connection
        sql: SQL query to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Tuple of (results, execution_time, error_message)
        - results: List of result tuples (empty if error)
        - execution_time: Time in seconds (-1 if error)
        - error_message: Error string or None
    """
    try:
        # Set timeout (SQLite doesn't have native timeout, so we use progress handler)
        start_time = time.time()
        
        def progress_handler():
            if time.time() - start_time > timeout:
                return 1  # Non-zero to interrupt
            return 0
        
        conn.set_progress_handler(progress_handler, 100)  # Check every 100 opcodes
        
        cursor = conn.cursor()
        t1 = time.time()
        cursor.execute(sql)
        results = cursor.fetchall()
        t2 = time.time()
        
        conn.set_progress_handler(None, 0)
        
        return results, t2 - t1, None
        
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            return [], -1, f"Query timeout after {timeout}s"
        return [], -1, f"SQL Error: {str(e)}"
    except Exception as e:
        return [], -1, f"Error: {str(e)}"


def measure_query_time(conn: sqlite3.Connection, sql: str, 
                       warmup: int = WARMUP_RUNS, 
                       runs: int = TIMED_RUNS) -> Tuple[float, Optional[str]]:
    """
    Measure median execution time of a query.
    
    Args:
        conn: Database connection
        sql: SQL query to execute
        warmup: Number of warmup runs
        runs: Number of timed runs
        
    Returns:
        Tuple of (median_time, error_message)
    """
    # Warmup runs
    for _ in range(warmup):
        _, exec_time, error = execute_query_with_timeout(conn, sql)
        if error:
            return -1, error
    
    # Timed runs
    times = []
    for _ in range(runs):
        _, exec_time, error = execute_query_with_timeout(conn, sql)
        if error:
            return -1, error
        times.append(exec_time)
    
    return median(times), None


# =============================================================================
# Transformer Evaluation
# =============================================================================

def evaluate_transformer(transformer: Callable[[str], str]) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a query transformer function.
    
    The transformer should take an SQL query string and return an optimized
    SQL query string that produces identical results.
    
    Args:
        transformer: Function that takes SQL string and returns optimized SQL
        
    Returns:
        Tuple of (reward, details_dict)
        - reward: Average speedup (original_time / transformed_time) across successful queries
        - details_dict: Per-query results and statistics
    """
    conn = get_connection(readonly=True)
    test_queries = get_test_queries()
    
    details = {
        "queries": {},
        "successful": 0,
        "failed": 0,
        "incorrect": 0,
        "total_speedup": 0.0,
    }
    
    for query_id, original_sql, description, hint in tqdm(test_queries, desc="Evaluating", unit="query"):
        query_result = {
            "description": description,
            "original_sql": original_sql.strip(),
            "transformed_sql": None,
            "original_time": None,
            "transformed_time": None,
            "speedup": None,
            "reward": 0.0,
            "status": "pending",
            "error": None,
        }
        
        try:
            # Get transformed query
            transformed_sql = transformer(original_sql)
            query_result["transformed_sql"] = transformed_sql.strip() if transformed_sql else None
            
            if not transformed_sql or not transformed_sql.strip():
                query_result["status"] = "failed"
                query_result["error"] = "Transformer returned empty query"
                details["failed"] += 1
                details["queries"][query_id] = query_result
                continue
            
            # Warm up BOTH queries first to ensure fair comparison
            # (SQLite caches query plans and data pages)
            for _ in range(WARMUP_RUNS):
                execute_query_with_timeout(conn, original_sql)
                execute_query_with_timeout(conn, transformed_sql)
            
            # Measure original query (multiple runs, take median)
            original_times = []
            original_results = None
            for _ in range(TIMED_RUNS):
                results, exec_time, error = execute_query_with_timeout(conn, original_sql)
                if error:
                    query_result["status"] = "failed"
                    query_result["error"] = f"Original query error: {error}"
                    details["failed"] += 1
                    break
                original_results = results
                original_times.append(exec_time)
            else:
                original_time = median(original_times)
                query_result["original_time"] = original_time
                original_hash = normalize_results(original_results)
            
            if query_result["status"] == "failed":
                details["queries"][query_id] = query_result
                continue
            
            # Measure transformed query (multiple runs, take median)
            transformed_times = []
            transformed_results = None
            for _ in range(TIMED_RUNS):
                results, exec_time, error = execute_query_with_timeout(conn, transformed_sql)
                if error:
                    query_result["status"] = "failed"
                    query_result["error"] = f"Transformed query error: {error}"
                    details["failed"] += 1
                    break
                transformed_results = results
                transformed_times.append(exec_time)
            else:
                transformed_time = median(transformed_times)
                query_result["transformed_time"] = transformed_time
                transformed_hash = normalize_results(transformed_results)
            
            if query_result["status"] == "failed":
                details["queries"][query_id] = query_result
                continue
            
            # Check correctness
            if original_hash != transformed_hash:
                query_result["status"] = "incorrect"
                query_result["error"] = f"Results mismatch: original {len(original_results)} rows, transformed {len(transformed_results)} rows"
                details["incorrect"] += 1
                details["queries"][query_id] = query_result
                continue
            
            # Calculate speedup
            if transformed_time > 0 and original_time > 0:
                speedup = original_time / transformed_time
                query_result["speedup"] = speedup
                details["total_speedup"] += speedup
            else:
                query_result["speedup"] = 1.0
                details["total_speedup"] += 1.0
            
            query_result["status"] = "success"
            details["successful"] += 1
            
        except Exception as e:
            query_result["status"] = "failed"
            query_result["error"] = f"Exception: {str(e)}"
            details["failed"] += 1
        
        details["queries"][query_id] = query_result
    
    conn.close()
    
    # Add summary statistics
    details["num_queries"] = len(test_queries)
    if details["successful"] > 0:
        details["avg_speedup"] = details["total_speedup"] / details["successful"]
    else:
        details["avg_speedup"] = 0.0
    
    # Reward is simply the average speedup
    total_reward = details["avg_speedup"]
    details["total_reward"] = total_reward
    
    return total_reward, details


# =============================================================================
# Main Evaluation Entry Point
# =============================================================================

def get_reward(code: str, 
               verbose: bool = True) -> Tuple[float, Optional[str], str]:
    """
    Evaluate a transformer from Python code.
    
    The code must define a function called `transform_query` that takes
    a SQL string and returns an optimized SQL string.
    
    Args:
        code: Python code string containing transform_query function
        verbose: Whether to print progress
        
    Returns:
        Tuple of (reward, error_message, details_string)
        - details_string: formatted string with per-query results
    """
    try:
        # Execute the code and extract the transform function
        namespace = {}
        exec(code, namespace)
        
        # Get the transform function
        if 'transform_query' not in namespace:
            return 0.0, "Code must define 'transform_query' function", ""
        
        transformer = namespace['transform_query']
        
        # Verify it's callable
        if not callable(transformer):
            return 0.0, "'transform_query' must be a callable function", ""
        
        if verbose:
            print("Evaluating query transformer...")
        
        reward, raw_details = evaluate_transformer(transformer)
        
        # Build details string
        lines = []
        lines.append(f"Summary: {raw_details['successful']}/{raw_details['num_queries']} successful, "
                     f"{raw_details['failed']} failed, {raw_details['incorrect']} incorrect, "
                     f"avg_speedup={raw_details['avg_speedup']:.2f}x")
        lines.append("")
        lines.append("Per-query results:")
        
        for query_id, query_result in raw_details.get("queries", {}).items():
            success = query_result["status"] == "success"
            speedup = query_result.get("speedup")
            error = query_result.get("error")
            original_sql = query_result.get("original_sql", "")
            
            # Clean up the SQL for display (normalize whitespace)
            sql_clean = " ".join(original_sql.split())
            
            status_symbol = "✓" if success else "✗"
            if speedup is not None:
                lines.append(f"  {status_symbol} {query_id}: speedup={speedup:.2f}x")
                lines.append(f"      SQL: {sql_clean}")
            elif error:
                lines.append(f"  {status_symbol} {query_id}: error={error}")
                lines.append(f"      SQL: {sql_clean}")
            else:
                lines.append(f"  {status_symbol} {query_id}")
                lines.append(f"      SQL: {sql_clean}")
        
        details_string = "\n".join(lines)
        
        if verbose:
            print(f"\nResults:")
            print(f"  Successful: {raw_details['successful']}/{raw_details['num_queries']}")
            print(f"  Failed: {raw_details['failed']}")
            print(f"  Incorrect: {raw_details['incorrect']}")
            print(f"  Avg Speedup: {raw_details['avg_speedup']:.2f}x")
            print(f"  Total Reward: {reward:.2f}")
        
        return reward, None, details_string
        
    except Exception as e:
        error_msg = f"Failed to evaluate transformer: {str(e)}"
        if verbose:
            print(error_msg)
        return 0.0, error_msg, ""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Get the path to initial_code.py (baseline transformer)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    initial_code_path = os.path.join(script_dir, "initial_code.py")
    
    print(f"Testing evaluation with baseline transformer: {initial_code_path}")
    
    with open(initial_code_path) as f:
        code = f.read()
    
    reward, error, details = get_reward(code, verbose=True)
    
    if error:
        print(f"\nError: {error}")
    else:
        print("\n\nDetails:")
        print(details)

