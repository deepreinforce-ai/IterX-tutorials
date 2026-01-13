import numpy as np
import importlib.util
from get_instance import GetData
import types
import warnings
import sys
import os
import tempfile
import threading
import traceback
import subprocess
import re

class BPONLINE():
    def __init__(self, score_function_path=None):
        """
        Initialize BPONLINE evaluator.
        
        Args:
            score_function_path: Path to a Python file containing a score function.
                                If None, the score function must be provided later.
        """
        getdate = GetData()
        self.instances, self.lb = getdate.get_instances()
        # Add a lock for thread safety during evaluation
        self._eval_lock = threading.Lock()
        
        # Load score function if path is provided
        self.score_function = None
        if score_function_path:
            self.load_score_function(score_function_path)
    
    def load_score_function(self, file_path):
        """
        Load a score function from a Python file.
        
        Args:
            file_path: Path to the Python file containing the score function
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Score function file not found: {file_path}")
        
        # Create a unique module name based on the file path
        module_name = f"score_module_{os.path.basename(file_path).replace('.py', '')}_{id(self)}"
        
        # Load the module from file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Could not create module spec for {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        if module is None:
            raise ImportError(f"Could not create module from spec")
        
        # Execute the module
        spec.loader.exec_module(module)
        
        # Extract the score function
        if not hasattr(module, 'score'):
            raise AttributeError(f"Module {file_path} does not have a 'score' function")
        
        self.score_function = module.score

    def get_valid_bin_indices(self, item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]

    def online_binpack(self, items: tuple, bins: np.ndarray):
        """
        Performs online binpacking of `items` into `bins`.
        Uses the loaded score function to determine bin priorities.
        """
        if self.score_function is None:
            raise ValueError("No score function loaded. Please load a score function first.")
        
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        # Add items to bins.
        n = 1
        for item in items:
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            # Score each bin based on heuristic.
            priorities = self.score_function(item, bins[valid_bin_indices])
            # Add item to bin with highest priority.
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
            packing[best_bin].append(item)
            n = n + 1
            
        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        return packing, bins

    def evaluate(self) -> float:
        """Evaluate heuristic function on a set of online binpacking instances."""
        if self.score_function is None:
            raise ValueError("No score function loaded. Please load a score function first.")
        
        total_fitness = 0.0
        num_datasets = 0
        
        for name, dataset in self.instances.items():
            num_bins_list = []
            for _, instance in dataset.items():
                capacity = instance['capacity']
                items = np.array(instance['items'])
                
                # Create num_items bins so there will always be space for all items,
                # regardless of packing order. Array has shape (num_items,).
                bins = np.array([capacity for _ in range(instance['num_items'])])
                # Pack items into bins and return remaining capacity in bins_packed, which
                # has shape (num_items,).
                bins = np.asarray(bins, dtype=np.float64)
                _, bins_packed = self.online_binpack(items, bins)
                # If remaining capacity in a bin is equal to initial capacity, then it is
                # unused. Count number of used bins.
                num_bins = (bins_packed != capacity).sum()
                
                num_bins_list.append(-num_bins)
            
            avg_num_bins = -np.mean(np.array(num_bins_list))
            fitness = (avg_num_bins - self.lb[name]) / self.lb[name]
            total_fitness += fitness
            num_datasets += 1
        
        # Return average fitness across all datasets
        return 1 - (total_fitness / num_datasets if num_datasets > 0 else 0.0)


def get_reward(code: str):
    """
    Evaluate a bin packing score function.
    
    Args:
        code: Python code string containing the score function
        
    Returns:
        Tuple of (reward, error_msg, details)
        - reward: float in range [0.0, 1.0] (higher = better packing)
        - error_msg: "" if successful, error description otherwise
        - details: "" (reserved for future use)
    """
    try:
        # Write code to temp file for numba JIT compilation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            bp = BPONLINE(temp_path)
            reward = bp.evaluate()
            return reward, "", ""
        finally:
            os.unlink(temp_path)
    except Exception as e:
        return 0.0, str(e), ""


if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), "initial_code.py")) as f:
        code = f.read()
    reward, error_msg, details = get_reward(code)
    print(f"Reward for initial code: {reward}, Error: {error_msg}")
