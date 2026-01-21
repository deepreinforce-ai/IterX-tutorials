# Task: Fast Sort
**Difficulty Level: ⭐ (1 star)**

---

## Background

Sorting is one of the most fundamental algorithms in computer science. While general-purpose sorting algorithms like quicksort achieve O(n log n) complexity, specialized algorithms can achieve better performance when the data has exploitable structure. For example, radix sort achieves O(nk) for integers, counting sort achieves O(n+k) for bounded ranges, and bucket sort can approach O(n) for uniformly distributed data.

In this task, the test data is generated with random but exploitable structure (clusters, partial ordering, duplicates, bounded ranges). The optimal sorting algorithm must be discovered through exploration, as the data properties vary with each generation.

The `get_reward()` function returns a tuple of **`(reward, error_msg, correctness_check, details)`**:
- **`reward`**: Inverse of average execution time (faster = higher reward). Returns `0` if correctness check fails.
- **`error_msg`**: Empty string `""` on success, or a description of the failure.
- **`correctness_check`**: `True` if sorted result matches `np.sort()`, `False` otherwise.
- **`details`**: Execution time statistics (avg, min, max over 20 runs).

---

## Task Description

**Goal:** Implement a fast sorting function called `mysort` that sorts a numpy array of integers.

**Input:** 
- `arr`: A numpy array of 500,000 integers with unknown but exploitable distribution

**Output:** A sorted numpy array (ascending order)

**Requirements:**
1. The function must be named `mysort`
2. Must accept a numpy array as input
3. Must return a sorted numpy array
4. Output must exactly match `np.sort(arr)`
5. You have access to numpy (as `np` or `numpy`). Feel free to use numba for performance.

**Things to Avoid:**
1. Modifying the evaluation harness
2. Hardcoding based on specific test data

---

## Reward Description

The reward is computed as the inverse of average execution time over 20 runs:

```
reward = 1.0 / average_time_in_seconds
```
---

## Initial Code

```python
import numpy as np
def mysort(arr):
    # Basic quicksort implementation - can you beat numpy's built-in sort?
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = arr[arr < pivot]
    middle = arr[arr == pivot]
    right = arr[arr > pivot]
    return np.concatenate([mysort(left), middle, mysort(right)])
```

---

## File Structure

```
sort_list/
├── README.md                 # This file
├── sort_test_data.npz        # Compressed test data
└── run_iterx.py              # IterX evaluation runner
```

---

## Running Evaluation

```bash
# Test evaluation locally
python -c "
from run_iterx import step_3_get_reward
code = '''
import numpy as np
def mysort(arr):
    return np.sort(arr)
'''
reward, err, correct, details = step_3_get_reward(code)
print(f'Reward: {reward:.2f}, Details: {details}')
"
```

---

## Running iterX

```bash
python run_iterx.py
```
