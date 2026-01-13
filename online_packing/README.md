# Task: Online Bin Packing
**Difficulty Level: ⭐⭐ (2 stars)**

---

## Background

Online bin packing is a widely used task for evaluating reinforcement learning and optimization algorithms. In this setting, items of varying sizes arrive sequentially and must be assigned to bins with limited capacity without knowledge of future items. The objective is to minimize the number of bins used. This problem formulation is broadly applicable to many resource allocation scenarios, including network routing, cloud computing resource allocation, and warehouse storage.

The online bin packing task can be boiled down to a ranking function, where given a new item and a set of bins with remaining capacity, the function assigns a score to each bin, and the item is placed into the bin with the highest score.

The `get_reward()` function returns a tuple of **`(reward, error_msg, details)`**:
- **`reward`**: Packing efficiency in range `(0.0, 1.0]`.
- **`error_msg`**: Empty string `""` on success, or a description of the failure.
- **`details`**: Always `""`.

---

## Task Description

**Goal:** Design a scoring function for online bin packing that minimizes the number of bins used.

**Input:** 
- `item`: Size of the current item to place (float)
- `bins`: Array of remaining capacities for each open bin (numpy array)

**Output:** An array of scores where higher values indicate preferred bins for placing the item

**Requirements:**
1. The function must be named `score`
2. Must accept `item` (float) and `bins` (numpy array) as arguments
3. Must return a numpy array of scores (same length as `bins`)
4. Use `@jit` decorator from numba for performance

**Things to Avoid:** None

---

## Reward Description
The reward measures how efficiently the algorithm packs items compared to the theoretical optimum:

```
reward = 1 - (bins_used - L1_bound) / L1_bound
```

- **Higher reward = better** (fewer bins = higher reward)
- **Reward range:** (0.0, 1.0]
- **Perfect score (1.0):** Uses exactly the theoretical minimum number of bins

---

## Initial Code

```python
import numpy as np
from numba import jit

@jit
def score(item, bins):
    """Naive Best Fit: prefer bins with least remaining space after placement."""
    return 1 / (bins - item + 1)
```

---

## File Structure

```
online_packing/
├── README.md                 # This file
├── eval_packing.py           # Evaluation script
├── initial_code.py           # Initial Best Fit algorithm
└── run_iterx.py              # IterX evaluation runner
```

---

## Running Evaluation

```python
from eval_packing import get_reward

# Get reward for a packing strategy
reward = get_reward('path/to/your/score.py')
print(reward)  # e.g., 0.85 (higher = better, max = 1.0)
```

---

## Running iterX

```bash
python run_iterx.py
```

