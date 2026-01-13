# Task: Design a Pair Scoring Function
**Difficulty Level: ⭐ (1 star)**

---

## Background

Let's begin with a simple task to illustrate how to use IterX. The goal is to discover a hidden pairwise scoring function that reproduces the ordering of a known list of pairs. The reward is computed by comparing the ordering obtained from a scoring function with the ground-truth ordering.

---

## Reward Mechanism

The scoring function is evaluated on a set of ordered pairs. The reward measures how well the function preserves the ground-truth ordering by counting concordant pairs. The `get_reward()` function returns a tuple of **`(reward, error_msg, details)`**:
- **`reward`**: Fraction of concordant pairs in range `[0.0, 1.0]`.
- **`error_msg`**: Empty string `""` on success, or a description of the failure.
- **`details`**: Always `""`.

```python
ordered_pairs = [
    (20, 20),
    (11, 99),
    ...
    (10, 11),
    (10, 10),
]

def get_reward(code_path):
    """
    Evaluate a scoring function based on how well it ranks pairs.
    Returns: (reward, error_msg, details)
        - reward: fraction of concordant pairs (0 to 1)
        - error_msg: "" on success, or error description
        - details: always ""
    """
    try:
    # Load the score function from the code path
    score_func = load_function(code_path)
    
    # Compute scores for all pairs
    scores = [score_func(a, b) for a, b in ordered_pairs]
    
    # Count concordant pairs: for i < j in ordered_pairs,
    # the score of pair i should be >= score of pair j
    concordant = 0
    total = 0
    
    for i in range(len(ordered_pairs)):
        for j in range(i + 1, len(ordered_pairs)):
            total += 1
            if scores[i] >= scores[j]:
                concordant += 1
    
        reward = concordant / total
        return reward, "", ""
    except Exception as e:
        return 0.0, str(e), ""
```

---

## Task Description

**Goal:** Design a scoring function for pairs of integers.

**Input:** Two integers `a` and `b` (each in range 1-100)

**Output:** A numeric score where higher values indicate better pairs

**Requirements:**
1. The function must be named `score`
2. Must accept two integer arguments `a` and `b`
3. Must return a numeric value

**Things to Avoid:** None

---

## Reward Description

The score function is evaluated on pairs, comparing all pairwise orderings. The reward (0 to 1) is the fraction where your function correctly assigns a higher score to the better pair.

```
reward = concordant_pairs / total_pairs
```

- **Higher reward = better** (more correct orderings = higher score)
- **Reward range:** [0.0, 1.0]
- **Perfect score (1.0):** All pairwise orderings are correct

---

## Initial Code

```python
def score(a, b):
    return a + b
```

---

## File Structure

```
pairwise_ranking/
├── README.md                 # This file
├── eval_pairwise.py          # Evaluation script
├── initial_code.py           # Initial scoring function
└── run_iterx.py              # IterX evaluation runner
```

---

## Running Evaluation

```python
from eval_pairwise import get_reward

# Get reward for a scoring function
reward = get_reward('path/to/your/score.py')
print(reward)  # e.g., 0.75 (higher = better, max = 1.0)
```

---

## Running iterX

```bash
python run_iterx.py
```

