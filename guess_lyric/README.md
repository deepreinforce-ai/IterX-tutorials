# Task: Guess the Lyric
**Difficulty Level: ⭐ (1 star)**

---

## Background

Let's show how to use IterX for non-coding tasks. For illustration purposes, we'll play a lyric-guessing game where we provide a reference lyric and ask LLMs to guess it correctly. The LLM must learn to generate text that matches a hidden reference through iterative refinement based on similarity feedback.

The `get_reward()` function returns a tuple of **`(reward, error_msg, details)`**:
- **`reward`**: Jaccard similarity in range `[0.0, 1.0]`.
- **`error_msg`**: Empty string `""` on success, or a description of the failure.
- **`details`**: Always `""`.

---

## Task Description

**Goal:** Guess the hidden reference lyric.

**Input:** None (the reference lyric is hidden)

**Output:** A guessed lyric string

**Requirements:** None

**Things to Avoid:** None

---

## Reward Description

The reward is computed as the Jaccard similarity between the guessed lyric and the reference lyric:

```
reward = |G ∩ R| / |G ∪ R|
```

Where:
- \(G\) = set of tokens in the guessed lyric
- \(R\) = set of tokens in the reference lyric
---

## Initial Code

```
Imagine there's no heaven
```

---

## File Structure

```
guess_lyric/
├── README.md                 # This file
├── eval_guess_lyric.py       # Evaluation script
├── initial_guess.txt         # Initial guess to optimize
└── run_iterx.py              # IterX evaluation runner
```

---

## Running Evaluation

```bash
python eval_guess_lyric.py
```

---

## Running iterX

```bash
python run_iterx.py
```

