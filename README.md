# Iterx Code Examples

This repository contains example tasks for [IterX](https://iterx.deep-reinforce.com/). Each task demonstrates how to integrate custom rewarding logic with the Iterx API for reinforcement learning from code feedback.

## Installation

```bash
pip install -r requirements.txt
```

## Tasks

| Task | Category | Difficulty | Description |
|------|----------|------------|-------------|
| [pairwise_ranking](./pairwise_ranking/) | Demo | ⭐ | illustration only |
| [guess_lyric](./guess_lyric/) | Demo | ⭐ | Illustration only |
| [online_packing](./online_packing/) | Algorithms | ⭐⭐ | Online bin packing optimization |
| [sql_query_optim](./sql_query_optim/) | Database | ⭐⭐⭐ | Optimize SQL queries for performance |
| [smart_contract](./smart_contract/) | Blockchain | ⭐⭐⭐ | Smart contract gas optimization |
| [cuda_optimization](./cuda_optimization/) | Systems / GPU | ⭐⭐⭐ | CUDA kernel optimization |
| [exploit_contract](./exploit_contract/) | Blockchain / Security | ⭐⭐⭐⭐ | Smart contract exploit generation |
| [optimizer](./optimizer/) | Machine Learning | ⭐⭐⭐⭐ | Design an optimizer better than Adam |
| [mev_arbitrage](./mev_arbitrage/) | Blockchain / DeFi | ⭐⭐⭐⭐⭐ | MEV arbitrage strategy design |

> **Note:** Tasks with ⭐ (1 star) difficulty are for **illustration purposes only**, demonstrating how to set up an Iterx task.

## Task Structure

Each task folder contains:

- `run_iterx.py` - Main script to create task and run evaluation loop
- `eval_*.py` - Custom evaluation logic with `get_reward()` function
- `initial_code.*` - Initial code/template provided to the model
- `README.md` - Task-specific documentation

## Usage

```bash
cd <task_folder>
python run_iterx.py
```

The script will:
1. Create a new task (or load existing task_id from `task_id.txt`)
2. Poll for unevaluated code submissions
3. Evaluate each submission using the custom `get_reward()` function
4. Submit scores back to the Iterx API
5. Repeat until the task is finished

