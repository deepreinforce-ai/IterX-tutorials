# Task: SQL Query Optimization
**Difficulty Level: ⭐⭐⭐ (3 stars)**

## Background

In this task, we build SQL query transformers that rewrite inefficient queries into optimized versions. Database query optimization is a fundamental problem in computer science where the same logical query can have vastly different execution times depending on how it's structured.

The evaluation uses a simulated e-commerce database with 1.6M+ rows across 5 tables. Your transformer receives SQL queries containing common inefficiencies (correlated subqueries, redundant operations, suboptimal joins) and must produce equivalent queries that execute faster. The schema and test queries can be easily adjusted to match your real-world database for domain-specific optimization.

The `get_reward(code_path)` function returns `(reward, error_msg, details)`:
- **reward**: Average speedup across all successful queries, where `speedup = original_time / transformed_time`. Higher is better (>1 means faster).
- **error_msg**: Error message if evaluation failed, None otherwise.
- **details**: String with per-query results (success/failure, speedup, error messages) for learning from specific failures.

---

## Task Description

**Goal:** Generate a Python function `transform_query(sql: str) -> str` that transforms SQL queries into optimized versions.

**Setup:**
- Database: SQLite with 1.6M+ rows
- Tables: users (100K), categories (100), products (10K), orders (500K), order_items (1M)
- Test queries: 25 queries across 8 optimization categories
- Database is cached at `/tmp/sql_query_optim_db.sqlite` (created on first run)

**Requirements:**
1. Function must be named `transform_query`
2. Must take a SQL string and return a SQL string
3. Transformed query must return **identical results** to the original
4. Transformed query should execute **faster** than the original
5. Must handle edge cases (empty results, NULL values)

**Things to Avoid:**
1. Returning queries that produce different results
2. Returning empty or invalid SQL
3. Infinite loops in transformation logic
4. Hardcoded transformations for specific query IDs

**Database Schema:**
```sql
-- Users table (100K rows)
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    city TEXT NOT NULL,
    age INTEGER NOT NULL,
    created_at DATE NOT NULL
);

-- Categories table (100 rows, hierarchical)
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id INTEGER  -- NULL for top-level
);

-- Products table (10K rows)
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    price REAL NOT NULL,
    stock INTEGER NOT NULL,
    created_at DATE NOT NULL
);

-- Orders table (500K rows)
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    total_amount REAL NOT NULL,
    status TEXT NOT NULL,
    created_at DATE NOT NULL
);

-- Order items table (1M rows)
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
```
---
## Reward Description
Average speedup across all successful queries, where `speedup = original_time / transformed_time`. Higher is better (>1 means faster).

## Initial Code

```python
import re

def transform_query(sql: str) -> str:
    """
    Transform an SQL query into an optimized version.
    
    This baseline implementation returns the query unchanged.
    
    Args:
        sql: Input SQL query string
        
    Returns:
        Optimized SQL query string (same query in this baseline)
    """
    # Baseline: return query unchanged
    return sql
```

---

## File Structure

```
iterx_code/sql_query_optim/
├── README.md              # This file
├── __init__.py            # Package exports
├── fake_database.py       # Database creation and caching
├── test_queries.py        # 25 test queries with hints
├── eval_sql_optim.py      # Evaluation function
├── initial_code.py        # Baseline transformer
└── run_iterx.py           # IterX evaluation runner
```

---

## Running iterX

```bash
python run_iterx.py
```

