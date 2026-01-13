"""
SQL Query Optimization Task

This module provides tools for evaluating SQL query transformations.
The goal is to optimize SQL queries for better performance while
maintaining correctness.
"""

try:
    from .eval_sql_optim import get_reward
from .fake_database import get_connection, get_schema_info
from .test_queries import get_test_queries, get_query_by_id
except ImportError:
    from eval_sql_optim import get_reward
    from fake_database import get_connection, get_schema_info
    from test_queries import get_test_queries, get_query_by_id

__all__ = [
    'get_reward',
    'get_connection',
    'get_schema_info',
    'get_test_queries',
    'get_query_by_id',
]
