"""
Initial/Baseline SQL Query Transformer

This is the starting point for the query optimization task.
The baseline transformer returns queries unchanged (identity function).

The goal is to improve this into a transformer that can recognize
inefficient SQL patterns and rewrite them for better performance
while maintaining correctness.
"""

import re
from typing import Optional


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


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_whitespace(sql: str) -> str:
    """Normalize whitespace in SQL query"""
    return ' '.join(sql.split())


def extract_select_columns(sql: str) -> Optional[str]:
    """Extract the column list from a SELECT statement"""
    match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else None


def extract_from_clause(sql: str) -> Optional[str]:
    """Extract the FROM clause (tables) from a query"""
    match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|LIMIT|$)', 
                      sql, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def extract_where_clause(sql: str) -> Optional[str]:
    """Extract the WHERE clause from a query"""
    match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', 
                      sql, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def has_correlated_subquery(sql: str) -> bool:
    """Check if query likely contains a correlated subquery"""
    # Look for patterns like "WHERE x.col = outer.col" inside subqueries
    return bool(re.search(r'\(\s*SELECT.*?WHERE.*?\.\w+\s*=\s*\w+\.\w+.*?\)', 
                          sql, re.IGNORECASE | re.DOTALL))


def has_in_subquery(sql: str) -> bool:
    """Check if query contains IN (subquery) pattern"""
    return bool(re.search(r'\bIN\s*\(\s*SELECT', sql, re.IGNORECASE))


def has_exists_subquery(sql: str) -> bool:
    """Check if query contains EXISTS (subquery) pattern"""
    return bool(re.search(r'\bEXISTS\s*\(\s*SELECT', sql, re.IGNORECASE))


def has_union(sql: str) -> bool:
    """Check if query contains UNION"""
    return bool(re.search(r'\bUNION\b', sql, re.IGNORECASE))


def has_distinct(sql: str) -> bool:
    """Check if query uses DISTINCT"""
    return bool(re.search(r'\bDISTINCT\b', sql, re.IGNORECASE))


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    test_query = """
        SELECT u.id, u.name, u.city,
            (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count
        FROM users u
        WHERE u.city = 'New York'
    """
    
    print("Test query:")
    print(test_query)
    
    print("\nTransformed query:")
    print(transform_query(test_query))
    
    print("\nAnalysis:")
    print(f"  Has correlated subquery: {has_correlated_subquery(test_query)}")
    print(f"  Has IN subquery: {has_in_subquery(test_query)}")
    print(f"  Has EXISTS subquery: {has_exists_subquery(test_query)}")
    print(f"  Has UNION: {has_union(test_query)}")
    print(f"  Has DISTINCT: {has_distinct(test_query)}")

