"""
Test Queries for SQL Query Optimization Task (Enhanced Version)

Contains 25 diverse SQL queries across 7 difficulty levels designed to 
highlight algorithmic inefficiencies that can be improved through query transformation.

Query Categories:
1. Level 1 (Q01-Q03): Simple subqueries - 1 level nesting
2. Level 2 (Q04-Q06): Medium nesting - 2 levels
3. Level 3 (Q07-Q09): Deep nesting - 3+ levels
4. Level 4 (Q10-Q13): Window functions
5. Level 5 (Q14-Q16): CTEs
6. Level 6 (Q17-Q20): Complex multi-pattern
7. Level 7 (Q21-Q25): HIGH OPTIMIZATION POTENTIAL - 5x+ speedup possible
"""

from typing import List, Tuple

# Each query is a tuple of (query_id, query_sql, description, optimization_hint)
TEST_QUERIES: List[Tuple[str, str, str, str]] = [
    
    # ==========================================================================
    # Level 1: Simple Subqueries (Q01-Q03) - 1 level nesting
    # ==========================================================================
    
    ("Q01", """
        SELECT u.id, u.name, u.city,
            (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count
        FROM users u
        WHERE u.city = 'New York'
    """, 
    "Correlated subquery for order count per user",
    "Convert correlated subquery to LEFT JOIN with GROUP BY"),
    
    ("Q02", """
        SELECT p.id, p.name, p.price,
            (SELECT AVG(r.rating) FROM reviews r WHERE r.product_id = p.id) as avg_rating
        FROM products p
        WHERE p.price > 100
    """,
    "Correlated subquery for average rating",
    "Convert to LEFT JOIN with GROUP BY"),
    
    ("Q03", """
        SELECT * FROM products
        WHERE id NOT IN (SELECT DISTINCT product_id FROM order_items)
    """,
    "NOT IN for products never ordered",
    "Convert NOT IN to LEFT JOIN with NULL check"),
    
    # ==========================================================================
    # Level 2: Medium Nesting (Q04-Q06) - 2 levels
    # ==========================================================================
    
    ("Q04", """
        SELECT * FROM orders
        WHERE user_id IN (
            SELECT id FROM users WHERE city IN (
                SELECT city FROM users GROUP BY city HAVING COUNT(*) > 1000
            )
        )
    """,
    "2-level nested IN subqueries",
    "Flatten to JOINs"),
    
    ("Q05", """
        SELECT p.id, p.name,
            (SELECT COUNT(*) FROM reviews r WHERE r.product_id = p.id 
             AND r.user_id IN (SELECT id FROM users WHERE membership_level = 'gold'))
        FROM products p
        WHERE p.price > 50
    """,
    "Correlated subquery with nested IN",
    "Convert to JOINs with GROUP BY"),
    
    ("Q06", """
        SELECT u.id, u.name
        FROM users u
        WHERE NOT EXISTS (
            SELECT 1 FROM orders o WHERE o.user_id = u.id
            AND NOT EXISTS (
                SELECT 1 FROM shipping s WHERE s.order_id = o.id AND s.status = 'delivered'
            )
        )
    """,
    "Double NOT EXISTS (all orders delivered)",
    "Convert to LEFT JOINs with NULL checks"),
    
    # ==========================================================================
    # Level 3: Deep Nesting (Q07-Q09) - 3+ levels
    # ==========================================================================
    
    ("Q07", """
        SELECT * FROM users
        WHERE id IN (
            SELECT user_id FROM orders WHERE id IN (
                SELECT order_id FROM order_items WHERE product_id IN (
                    SELECT id FROM products WHERE category_id IN (
                        SELECT id FROM categories WHERE name LIKE 'Electronics%'
                    )
                )
            )
        )
    """,
    "4-level nested IN: users who bought electronics",
    "Flatten to chained JOINs"),
    
    ("Q08", """
        SELECT * FROM products
        WHERE id NOT IN (
            SELECT product_id FROM reviews WHERE user_id IN (
                SELECT id FROM users WHERE id IN (
                    SELECT user_id FROM orders WHERE status = 'cancelled'
                )
            )
        )
    """,
    "3-level NOT IN: products not reviewed by users with cancelled orders",
    "Convert to LEFT JOINs"),
    
    ("Q09", """
        SELECT s.id, s.name
        FROM suppliers s
        WHERE EXISTS (
            SELECT 1 FROM products p WHERE p.supplier_id = s.id
            AND EXISTS (
                SELECT 1 FROM order_items oi WHERE oi.product_id = p.id
                AND EXISTS (
                    SELECT 1 FROM orders o WHERE o.id = oi.order_id AND o.status = 'delivered'
                )
            )
        )
    """,
    "3-level nested EXISTS",
    "Convert to JOINs"),
    
    # ==========================================================================
    # Level 4: Window Functions (Q10-Q13)
    # ==========================================================================
    
    ("Q10", """
        SELECT o.id, o.user_id, o.total_amount,
            (SELECT COUNT(*) FROM orders o2 WHERE o2.user_id = o.user_id AND o2.id <= o.id) as order_rank
        FROM orders o
        WHERE o.status = 'delivered'
    """,
    "Running count via correlated subquery",
    "Use ROW_NUMBER() window function"),
    
    ("Q11", """
        SELECT o.id, o.user_id, o.total_amount,
            (SELECT SUM(o2.total_amount) FROM orders o2 
             WHERE o2.user_id = o.user_id AND o2.created_at <= o.created_at) as running_total
        FROM orders o
        WHERE o.created_at >= '2024-01-01'
    """,
    "Running total via correlated subquery",
    "Use SUM() OVER (PARTITION BY ... ORDER BY ...)"),
    
    ("Q12", """
        SELECT p.id, p.name, p.price,
            (SELECT price FROM products p2 
             WHERE p2.category_id = p.category_id 
             AND p2.created_at < p.created_at
             ORDER BY p2.created_at DESC LIMIT 1) as prev_product_price
        FROM products p
    """,
    "Previous value via correlated subquery",
    "Use LAG() window function"),
    
    ("Q13", """
        SELECT o.id, o.user_id, o.total_amount,
            o.total_amount - (SELECT AVG(o2.total_amount) FROM orders o2 WHERE o2.user_id = o.user_id) as diff_from_avg
        FROM orders o
        WHERE o.status = 'delivered'
    """,
    "Difference from average via correlated subquery",
    "Use AVG() OVER (PARTITION BY user_id)"),
    
    # ==========================================================================
    # Level 5: CTEs (Q14-Q16)
    # ==========================================================================
    
    ("Q14", """
        SELECT u.id, u.name, subq.order_count, subq.total_spent
        FROM users u
        JOIN (
            SELECT user_id, COUNT(*) as order_count, SUM(total_amount) as total_spent
            FROM orders
            GROUP BY user_id
        ) subq ON u.id = subq.user_id
        WHERE subq.order_count > 5
    """,
    "Derived table that could be CTE",
    "Convert to WITH clause for readability"),
    
    ("Q15", """
        SELECT u.id, u.name,
            (SELECT COUNT(*) FROM orders WHERE user_id = u.id) as order_count,
            (SELECT AVG(total_amount) FROM orders WHERE user_id = u.id) as avg_order,
            (SELECT MAX(total_amount) FROM orders WHERE user_id = u.id) as max_order
        FROM users u
        WHERE u.membership_level = 'platinum'
    """,
    "Multiple aggregates from same table",
    "Use CTE or single subquery with multiple aggregates"),
    
    ("Q16", """
        SELECT p.id, p.name, p.price,
            (SELECT AVG(r.rating) FROM reviews r WHERE r.product_id = p.id) as avg_rating,
            (SELECT COUNT(*) FROM reviews r WHERE r.product_id = p.id) as review_count,
            (SELECT SUM(quantity) FROM order_items oi WHERE oi.product_id = p.id) as total_sold
        FROM products p
        WHERE p.category_id IN (
            SELECT id FROM categories WHERE level = 2
        )
    """,
    "Multiple correlated subqueries with aggregates",
    "Use CTEs to pre-compute aggregations"),
    
    # ==========================================================================
    # Level 6: Complex Multi-Pattern (Q17-Q20)
    # ==========================================================================
    
    ("Q17", """
        SELECT u.id, u.name,
            (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count,
            (SELECT SUM(oi.quantity) FROM order_items oi 
             WHERE oi.order_id IN (SELECT id FROM orders WHERE user_id = u.id)) as items_bought,
            (SELECT AVG(r.rating) FROM reviews r WHERE r.user_id = u.id) as avg_rating_given,
            (SELECT COUNT(*) FROM wishlists w WHERE w.user_id = u.id) as wishlist_count
        FROM users u
        WHERE u.id IN (
            SELECT user_id FROM orders WHERE total_amount > 500
        )
        AND u.membership_level IN ('gold', 'platinum')
    """,
    "Multiple correlated subqueries + IN + filter",
    "Combine CTEs with JOINs"),
    
    ("Q18", """
        SELECT s.id, s.name, s.rating,
            (SELECT COUNT(*) FROM products p WHERE p.supplier_id = s.id) as product_count,
            (SELECT SUM(oi.quantity) FROM order_items oi
             WHERE oi.product_id IN (SELECT id FROM products WHERE supplier_id = s.id)) as total_items_sold,
            (SELECT AVG(r.rating) FROM reviews r
             WHERE r.product_id IN (SELECT id FROM products WHERE supplier_id = s.id)) as avg_product_rating
        FROM suppliers s
        WHERE s.rating > 4.0
        AND EXISTS (
            SELECT 1 FROM products p WHERE p.supplier_id = s.id
        )
    """,
    "Supplier analytics with nested subqueries",
    "Use CTEs and JOINs for aggregation"),
    
    ("Q19", """
        SELECT u.id, u.name, u.city,
            (SELECT o.total_amount FROM orders o WHERE o.user_id = u.id 
             ORDER BY o.created_at DESC LIMIT 1) as last_order_amount,
            (SELECT COUNT(*) FROM orders o2 WHERE o2.user_id = u.id) as total_orders,
            (SELECT SUM(total_amount) FROM orders WHERE user_id = u.id 
             AND created_at >= '2024-01-01') as recent_total
        FROM users u
        WHERE u.id IN (SELECT user_id FROM orders GROUP BY user_id HAVING COUNT(*) > 5)
    """,
    "User analytics with multiple correlated subqueries",
    "Use window functions + CTEs"),
    
    ("Q20", """
        SELECT p.id, p.name,
            (SELECT COUNT(*) FROM reviews r WHERE r.product_id = p.id 
             AND r.rating >= (SELECT AVG(rating) FROM reviews WHERE product_id = p.id)) as above_avg_reviews,
            (SELECT u.name FROM users u WHERE u.id = (
                SELECT user_id FROM orders WHERE id = (
                    SELECT order_id FROM order_items WHERE product_id = p.id 
                    GROUP BY order_id ORDER BY SUM(quantity) DESC LIMIT 1
                )
            )) as top_buyer,
            (SELECT SUM(quantity) FROM order_items WHERE product_id = p.id
             AND order_id IN (SELECT id FROM orders WHERE user_id IN (
                 SELECT id FROM users WHERE membership_level = 'platinum'
             ))) as platinum_purchases
        FROM products p
        WHERE p.price > (SELECT AVG(price) FROM products WHERE category_id = p.category_id)
        AND p.id IN (SELECT product_id FROM wishlists GROUP BY product_id HAVING COUNT(*) > 5)
    """,
    "Extreme nesting with aggregates, filters, and lookups",
    "Complete restructuring required"),
    
    # ==========================================================================
    # Level 7: High Optimization Potential (Q21-Q25) - 5x+ speedup possible
    # These queries have REDUNDANT or INEFFICIENT patterns that can be eliminated
    # ==========================================================================
    
    # Pattern: Redundant identity subquery (SELECT id WHERE id IN (SELECT id ...))
    ("Q21", """
        SELECT * FROM orders
        WHERE user_id IN (
            SELECT id FROM users WHERE id IN (
                SELECT id FROM users WHERE id IN (
                    SELECT user_id FROM orders WHERE total_amount > 1000
                )
            )
        )
    """,
    "Triple-redundant identity subquery chain",
    "Eliminate redundant levels: WHERE user_id IN (SELECT user_id FROM orders WHERE total_amount > 1000)"),
    
    # Pattern: Multiple NOT IN on same column that can be combined
    ("Q22", """
        SELECT * FROM products
        WHERE id NOT IN (
            SELECT product_id FROM order_items WHERE quantity < 5
        )
        AND id NOT IN (
            SELECT product_id FROM reviews WHERE rating < 3
        )
        AND id NOT IN (
            SELECT product_id FROM wishlists WHERE created_at < '2024-01-01'
        )
    """,
    "Multiple NOT IN anti-joins on same column",
    "Combine into single NOT EXISTS with OR, or use LEFT JOINs"),
    
    # Pattern: Deeply nested NOT IN with redundant user lookup (similar to Q08)
    ("Q23", """
        SELECT * FROM suppliers
        WHERE id NOT IN (
            SELECT supplier_id FROM products WHERE id IN (
                SELECT product_id FROM order_items WHERE order_id IN (
                    SELECT id FROM orders WHERE id IN (
                        SELECT order_id FROM shipping WHERE status = 'delayed'
                    )
                )
            )
        )
    """,
    "4-level nested NOT IN with redundant order lookup",
    "Flatten: NOT IN (SELECT supplier_id FROM products JOIN order_items JOIN shipping WHERE status='delayed')"),
    
    # Pattern: Repeated correlated subquery accessing same large table
    ("Q24", """
        SELECT o.id, o.user_id, o.total_amount,
            (SELECT COUNT(*) FROM order_items WHERE order_id = o.id) as item_count,
            (SELECT SUM(quantity) FROM order_items WHERE order_id = o.id) as total_qty,
            (SELECT AVG(unit_price) FROM order_items WHERE order_id = o.id) as avg_price,
            (SELECT MAX(quantity) FROM order_items WHERE order_id = o.id) as max_qty,
            (SELECT MIN(unit_price) FROM order_items WHERE order_id = o.id) as min_price
        FROM orders o
        WHERE o.status = 'delivered'
    """,
    "5 correlated subqueries to same table (order_items)",
    "Single JOIN with multiple aggregates: JOIN (SELECT order_id, COUNT(*), SUM(), AVG(), MAX(), MIN() FROM order_items GROUP BY order_id)"),
    
    # Pattern: Nested EXISTS with redundant table access
    ("Q25", """
        SELECT * FROM users
        WHERE id IN (
            SELECT user_id FROM orders WHERE user_id IN (
                SELECT user_id FROM orders WHERE user_id IN (
                    SELECT user_id FROM orders WHERE status = 'delivered' AND total_amount > 500
                )
            )
        )
    """,
    "Triple-redundant nested IN through same table",
    "Simplify to: WHERE id IN (SELECT user_id FROM orders WHERE status='delivered' AND total_amount > 500)"),
    
]


def get_test_queries() -> List[Tuple[str, str, str, str]]:
    """Get all test queries."""
    return TEST_QUERIES


def get_query_by_id(query_id: str) -> Tuple[str, str, str, str]:
    """Get a specific query by ID."""
    for q in TEST_QUERIES:
        if q[0] == query_id:
            return q
    raise ValueError(f"Query {query_id} not found")


def get_queries_by_level(level: int) -> List[Tuple[str, str, str, str]]:
    """
    Get queries by difficulty level (1-7).
    
    Level mapping:
    1: Q01-Q03 (Simple subqueries)
    2: Q04-Q06 (Medium nesting)
    3: Q07-Q09 (Deep nesting)
    4: Q10-Q13 (Window functions)
    5: Q14-Q16 (CTEs)
    6: Q17-Q20 (Complex multi-pattern)
    7: Q21-Q25 (High optimization potential - 5x+ speedup)
    """
    ranges = {
        1: (1, 3),
        2: (4, 6),
        3: (7, 9),
        4: (10, 13),
        5: (14, 16),
        6: (17, 20),
        7: (21, 25),
    }
    
    if level not in ranges:
        raise ValueError(f"Level must be 1-7, got {level}")
    
    start, end = ranges[level]
    return [q for q in TEST_QUERIES if start <= int(q[0][1:]) <= end]


if __name__ == "__main__":
    print(f"Total queries: {len(TEST_QUERIES)}")
    print("\nQueries by level:")
    for level in range(1, 8):
        queries = get_queries_by_level(level)
        print(f"  Level {level}: {len(queries)} queries ({queries[0][0]}-{queries[-1][0]})")
    
    print("\n\nLevel 7 (High Optimization Potential):")
    for q in get_queries_by_level(7):
        print(f"  {q[0]}: {q[2]}")
        print(f"       Hint: {q[3]}")
