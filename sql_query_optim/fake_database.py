"""
Fake Database for SQL Query Optimization Task (Enhanced Version)

Creates a cached SQLite database with 12 tables and 3M+ rows for testing
SQL query transformations.

Schema:
- users (100K rows)
- categories (100 rows, hierarchical)
- products (10K rows)
- orders (500K rows)
- order_items (1M rows)
- suppliers (1K rows)
- inventory (50K rows)
- warehouses (50 rows)
- reviews (500K rows)
- wishlists (100K rows)
- shipping (500K rows)
- promotions (1K rows)

The database is created once and cached to disk for reuse across evaluations.
"""

import sqlite3
import os
import random
import string
import fcntl
import time
from typing import Optional
from datetime import datetime, timedelta

# Database paths
DB_PATH = "/tmp/sql_query_optim_db_v2.sqlite"
LOCK_PATH = "/tmp/sql_query_optim_db_v2.lock"

# Data generation parameters (reduced for faster evaluation)
NUM_USERS = 20_000
NUM_CATEGORIES = 100
NUM_PRODUCTS = 5_000
NUM_ORDERS = 100_000
NUM_ORDER_ITEMS = 200_000
NUM_SUPPLIERS = 500
NUM_WAREHOUSES = 50
NUM_INVENTORY = 10_000
NUM_REVIEWS = 100_000
NUM_WISHLISTS = 20_000
NUM_SHIPPING = 100_000
NUM_PROMOTIONS = 500

# Random data pools
CITIES = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
          'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
          'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
          'Seattle', 'Denver', 'Boston', 'Detroit', 'Portland']

COUNTRIES = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia', 'Brazil', 'India', 'China']

CATEGORY_NAMES = ['Electronics', 'Books', 'Clothing', 'Home & Garden', 'Sports',
                  'Toys', 'Automotive', 'Health', 'Beauty', 'Food']

ORDER_STATUSES = ['pending', 'processing', 'shipped', 'delivered', 'cancelled', 'returned']

SHIPPING_METHODS = ['standard', 'express', 'overnight', 'pickup']

SHIPPING_STATUSES = ['preparing', 'in_transit', 'out_for_delivery', 'delivered', 'failed']


def random_string(length: int) -> str:
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def random_date(start_year: int = 2020, end_year: int = 2024) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    date = start + timedelta(days=random_days)
    return date.strftime('%Y-%m-%d')


def random_email(name: str) -> str:
    domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'example.com']
    return f"{name.lower().replace(' ', '.')}@{random.choice(domains)}"


def _create_schema(cursor):
    """Create database schema with all 12 tables and indexes"""
    cursor.executescript("""
        -- Users table
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            city TEXT NOT NULL,
            country TEXT NOT NULL,
            age INTEGER NOT NULL,
            membership_level TEXT NOT NULL,
            created_at DATE NOT NULL
        );
        
        -- Categories table (hierarchical, 3 levels)
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id INTEGER,
            level INTEGER NOT NULL,
            FOREIGN KEY (parent_id) REFERENCES categories(id)
        );
        
        -- Suppliers table
        CREATE TABLE suppliers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            country TEXT NOT NULL,
            rating REAL NOT NULL,
            active INTEGER NOT NULL
        );
        
        -- Warehouses table
        CREATE TABLE warehouses (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            country TEXT NOT NULL,
            capacity INTEGER NOT NULL
        );
        
        -- Products table
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            supplier_id INTEGER NOT NULL,
            price REAL NOT NULL,
            cost REAL NOT NULL,
            stock INTEGER NOT NULL,
            rating REAL,
            created_at DATE NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(id),
            FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
        );
        
        -- Inventory table (product stock per warehouse)
        CREATE TABLE inventory (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            warehouse_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            last_updated DATE NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (warehouse_id) REFERENCES warehouses(id)
        );
        
        -- Promotions table
        CREATE TABLE promotions (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            discount_percent REAL NOT NULL,
            min_order_amount REAL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            active INTEGER NOT NULL
        );
        
        -- Orders table
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            promotion_id INTEGER,
            total_amount REAL NOT NULL,
            discount_amount REAL NOT NULL,
            status TEXT NOT NULL,
            created_at DATE NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (promotion_id) REFERENCES promotions(id)
        );
        
        -- Order items table
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        
        -- Shipping table
        CREATE TABLE shipping (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            warehouse_id INTEGER NOT NULL,
            method TEXT NOT NULL,
            status TEXT NOT NULL,
            shipped_date DATE,
            delivered_date DATE,
            tracking_number TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (warehouse_id) REFERENCES warehouses(id)
        );
        
        -- Reviews table
        CREATE TABLE reviews (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            order_id INTEGER,
            rating INTEGER NOT NULL,
            title TEXT,
            content TEXT,
            helpful_votes INTEGER NOT NULL,
            created_at DATE NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (order_id) REFERENCES orders(id)
        );
        
        -- Wishlists table
        CREATE TABLE wishlists (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            added_at DATE NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        
        -- Indexes for query optimization
        CREATE INDEX idx_users_city ON users(city);
        CREATE INDEX idx_users_country ON users(country);
        CREATE INDEX idx_users_age ON users(age);
        CREATE INDEX idx_users_membership ON users(membership_level);
        CREATE INDEX idx_users_created ON users(created_at);
        
        CREATE INDEX idx_categories_parent ON categories(parent_id);
        CREATE INDEX idx_categories_level ON categories(level);
        
        CREATE INDEX idx_products_category ON products(category_id);
        CREATE INDEX idx_products_supplier ON products(supplier_id);
        CREATE INDEX idx_products_price ON products(price);
        CREATE INDEX idx_products_rating ON products(rating);
        
        CREATE INDEX idx_inventory_product ON inventory(product_id);
        CREATE INDEX idx_inventory_warehouse ON inventory(warehouse_id);
        
        CREATE INDEX idx_orders_user ON orders(user_id);
        CREATE INDEX idx_orders_promotion ON orders(promotion_id);
        CREATE INDEX idx_orders_status ON orders(status);
        CREATE INDEX idx_orders_created ON orders(created_at);
        CREATE INDEX idx_orders_amount ON orders(total_amount);
        
        CREATE INDEX idx_order_items_order ON order_items(order_id);
        CREATE INDEX idx_order_items_product ON order_items(product_id);
        
        CREATE INDEX idx_shipping_order ON shipping(order_id);
        CREATE INDEX idx_shipping_warehouse ON shipping(warehouse_id);
        CREATE INDEX idx_shipping_status ON shipping(status);
        
        CREATE INDEX idx_reviews_user ON reviews(user_id);
        CREATE INDEX idx_reviews_product ON reviews(product_id);
        CREATE INDEX idx_reviews_order ON reviews(order_id);
        CREATE INDEX idx_reviews_rating ON reviews(rating);
        
        CREATE INDEX idx_wishlists_user ON wishlists(user_id);
        CREATE INDEX idx_wishlists_product ON wishlists(product_id);
        
        CREATE INDEX idx_suppliers_country ON suppliers(country);
        CREATE INDEX idx_suppliers_rating ON suppliers(rating);
        
        CREATE INDEX idx_promotions_active ON promotions(active);
        CREATE INDEX idx_promotions_dates ON promotions(start_date, end_date);
    """)


def _populate_users(cursor):
    print(f"  Populating users ({NUM_USERS} rows)...")
    membership_levels = ['bronze', 'silver', 'gold', 'platinum']
    batch_size = 10000
    for batch_start in range(0, NUM_USERS, batch_size):
        batch_end = min(batch_start + batch_size, NUM_USERS)
        data = []
        for i in range(batch_start, batch_end):
            name = f"User_{i}"
            email = random_email(name)
            city = random.choice(CITIES)
            country = random.choice(COUNTRIES)
            age = random.randint(18, 80)
            membership = random.choice(membership_levels)
            created_at = random_date(2018, 2023)
            data.append((i, name, email, city, country, age, membership, created_at))
        cursor.executemany("INSERT INTO users VALUES (?,?,?,?,?,?,?,?)", data)


def _populate_categories(cursor):
    print(f"  Populating categories ({NUM_CATEGORIES} rows)...")
    data = []
    # Level 0: Top-level categories (10)
    for i, name in enumerate(CATEGORY_NAMES):
        data.append((i, name, None, 0))
    # Level 1: Sub-categories (30)
    for i in range(10, 40):
        parent_id = random.randint(0, 9)
        name = f"{CATEGORY_NAMES[parent_id]}_Sub{i-10}"
        data.append((i, name, parent_id, 1))
    # Level 2: Sub-sub-categories (60)
    for i in range(40, NUM_CATEGORIES):
        parent_id = random.randint(10, 39)
        name = f"Category_{i}"
        data.append((i, name, parent_id, 2))
    cursor.executemany("INSERT INTO categories VALUES (?,?,?,?)", data)


def _populate_suppliers(cursor):
    print(f"  Populating suppliers ({NUM_SUPPLIERS} rows)...")
    data = []
    for i in range(NUM_SUPPLIERS):
        name = f"Supplier_{i}"
        country = random.choice(COUNTRIES)
        rating = round(random.uniform(1.0, 5.0), 1)
        active = 1 if random.random() > 0.1 else 0
        data.append((i, name, country, rating, active))
    cursor.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?)", data)


def _populate_warehouses(cursor):
    print(f"  Populating warehouses ({NUM_WAREHOUSES} rows)...")
    data = []
    for i in range(NUM_WAREHOUSES):
        name = f"Warehouse_{i}"
        city = random.choice(CITIES)
        country = random.choice(COUNTRIES)
        capacity = random.randint(10000, 100000)
        data.append((i, name, city, country, capacity))
    cursor.executemany("INSERT INTO warehouses VALUES (?,?,?,?,?)", data)


def _populate_products(cursor):
    print(f"  Populating products ({NUM_PRODUCTS} rows)...")
    batch_size = 5000
    for batch_start in range(0, NUM_PRODUCTS, batch_size):
        batch_end = min(batch_start + batch_size, NUM_PRODUCTS)
        data = []
        for i in range(batch_start, batch_end):
            name = f"Product_{i}"
            category_id = random.randint(0, NUM_CATEGORIES - 1)
            supplier_id = random.randint(0, NUM_SUPPLIERS - 1)
            price = round(random.uniform(1.0, 999.99), 2)
            cost = round(price * random.uniform(0.3, 0.7), 2)
            stock = random.randint(0, 1000)
            rating = round(random.uniform(1.0, 5.0), 1) if random.random() > 0.1 else None
            created_at = random_date(2019, 2023)
            data.append((i, name, category_id, supplier_id, price, cost, stock, rating, created_at))
        cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", data)


def _populate_inventory(cursor):
    print(f"  Populating inventory ({NUM_INVENTORY} rows)...")
    batch_size = 10000
    for batch_start in range(0, NUM_INVENTORY, batch_size):
        batch_end = min(batch_start + batch_size, NUM_INVENTORY)
        data = []
        for i in range(batch_start, batch_end):
            product_id = random.randint(0, NUM_PRODUCTS - 1)
            warehouse_id = random.randint(0, NUM_WAREHOUSES - 1)
            quantity = random.randint(0, 500)
            last_updated = random_date(2023, 2024)
            data.append((i, product_id, warehouse_id, quantity, last_updated))
        cursor.executemany("INSERT INTO inventory VALUES (?,?,?,?,?)", data)


def _populate_promotions(cursor):
    print(f"  Populating promotions ({NUM_PROMOTIONS} rows)...")
    data = []
    for i in range(NUM_PROMOTIONS):
        code = f"PROMO{i:04d}"
        discount = round(random.uniform(5, 50), 0)
        min_amount = round(random.uniform(20, 200), 2) if random.random() > 0.3 else None
        start_date = random_date(2022, 2023)
        end_date = random_date(2024, 2025)
        active = 1 if random.random() > 0.3 else 0
        data.append((i, code, discount, min_amount, start_date, end_date, active))
    cursor.executemany("INSERT INTO promotions VALUES (?,?,?,?,?,?,?)", data)


def _populate_orders(cursor):
    print(f"  Populating orders ({NUM_ORDERS} rows)...")
    batch_size = 50000
    for batch_start in range(0, NUM_ORDERS, batch_size):
        batch_end = min(batch_start + batch_size, NUM_ORDERS)
        data = []
        for i in range(batch_start, batch_end):
            user_id = random.randint(0, NUM_USERS - 1)
            promotion_id = random.randint(0, NUM_PROMOTIONS - 1) if random.random() > 0.7 else None
            total_amount = round(random.uniform(10.0, 5000.0), 2)
            discount_amount = round(total_amount * random.uniform(0, 0.3), 2) if promotion_id else 0
            status = random.choice(ORDER_STATUSES)
            created_at = random_date(2022, 2024)
            data.append((i, user_id, promotion_id, total_amount, discount_amount, status, created_at))
        cursor.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", data)


def _populate_order_items(cursor):
    print(f"  Populating order_items ({NUM_ORDER_ITEMS} rows)...")
    batch_size = 100000
    for batch_start in range(0, NUM_ORDER_ITEMS, batch_size):
        batch_end = min(batch_start + batch_size, NUM_ORDER_ITEMS)
        data = []
        for i in range(batch_start, batch_end):
            order_id = random.randint(0, NUM_ORDERS - 1)
            product_id = random.randint(0, NUM_PRODUCTS - 1)
            quantity = random.randint(1, 10)
            unit_price = round(random.uniform(1.0, 500.0), 2)
            data.append((i, order_id, product_id, quantity, unit_price))
        cursor.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", data)


def _populate_shipping(cursor):
    print(f"  Populating shipping ({NUM_SHIPPING} rows)...")
    batch_size = 50000
    for batch_start in range(0, NUM_SHIPPING, batch_size):
        batch_end = min(batch_start + batch_size, NUM_SHIPPING)
        data = []
        for i in range(batch_start, batch_end):
            order_id = random.randint(0, NUM_ORDERS - 1)
            warehouse_id = random.randint(0, NUM_WAREHOUSES - 1)
            method = random.choice(SHIPPING_METHODS)
            status = random.choice(SHIPPING_STATUSES)
            shipped_date = random_date(2022, 2024) if status != 'preparing' else None
            delivered_date = random_date(2022, 2024) if status == 'delivered' else None
            tracking = f"TRK{i:08d}" if shipped_date else None
            data.append((i, order_id, warehouse_id, method, status, shipped_date, delivered_date, tracking))
        cursor.executemany("INSERT INTO shipping VALUES (?,?,?,?,?,?,?,?)", data)


def _populate_reviews(cursor):
    print(f"  Populating reviews ({NUM_REVIEWS} rows)...")
    batch_size = 50000
    for batch_start in range(0, NUM_REVIEWS, batch_size):
        batch_end = min(batch_start + batch_size, NUM_REVIEWS)
        data = []
        for i in range(batch_start, batch_end):
            user_id = random.randint(0, NUM_USERS - 1)
            product_id = random.randint(0, NUM_PRODUCTS - 1)
            order_id = random.randint(0, NUM_ORDERS - 1) if random.random() > 0.2 else None
            rating = random.randint(1, 5)
            title = f"Review_{i}" if random.random() > 0.3 else None
            content = f"Content for review {i}" if random.random() > 0.2 else None
            helpful_votes = random.randint(0, 100)
            created_at = random_date(2022, 2024)
            data.append((i, user_id, product_id, order_id, rating, title, content, helpful_votes, created_at))
        cursor.executemany("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?)", data)


def _populate_wishlists(cursor):
    print(f"  Populating wishlists ({NUM_WISHLISTS} rows)...")
    batch_size = 50000
    for batch_start in range(0, NUM_WISHLISTS, batch_size):
        batch_end = min(batch_start + batch_size, NUM_WISHLISTS)
        data = []
        for i in range(batch_start, batch_end):
            user_id = random.randint(0, NUM_USERS - 1)
            product_id = random.randint(0, NUM_PRODUCTS - 1)
            added_at = random_date(2022, 2024)
            data.append((i, user_id, product_id, added_at))
        cursor.executemany("INSERT INTO wishlists VALUES (?,?,?,?)", data)


def _create_database():
    """Create and populate the database"""
    random.seed(42)
    
    t1 = time.time()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    
    print("Creating schema...")
    _create_schema(cursor)
    
    print("Populating tables...")
    _populate_users(cursor)
    _populate_categories(cursor)
    _populate_suppliers(cursor)
    _populate_warehouses(cursor)
    _populate_products(cursor)
    _populate_inventory(cursor)
    _populate_promotions(cursor)
    _populate_orders(cursor)
    _populate_order_items(cursor)
    _populate_shipping(cursor)
    _populate_reviews(cursor)
    _populate_wishlists(cursor)
    
    print("Committing...")
    conn.commit()
    
    print("Analyzing tables...")
    cursor.execute("ANALYZE")
    conn.commit()
    
    conn.close()
    
    t2 = time.time()
    print(f"Database created in {t2-t1:.1f} seconds")
    
    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Database size: {size_mb:.1f} MB")


def get_connection(readonly: bool = True) -> sqlite3.Connection:
    """Get a database connection, creating the database if it doesn't exist."""
    if os.path.exists(DB_PATH):
        if readonly:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        return conn
    
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    
    with open(LOCK_PATH, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not os.path.exists(DB_PATH):
                print("First run: Creating database (this may take ~60 seconds)...")
                _create_database()
                print("Database ready!")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    
    if readonly:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    
    return conn


def get_schema_info() -> dict:
    """Get database schema information for the query transformer."""
    return {
        "tables": {
            "users": ["id", "name", "email", "city", "country", "age", "membership_level", "created_at"],
            "categories": ["id", "name", "parent_id", "level"],
            "suppliers": ["id", "name", "country", "rating", "active"],
            "warehouses": ["id", "name", "city", "country", "capacity"],
            "products": ["id", "name", "category_id", "supplier_id", "price", "cost", "stock", "rating", "created_at"],
            "inventory": ["id", "product_id", "warehouse_id", "quantity", "last_updated"],
            "promotions": ["id", "code", "discount_percent", "min_order_amount", "start_date", "end_date", "active"],
            "orders": ["id", "user_id", "promotion_id", "total_amount", "discount_amount", "status", "created_at"],
            "order_items": ["id", "order_id", "product_id", "quantity", "unit_price"],
            "shipping": ["id", "order_id", "warehouse_id", "method", "status", "shipped_date", "delivered_date", "tracking_number"],
            "reviews": ["id", "user_id", "product_id", "order_id", "rating", "title", "content", "helpful_votes", "created_at"],
            "wishlists": ["id", "user_id", "product_id", "added_at"],
        },
        "cardinality": {
            "users": NUM_USERS,
            "categories": NUM_CATEGORIES,
            "suppliers": NUM_SUPPLIERS,
            "warehouses": NUM_WAREHOUSES,
            "products": NUM_PRODUCTS,
            "inventory": NUM_INVENTORY,
            "promotions": NUM_PROMOTIONS,
            "orders": NUM_ORDERS,
            "order_items": NUM_ORDER_ITEMS,
            "shipping": NUM_SHIPPING,
            "reviews": NUM_REVIEWS,
            "wishlists": NUM_WISHLISTS,
        }
    }


def reset_database():
    """Delete and recreate the database"""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(LOCK_PATH):
        os.remove(LOCK_PATH)
    get_connection()


if __name__ == "__main__":
    print("Testing fake database...")
    if os.path.exists(DB_PATH):
        print(f"Removing existing database: {DB_PATH}")
        os.remove(DB_PATH)
    
    conn = get_connection()
    
    print("\nTable row counts:")
    for table in ["users", "categories", "suppliers", "warehouses", "products", 
                  "inventory", "promotions", "orders", "order_items", "shipping", 
                  "reviews", "wishlists"]:
        result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table}: {result[0]:,}")
    
    conn.close()
    print("\nDone!")
