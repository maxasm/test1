"""
Training data for Vanna AI agent.

This module contains DDL documentation and example SQL queries that can be
used to train the agent for better query generation accuracy.

Run this script to seed the agent memory with training data:
    python training_data.py
"""

# -----------------------------------------------------------------------------
# DDL Documentation - Schema descriptions for the agent
# -----------------------------------------------------------------------------
DDL_DOCUMENTATION = """
## Database Schema: E-Commerce Platform

### Table: users
Stores customer information.
- id (INT, PK): Unique user identifier
- email (VARCHAR 255, UNIQUE): User's email address
- full_name (VARCHAR 255): User's full name
- created_at (DATETIME): Account creation timestamp

### Table: products
Product catalog.
- id (INT, PK): Unique product identifier
- name (VARCHAR 255): Product name
- category (VARCHAR 100, INDEXED): Product category (e.g., 'electronics', 'furniture', 'stationery', 'grocery')
- price (DECIMAL 10,2): Product price in USD
- created_at (DATETIME): Product creation timestamp

### Table: orders
Customer orders.
- id (INT, PK): Unique order identifier
- user_id (INT, FK -> users.id): Customer who placed the order
- status (ENUM): Order status - 'pending', 'paid', 'shipped', 'cancelled', 'refunded'
- order_date (DATETIME, INDEXED): When the order was placed
- total_amount (DECIMAL 10,2): Total order value in USD

### Table: order_items
Line items within an order.
- id (INT, PK): Unique item identifier
- order_id (INT, FK -> orders.id): Parent order
- product_id (INT, FK -> products.id): Product purchased
- quantity (INT): Number of units
- unit_price (DECIMAL 10,2): Price per unit at time of purchase
- line_total (DECIMAL 10,2): quantity * unit_price

### Table: payments
Payment records for orders.
- id (INT, PK): Unique payment identifier
- order_id (INT, FK -> orders.id, UNIQUE): One payment per order
- provider (ENUM): Payment provider - 'card', 'paypal', 'bank'
- amount (DECIMAL 10,2): Payment amount in USD
- status (ENUM): Payment status - 'pending', 'completed', 'failed', 'refunded'
- paid_at (DATETIME, NULL): When payment was completed

### Relationships
- users 1:N orders (one user can have many orders)
- orders 1:N order_items (one order can have many items)
- products 1:N order_items (one product can be in many order items)
- orders 1:1 payments (one payment per order)
"""

# -----------------------------------------------------------------------------
# Example Queries - Golden queries for common questions
# -----------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    {
        "question": "Show me all users",
        "sql": "SELECT * FROM users ORDER BY created_at DESC",
    },
    {
        "question": "List all products",
        "sql": "SELECT * FROM products ORDER BY name",
    },
    {
        "question": "Show me all orders",
        "sql": "SELECT o.*, u.email, u.full_name FROM orders o JOIN users u ON o.user_id = u.id ORDER BY o.order_date DESC",
    },
    {
        "question": "Top 5 products by revenue",
        "sql": """SELECT p.name, p.category, SUM(oi.line_total) as total_revenue
FROM products p
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE o.status NOT IN ('cancelled', 'refunded')
GROUP BY p.id, p.name, p.category
ORDER BY total_revenue DESC
LIMIT 5""",
    },
    {
        "question": "Top customers by total spending",
        "sql": """SELECT u.id, u.email, u.full_name, SUM(o.total_amount) as total_spent
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.status IN ('paid', 'shipped')
GROUP BY u.id, u.email, u.full_name
ORDER BY total_spent DESC
LIMIT 10""",
    },
    {
        "question": "Monthly revenue",
        "sql": """SELECT 
    DATE_FORMAT(o.order_date, '%Y-%m') as month,
    COUNT(DISTINCT o.id) as order_count,
    SUM(o.total_amount) as revenue
FROM orders o
WHERE o.status IN ('paid', 'shipped')
GROUP BY DATE_FORMAT(o.order_date, '%Y-%m')
ORDER BY month DESC""",
    },
    {
        "question": "Orders by status",
        "sql": """SELECT status, COUNT(*) as count, SUM(total_amount) as total_value
FROM orders
GROUP BY status
ORDER BY count DESC""",
    },
    {
        "question": "Products by category",
        "sql": """SELECT category, COUNT(*) as product_count, AVG(price) as avg_price
FROM products
GROUP BY category
ORDER BY product_count DESC""",
    },
    {
        "question": "Recent orders from this month",
        "sql": """SELECT o.*, u.email, u.full_name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.order_date >= DATE_FORMAT(CURDATE(), '%Y-%m-01')
ORDER BY o.order_date DESC""",
    },
    {
        "question": "Users who signed up this month",
        "sql": """SELECT * FROM users
WHERE created_at >= DATE_FORMAT(CURDATE(), '%Y-%m-01')
ORDER BY created_at DESC""",
    },
    {
        "question": "Payment success rate",
        "sql": """SELECT 
    provider,
    COUNT(*) as total,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
    ROUND(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate
FROM payments
GROUP BY provider""",
    },
    {
        "question": "Order details with items",
        "sql": """SELECT 
    o.id as order_id,
    o.order_date,
    o.status,
    u.full_name as customer,
    p.name as product,
    oi.quantity,
    oi.unit_price,
    oi.line_total
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
ORDER BY o.order_date DESC, o.id, oi.id""",
    },
    {
        "question": "Average order value",
        "sql": """SELECT 
    ROUND(AVG(total_amount), 2) as avg_order_value,
    ROUND(MIN(total_amount), 2) as min_order,
    ROUND(MAX(total_amount), 2) as max_order
FROM orders
WHERE status NOT IN ('cancelled')""",
    },
    {
        "question": "Cancelled or refunded orders",
        "sql": """SELECT o.*, u.email, u.full_name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.status IN ('cancelled', 'refunded')
ORDER BY o.order_date DESC""",
    },
    {
        "question": "Products never ordered",
        "sql": """SELECT p.*
FROM products p
LEFT JOIN order_items oi ON p.id = oi.product_id
WHERE oi.id IS NULL""",
    },
]

# -----------------------------------------------------------------------------
# Business Rules - Context for the agent
# -----------------------------------------------------------------------------
BUSINESS_RULES = """
## Business Rules

1. **Revenue Calculations**: Only count orders with status 'paid' or 'shipped' for revenue metrics. Exclude 'cancelled' and 'refunded' orders.

2. **Active Users**: Users who have placed at least one order in the last 30 days.

3. **Order Value**: The total_amount in orders table already includes the sum of all line items.

4. **Payment Status**: A payment with status 'completed' means successful payment. 'failed' means the payment was rejected.

5. **Currency**: All monetary values are in USD.

6. **Date Formats**: Use DATE_FORMAT(date, '%Y-%m') for monthly grouping, DATE_FORMAT(date, '%Y-%m-%d') for daily.

7. **Current Period Queries**: Use CURDATE() for current date, DATE_FORMAT(CURDATE(), '%Y-%m-01') for first day of current month.
"""


def get_all_training_content() -> list[str]:
    """Get all training content as a list of strings to save as text memories."""
    content = [DDL_DOCUMENTATION, BUSINESS_RULES]
    
    # Format example queries
    for example in EXAMPLE_QUERIES:
        content.append(
            f"Question: {example['question']}\nSQL: {example['sql']}"
        )
    
    return content


if __name__ == "__main__":
    # When run directly, print the training content
    print("=== DDL Documentation ===")
    print(DDL_DOCUMENTATION)
    print("\n=== Business Rules ===")
    print(BUSINESS_RULES)
    print("\n=== Example Queries ===")
    for i, example in enumerate(EXAMPLE_QUERIES, 1):
        print(f"\n{i}. {example['question']}")
        print(f"   SQL: {example['sql'][:100]}...")
