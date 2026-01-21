SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  email VARCHAR(255) NOT NULL,
  full_name VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY ux_users_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE products (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  category VARCHAR(100) NOT NULL,
  price DECIMAL(10,2) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY ix_products_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE orders (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  user_id INT UNSIGNED NOT NULL,
  status ENUM('pending','paid','shipped','cancelled','refunded') NOT NULL DEFAULT 'pending',
  order_date DATETIME NOT NULL,
  total_amount DECIMAL(10,2) NOT NULL DEFAULT 0.00,
  PRIMARY KEY (id),
  KEY ix_orders_user_id (user_id),
  KEY ix_orders_order_date (order_date),
  CONSTRAINT fk_orders_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE RESTRICT
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE order_items (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  order_id INT UNSIGNED NOT NULL,
  product_id INT UNSIGNED NOT NULL,
  quantity INT UNSIGNED NOT NULL,
  unit_price DECIMAL(10,2) NOT NULL,
  line_total DECIMAL(10,2) NOT NULL,
  PRIMARY KEY (id),
  KEY ix_order_items_order_id (order_id),
  KEY ix_order_items_product_id (product_id),
  CONSTRAINT fk_order_items_order
    FOREIGN KEY (order_id) REFERENCES orders(id)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT fk_order_items_product
    FOREIGN KEY (product_id) REFERENCES products(id)
    ON DELETE RESTRICT
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE payments (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  order_id INT UNSIGNED NOT NULL,
  provider ENUM('card','paypal','bank') NOT NULL,
  amount DECIMAL(10,2) NOT NULL,
  status ENUM('pending','completed','failed','refunded') NOT NULL DEFAULT 'pending',
  paid_at DATETIME NULL,
  PRIMARY KEY (id),
  UNIQUE KEY ux_payments_order_id (order_id),
  CONSTRAINT fk_payments_order
    FOREIGN KEY (order_id) REFERENCES orders(id)
    ON DELETE CASCADE
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO users (email, full_name, created_at) VALUES
  ('alice@example.com', 'Alice Johnson', '2025-12-02 10:15:00'),
  ('bob@example.com', 'Bob Smith', '2025-12-10 14:20:00'),
  ('carol@example.com', 'Carol White', '2026-01-03 09:05:00'),
  ('dave@example.com', 'Dave Brown', '2026-01-07 16:45:00'),
  ('erin@example.com', 'Erin Davis', '2026-01-15 11:30:00');

INSERT INTO products (name, category, price, created_at) VALUES
  ('Wireless Mouse', 'electronics', 25.99, '2025-11-25 08:00:00'),
  ('Mechanical Keyboard', 'electronics', 89.00, '2025-11-25 08:00:00'),
  ('USB-C Cable', 'electronics', 9.50, '2025-12-01 08:00:00'),
  ('Office Chair', 'furniture', 199.99, '2025-12-05 08:00:00'),
  ('Notebook Pack', 'stationery', 12.49, '2025-12-20 08:00:00'),
  ('Coffee Beans 1kg', 'grocery', 18.75, '2026-01-05 08:00:00');

INSERT INTO orders (user_id, status, order_date, total_amount) VALUES
  (1, 'paid', '2025-12-12 12:00:00', 115.48),
  (2, 'shipped', '2025-12-22 09:30:00', 199.99),
  (3, 'paid', '2026-01-08 18:10:00', 44.99),
  (4, 'cancelled', '2026-01-10 13:05:00', 12.49),
  (5, 'paid', '2026-01-18 07:55:00', 37.50),
  (1, 'refunded', '2026-01-19 15:40:00', 25.99);

INSERT INTO order_items (order_id, product_id, quantity, unit_price, line_total) VALUES
  (1, 1, 1, 25.99, 25.99),
  (1, 2, 1, 89.00, 89.00),
  (1, 3, 2, 9.50, 19.00),
  (2, 4, 1, 199.99, 199.99),
  (3, 5, 1, 12.49, 12.49),
  (3, 1, 1, 25.99, 25.99),
  (3, 3, 1, 9.50, 9.50),
  (4, 5, 1, 12.49, 12.49),
  (5, 6, 2, 18.75, 37.50),
  (6, 1, 1, 25.99, 25.99);

INSERT INTO payments (order_id, provider, amount, status, paid_at) VALUES
  (1, 'card', 115.48, 'completed', '2025-12-12 12:01:00'),
  (2, 'bank', 199.99, 'completed', '2025-12-22 09:31:00'),
  (3, 'paypal', 44.99, 'completed', '2026-01-08 18:11:00'),
  (4, 'card', 12.49, 'failed', NULL),
  (5, 'card', 37.50, 'completed', '2026-01-18 07:56:00'),
  (6, 'card', 25.99, 'refunded', '2026-01-19 15:41:00');

SET FOREIGN_KEY_CHECKS = 1;
