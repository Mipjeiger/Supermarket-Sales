-- Active: 1782786629854@@localhost@5432@sales_market
CREATE SCHEMA engineering;

CREATE TABLE engineering.supermarket (
    order_id VARCHAR(50) PRIMARY KEY,
    order_date TIMESTAMP,
    ship_date TIMESTAMP,
    ship_mode VARCHAR(50),
    customer_name VARCHAR(50),
    segment VARCHAR(50),
    state VARCHAR(50),
    country VARCHAR(50),
    market VARCHAR(50),
    region VARCHAR(50),
    product_id VARCHAR(100),
    category VARCHAR(50),
    sub_category VARCHAR(50),
    sales DOUBLE PRECISION,
    quantity INTEGER,
    discount DOUBLE PRECISION,
    profit DOUBLE PRECISION,
    shipping_cost DOUBLE PRECISION,
    order_priority VARCHAR(50),
    year INTEGER,
    unite_price DOUBLE PRECISION,
    profit_margin DOUBLE PRECISION
);

SELECT * FROM engineering.supermarket;