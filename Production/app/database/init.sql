-- Active: 1782786629854@@localhost@5432@sales_market
CREATE SCHEMA engineering;

DROP TABLE IF EXISTS engineering.supermarket;
CREATE TABLE engineering.supermarket (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100),
    order_date TIMESTAMP,
    ship_date TIMESTAMP,
    ship_mode VARCHAR(100),
    customer_name VARCHAR(255),  -- Increased to allow longer full names
    segment VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    market VARCHAR(100),
    region VARCHAR(100),
    product_id VARCHAR(255),     -- Increased capacity boundary
    category VARCHAR(100),
    sub_category VARCHAR(100),
    product_name TEXT,           -- Changed to TEXT to accept infinite item descriptions 🚀
    sales DOUBLE PRECISION,
    quantity INTEGER,
    discount DOUBLE PRECISION,
    profit DOUBLE PRECISION,
    shipping_cost DOUBLE PRECISION,
    order_priority VARCHAR(100),
    year INTEGER,
    unit_price DOUBLE PRECISION,   
    profit_margin DOUBLE PRECISION
);

SELECT * FROM engineering.supermarket;