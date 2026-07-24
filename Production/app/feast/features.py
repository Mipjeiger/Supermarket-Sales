from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String
from pathlib import Path

# Define Path to the parquet files
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"

# 1. Define Entity (Primary Key)
customer = Entity(name="order_id", value_type=ValueType.INT64, join_keys=["order_id"])

# 2. Define Data Sources pointing to parquet files
fraud_source = FileSource(
    name="fraud_feature_source",
    path=str(BASE_DATA_DIR / "X_features_fraud_with_order_date.parquet"),
    timestamp_field="order_date",
)

sales_source = FileSource(
    name="sales_feature_source",
    path=str(BASE_DATA_DIR / "X_features_with_order_date.parquet"),
    timestamp_field="order_date",
)

# 3. Define Fraud Feature View
fraud_feature_view = FeatureView(
    name="fraud_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="ship_mode", dtype=Int64),
        Field(name="customer_name", dtype=Int64),
        Field(name="segment", dtype=Int64),
        Field(name="state", dtype=Int64),
        Field(name="country", dtype=Int64),
        Field(name="market", dtype=Int64),
        Field(name="region", dtype=Int64),
        Field(name="category", dtype=Int64),
        Field(name="sub_category", dtype=Int64),
        Field(name="product_name", dtype=Int64),
        Field(name="quantity", dtype=Int64),
        Field(name="discount", dtype=Float64),
        Field(name="profit", dtype=Float64),
        Field(name="shipping_cost", dtype=Float64),
        Field(name="order_priority", dtype=Int64),
        Field(name="year", dtype=Int64),
        Field(name="unit_price", dtype=Float64),
        Field(name="profit_margin", dtype=Float64),
        Field(name="sales", dtype=Float64),
        Field(name="shipping_days", dtype=Int64),
    ],
    online=True,
    source=fraud_source,
)

# 4. Define Sales Feature View
sales_feature_view = FeatureView(
    name="sales_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="ship_mode", dtype=Int64),
        Field(name="customer_name", dtype=Int64),
        Field(name="segment", dtype=Int64),
        Field(name="state", dtype=Int64),
        Field(name="country", dtype=Int64),
        Field(name="market", dtype=Int64),
        Field(name="region", dtype=Int64),
        Field(name="category", dtype=Int64),
        Field(name="sub_category", dtype=Int64),
        Field(name="product_name", dtype=Int64),
        Field(name="quantity", dtype=Int64),
        Field(name="discount", dtype=Float64),
        Field(name="profit", dtype=Float64),
        Field(name="shipping_cost", dtype=Float64),
        Field(name="order_priority", dtype=Int64),
        Field(name="year", dtype=Int64),
        Field(name="unit_price", dtype=Float64),
        Field(name="profit_margin", dtype=Float64),
    ],
    online=True,
    source=sales_source,
)
