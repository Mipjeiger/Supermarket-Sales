import psycopg2
import json
import time
from kafka import KafkaProducer
from app.core.config import settings
from sqlalchemy import create_engine, text

# Initialize core engines
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
)

# Create a SQLAIchemy engine using settings framework
engine = create_engine(settings.POSTGRES_URL)
print("🚀 Extracting rows from engineering.supermarket and streaming to Kafka...")

# Open connection to PostgreSQL and stream rows to Kafka
with engine.connect() as conn:
    query = text("SELECT order_id, category, sub_category, sales, quantity, unit_price, profit_margin FROM engineering.supermarket LIMIT 20;")
    result = conn.execute(query)

    # Extract keys from the cursor mapping
    columns = result.keys()

    for row in result.fetchall():
        payload = dict(zip(columns, row))

        # Send directly to active consumer topic setup
        producer.send("supermarket-transactions", value=payload)
        print(f"📡 Streamed row to Kafka: {payload}")
        time.sleep(1)

print("✅ Completed streaming all rows to Kafka. Closing producer connection.")