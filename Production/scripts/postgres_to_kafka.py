import psycopg2
import json
import time
from kafka import KafkaProducer
from app.core.config import settings
from sqlalchemy import create_engine, text

# Initialize core engines
producer = KafkaProducer(
    bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
    api_version=(3, 6, 0),
    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
)

conn = None
cursor = None

try:
    # Connect to PostgreSQL using the native connection string property
    print("📡 Connecting to PostgreSQL cluster...")
    conn = psycopg2.connect(settings.POSTGRES_RAW_URI)
    cursor = conn.cursor()

    # Execute pure SQL query
    print("🚀 Extracting rows from engineering.supermarket and streaming to Kafka...")
    query = """
        SELECT order_id, category, sub_category, sales, quantity, unit_price, profit_margin 
        FROM engineering.supermarket 
        LIMIT 80;
        """
    cursor.execute(query)

    # Dynamically extract column header mappings from cursor description
    columns = [desc[0] for desc in cursor.description]

    streamed_count = 0

    # Loop through matching database records
    for row in cursor.fetchall():
        # Zip column headers with tuple data to construct a structured dictionary
        payload = dict(zip(columns, row))
        producer.send("supermarket-transactions", value=payload)

        streamed_count += 1
        print(f"📤 Streamed row {streamed_count}: {payload}")

        # 1-second interval delay to simulate transactional stream timing velocity
        time.sleep(1)

    print(f"\n✅ Completed streaming all {streamed_count} rows to Kafka.")

except Exception as e:
    print(f"❌ Error during PostgreSQL to Kafka streaming: {str(e)}")

finally:
    # Cleanup Block: Ensure DB resources close gracefully
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    if producer:
        producer.flush()
        producer.close()

    print("🔒 Database connections and Kafka producer flushed and securely closed.")
