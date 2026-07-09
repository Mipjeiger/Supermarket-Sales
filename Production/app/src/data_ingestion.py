import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from app.core.config import settings

# Setup clean production logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def ingest_cleaned_data_to_postgres():
    """Reads the cleaned parquet file and uploads it directly to PostgreSQL database."""
    
    parquet_path = settings.DATA_CLEANED
    
    if not parquet_path.exists():
        logger.error(f"❌ Cleaned Parquet file not found at: {parquet_path}")
        return

    logger.info(f"📂 Reading cleaned dataset from: {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)

    # Keep only the first occurrence of any order_id
    logger.info("🧹 Removing transaction line-item duplicates based on order_id...")
    df = df.drop_duplicates(subset=["order_id"], keep="first")

    logger.info("🔌 Connecting to PostgreSQL Engine...")
    engine = create_engine(settings.POSTGRES_URL)

    try:
        # Wrap execution inside a secure transaction block
        with engine.begin() as connection:
            
            # 1. Ensure the schema exists
            logger.info("🛠️ Verifying 'engineering' database schema exists...")
            connection.execute(text("CREATE SCHEMA IF NOT EXISTS engineering;"))
            
            # 2. Clear existing data safely before reloading (Full Refresh Mode)
            logger.info("🗑️ Wiping existing records from 'engineering.supermarket' for full refresh...")
            connection.execute(text("TRUNCATE TABLE engineering.supermarket RESTART IDENTITY;"))
            
            # 3. Stream data to your specific schema table
            logger.info(f"🚀 Ingesting {len(df)} records into 'engineering.supermarket'...")
            df.to_sql(
                name="supermarket",
                con=connection,
                schema="engineering",
                if_exists="append",
                index=False
            )
            
        logger.info("✅ Data ingestion completed successfully!")

    except Exception as e:
        logger.error(f"❌ An error occurred during data ingestion: {str(e)}")
        raise e

if __name__ == "__main__":
    ingest_cleaned_data_to_postgres()