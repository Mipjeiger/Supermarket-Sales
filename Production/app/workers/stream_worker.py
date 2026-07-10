import asyncio
import json
import logging
import pandas as pd
from aiokafka import AIOKafkaConsumer
from app.services.anomaly_agent import anomaly_agent
from app.core.config import settings

logger = logging.getLogger(__name__)


async def run_transaction_consumer():
    """Subscribes to Kafka broker and continously feeds data to the ML engine."""
    loop = asyncio.get_running_loop()

    consumer = AIOKafkaConsumer(
        "supermarket-transactions",
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="security-audit-group",
        auto_offset_reset="latest",
    )

    await consumer.start()
    logger.info(
        "📡 Kafka Streaming Consumer actively listening to 'supermarket-transactions'..."
    )

    try:
        async for msg in consumer:
            try:
                # Unpack streaming bytes into clean structural dictionary layouts
                payload = json.loads(msg.value.decode("utf-8"))

                streaming_row = payload.get("streaming_db_row", {})
                risk_metrics = payload.get("risk_metrics", {"abuse_score": 0.0})

                if not streaming_row:
                    logger.warning(
                        "⚠️ Received empty streaming row. Skipping processing."
                    )
                    continue

                df_row = pd.DataFrame([streaming_row])

                # Execute async analytical evaluation inside the core security model agent
                brief = await anomaly_agent.evaluate_and_notify_stream(
                    streaming_db_row=df_row, risk_metrics=risk_metrics
                )
                logger.info(f"✅ Stream processed successfully. Summary: {brief}")

            except Exception as entry_err:
                logger.error(f"❌ Error processing streaming entry: {str(entry_err)}")

    finally:
        await consumer.stop()
        logger.info("🛑 Kafka Streaming Consumer has been stopped.")
