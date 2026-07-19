import logging
import asyncio
import json
import pandas as pd
from app.workers.stream_worker import run_transaction_consumer
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.endpoints import recommendation, security, sales_prediction
from app.services.model_registry import model_registry
from app.core.redis import redis_cache
from aiokafka import AIOKafkaConsumer
from app.services.behavior_analyst import behavior_analyst
from app.services.anomaly_agent import anomaly_agent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def consume_transaction_stream():
    """Asynchronous Kafka consumer for real-time transaction streaming."""
    consumer = AIOKafkaConsumer(
        "supermarket-transactions",
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="security-audit-group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )

    await consumer.start()
    logger.info("📡 Kafka consumer started")

    try:
        async for msg in consumer:
            transaction = msg.value
            logger.info(f"📥 Received kafka transaction: {transaction}")

            # Run predictive AI layer 
            is_fraud = transaction.get("sales", 0) > 4000000 # Mock indicator evaluation

            if is_fraud:
                logger.warning(f"⚠️ Fraud anomaly spotted on order {transaction.get('order_id')}!")

            # Trigger end-to-end Agentic LLM investigative layer for further analysis
            analysis_result = await behavior_analyst.analyze_transaction(transaction)

            # Dispatched deep context summary analysis
            await behavior_analyst.dispatch_analysis_summary(analysis_result)

    except Exception as e:
        logger.error(f"❌ Error in Kafka consumer: {str(e)}")
    finally:
        await consumer.stop()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown lifecycle events safely."""
    logger.info("🚀 Booting up Supermarket Compound ML-LLM Engine...")

    kafka_task = None

    try:
        # Load all serialization binaries using the centralized file registry service
        model_registry.load_model
        logger.info(
            "📊 All ML models (fraud + sales) cached successfully into worker memory."
        )

        # Fire up the background kafka consumer thread loop
        asyncio.create_task(run_transaction_consumer())
        logger.info(
            "📡 Kafka consumer thread loop initialized for real-time transaction streaming."
        )

        # Startup redis async connection pool for caching and pub/sub operations
        redis_cache.initialize()

        # Launch Kafka streaming processor
        kafka_task = asyncio.create_task(consume_transaction_stream())

    except Exception as e:
        logger.error(
            f"🚨 Critical failure during model registration startup sequence: {str(e)}"
        )

    yield

    # Close connection
    if kafka_task is not None:
        kafka_task.cancel()
    await redis_cache.close()

    logger.info("🛑 Shutting down Supermarket Compound ML-LLM Engine...")

# Initialize FastAPI with metadata parsed
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Production CORS Middleware Configuration
# Allows safe asymmetric communication between your web dashboards and backend nodes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bind modular structural routing namespaces
app.include_router(
    recommendation.router,
    prefix="/api/v1/recommendation",
    tags=["Contextual Recommendation"],
)

app.include_router(
    security.router,
    prefix="/api/v1/security",
    tags=["Real-time Streaming Security Operations"],
)

app.include_router(
    sales_prediction.router,
    prefix="/api/v1",
    tags=["Sales Prediction"]
)

@app.get("/")
def root_health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "sales_models": settings.SALES_MODEL_PATH,
        "fraud_models": settings.FRAUD_MODEL_PATH,
        "infrastructure": "Docker container mesh",
        "vector_store": "Qdrant Cloud Managed Cluster",
        "debug_mode": settings.DEBUG,
    }