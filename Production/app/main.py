import logging
import asyncio
import json
import time
import pandas as pd
from app.workers.stream_worker import run_transaction_consumer
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.endpoints import (
    recommendation,
    security,
    sales_prediction,
    fraud_prediction,
)
from app.services.model_registry import model_registry
from app.core.redis import redis_cache
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from app.services.behavior_analyst import behavior_analyst
from app.services.anomaly_agent import anomaly_agent
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from app.monitoring.metrics import metrics_collector
from app.workers.stream_worker import run_transaction_consumer
from app.api.v1.endpoints.llm import router as llm_router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown lifecycle events safely."""
    logger.info("🚀 Supermarket API Machine Learning Engine system is started...")

    kafka_task = None
    try:
        # Load all serialization binaries using the centralized file registry service
        model_registry.load_model
        logger.info(
            "📊 All ML models (fraud + sales) cached successfully into worker memory."
        )

        # Initialize metrics structure for prometheus
        metrics_collector.initialize_startup_metrics()

        # Startup redis async connection pool for caching and pub/sub operations
        redis_cache.initialize()

        # Launch Kafka streaming processor
        kafka_task = asyncio.create_task(run_transaction_consumer())

    except Exception as e:
        logger.error(
            f"🚨 Critical failure during model registration startup sequence: {str(e)}"
        )

    yield

    # Shutdown sequence
    logger.info("🛑 Shutting down Supermarket Compound API ML-LLM Engine...")
    if kafka_task is not None:
        kafka_task.cancel()
        try:
            await kafka_task
        except asyncio.CancelledError:
            logger.info("✅ Kafka consumer task cancelled cleanly.")

    await redis_cache.close()


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

# Register LLM Intelligence Gateway router for investigative insights and anomaly detection
app.include_router(llm_router, prefix="/api/v1", tags=["LLM Intelligence Gateway"])

# Sales Prediction API with explicit pipeline mode choices and simulation multipliers
app.include_router(sales_prediction.router, prefix="/api/v1", tags=["Sales Prediction"])

# Fraud Detection API with real-time anomaly detection and LLM investigative analysis
app.include_router(fraud_prediction.router, prefix="/api/v1", tags=["Fraud Detection"])

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


@app.get("/metrics", tags=["Telemetry"])
def get_metrics():
    """Exposes the raw prometheus text payload directly to the scraper"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
