import logging
import asyncio
from app.workers.stream_worker import run_transaction_consumer
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.endpoints import recommendation, security
from app.services.model_registry import model_registry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown lifecycle events safely."""
    logger.info("🚀 Booting up Supermarket Compound ML-LLM Engine...")

    try:
        # Load serialization binaries using the centralized file registry service
        model_registry.load_model("sales", "CatboostRegressor_model", ".joblib")
        model_registry.load_model("fraud", "XGBClassifier", ".pkl")
        logger.info(
            "📊 Upstream CatBoost and XGBoost models cached successfully into worker memory."
        )

        # Fire up the background kafka consumer thread loop
        asyncio.create_task(run_transaction_consumer())
        logger.info(
            "📡 Kafka consumer thread loop initialized for real-time transaction streaming."
        )
    except Exception as e:
        logger.error(
            f"🚨 Critical failure during model registration startup sequence: {str(e)}"
        )

    yield

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


@app.get("/")
def root_health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "infrastructure": "Docker container mesh",
        "vector_store": "Qdrant Cloud Managed Cluster",
        "debug_mode": settings.DEBUG,
    }
