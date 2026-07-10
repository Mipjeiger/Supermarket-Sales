from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ENV File Path
ENV_PATH = BASE_DIR / ".env"
print(f"ENV_PATH: {ENV_PATH}")

class Settings(BaseSettings):
    """Application configuration for loading environment variables with Pydantic validation."""

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Configuration
    PROJECT_NAME: str = "Supermarket ML Operations API"
    DEBUG: bool = False
    API_VERSION : str = "v1"
    PORT: int = 8000

    # HuggingFace API Configuration
    HUGGINGFACE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # File Paths Configuration
    MODEL_PATH: Path = BASE_DIR / "Models"
    MODEL_PATH_ML: list = list(MODEL_PATH.glob("*.pkl"))
    SALES_MODEL_PATH: list = list((MODEL_PATH_ML[0] / "sales_ml_models").glob("*.joblib")) if MODEL_PATH_ML else []
    FRAUD_MODEL_PATH: list = list((MODEL_PATH_ML[0] / "fraud_ml_models").glob("*.pkl")) if MODEL_PATH_ML else []
    MLFLOW_PATH: Path = BASE_DIR / "mlflow"
    DATA_RAW: Path = BASE_DIR / "app" / "data" / "raw" / "SuperStoreOrders - SuperStoreOrders.csv"
    DATA_CLEANED: Path = BASE_DIR / "app" / "data" / "cleaned" / "data_sales_cleaned.parquet"

    # PostgreSQL Database Configuration
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = f"http://mlflow:5002"
    MLFLOW_ARTIFACT_LOCATION: str = str(MLFLOW_PATH / "artifacts")
    MLFLOW_EXPERIMENT_NAME: str = "supermarket_sales"

    # Slack Configuration
    SLACK_WEBHOOK_URL: Optional[str] = None
    SLACK_CHANNEL: str = "supermarket_sales"

    # Qdrant Configuration
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None

    # Prometheus Configuration
    PROMETHEUS_PORT: int = 9090

    # Monitoring Configuration
    MONITORING_INTERVAL: int = 3600  # 1 hour
    ALERT_THRESHOLD_RMSE: float = 1.5
    ALERT_THRESHOLD_LATENCY: float = 1.0  # seconds

    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:29092"

    # Computed property for the PostgreSQL connection URL
    @property
    def POSTGRES_URL(self) -> str:
        """Synchronous connection string for PostgreSQL database especially in Docker"""
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
# Singleton
settings = Settings()

# Create MLflow directories if they don't exist
settings.MLFLOW_PATH.mkdir(exist_ok=True)
(settings.MLFLOW_PATH / "artifacts").mkdir(exist_ok=True)