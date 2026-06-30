from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ENV File Path
ENV_PATH = BASE_DIR / ".env"
print(f"ENV_PATH: {ENV_PATH}")

class Settings(BaseSettings):
    """Application configuration for loading environment variables with Pydantic validation."""

    # API Configuration
    PROJECT_NAME: str = "SuperMarket Sales Prediction"
    DEBUG: bool = False

    # HuggingFace API Configuration
    HUGGINGFACE_API_KEY: str
    GROQ_API_KEY: str

    # Model Path Configuration
    MODEL_PATH: str = BASE_DIR / "Models"

    # Data Ingestion Configuration
    DATA_RAW: str = BASE_DIR / "data" / "raw"
    DATA_CLEANED: str = BASE_DIR /  "data" / "cleaned"

    # PostgreSQL Database Configuration
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    # Computed property for the PostgreSQL connection URL
    @property
    def POSTGRES_URL(self) -> str:
        """Synchronus connection string for PostgreSQL database especially in Docker"""
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
# Singleton
settings = Settings()