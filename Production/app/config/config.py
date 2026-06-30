from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    PROJECT_NAME: str = "SuperMarket Sales Prediction"
    DEBUG: bool = False

    # HuggingFace API Configuration
    HUGGINGFACE_API_KEY: str
    GROQ_API_KEY: str

    # File Paths Configuration
    MODEL_PATH: Path = BASE_DIR / "Models"
    DATA_RAW: Path = BASE_DIR / "app" / "data" / "raw" / "SuperStoreOrders - SuperStoreOrders.csv"
    DATA_CLEANED: Path = BASE_DIR / "app" / "data" / "cleaned" / "data_sales_cleaned.parquet"

    # PostgreSQL Database Configuration
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    # Computed property for the PostgreSQL connection URL
    @property
    def POSTGRES_URL(self) -> str:
        """Synchronous connection string for PostgreSQL database especially in Docker"""
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
# Singleton
settings = Settings()