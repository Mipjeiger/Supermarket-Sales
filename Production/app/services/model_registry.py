import joblib
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Centralized in-memory registry for managing and retrieving trained classical ML models."""

    def __init__(self):
        self.models = {}
        self.base_path = settings.MODEL_PATH

    def load_model(self, domain: str, model_name: str, ext: str = ".joblib"):
        """
        Dynamically builds the resolution path based on domain directories 
        (e.g., sales_ml_models or fraud_ml_models) and caches the loaded model.
        """
        path = self.base_path / f"{domain}_ml_models" / f"{model_name}{ext}"
        
        # Fallback check directly in base path if structural positioning varies
        if not path.exists():
            path = self.base_path / f"{model_name}{ext}"

        if path.exists():
            try:
                self.models[model_name] = joblib.load(path)
                logger.info(f"✅ Model '{model_name}' cached successfully from: {path}")
            except Exception as e:
                logger.error(f"❌ Critical failure loading binary model file '{model_name}': {str(e)}")
        else:
            logger.warning(f"⚠️ Model path resolution target missing: {path}")

    def get_model(self, model_name: str):
        """Retrieves a pre-loaded model instance from the shared global cache map."""
        return self.models.get(model_name)

# Thread-safe module singleton initialization
model_registry = ModelRegistry()