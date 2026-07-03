import logging
from app.config.config import settings

"""List all available ML models from directory"""

logger = logging.getLogger(__name__)

def load_models():
    """Load all ML models from the specified directory."""
    try:
        model_files = settings.MODEL_PATH_ML
        if not model_files:
            logger.warning("No model files found in the specified directory.")
            return []
        
        models = [str(model_file) for model_file in model_files]
        logger.info(f"Loaded {len(models)} models: {models}")
        return models
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return []