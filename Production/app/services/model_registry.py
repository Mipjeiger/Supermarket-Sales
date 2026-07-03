import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import mlflow
from mlflow.tracking import MlflowClient
from app.config.config import settings

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Service for managing models with MLflow"""

    _instance = None
    _models = {}
    _performance = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the model registry"""
        try:
            # Set MLFlow tracking URI
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

            # Set experiment
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            # Load all models from local models directory
            self._load_local_models()

            # Load performance data
            self._load_performance_data()

            logger.info("✅ ModelRegistry initialized successfully.")
            logger.info(f"📊 Loaded {len(self._models)} models")

        except Exception as e:
            logger.error(f"❌ Failed to initialize model registry: {str(e)}")
            self._fallback_load()

    def _load_local_models(self):
        """Load models from local .pkl files"""
        model_dir = Path(settings.MODEL_PATH)

        if not model_dir.exists():
            logger.warning(f"⚠️ Model directory does not exist: {model_dir}. Creating it.")
            return
        
        # Find all .pkl files
        pkl_files = list(model_dir.glob("*.pkl"))

        for pkl_file in pkl_files:
            try:
                model_name = pkl_file.stem.replace("model_", "")
                self._models[model_name] = joblib.load(pkl_file)
                logger.info(f"✅ Loaded model: {model_name} from {pkl_file}")
            
            except Exception as e:
                logger.error(f"❌ Failed to load model from {pkl_file}: {str(e)}")

    def _load_performance_data(self):
        """Load model performance data from CSV."""
        perf_file = Path(settings.MODEL_PATH) / "model_performance_comparison.csv"

        if perf_file.exists():
            try:
                df = pd.read_csv(perf_file)

                for _, row in df.iterrows():
                    model_name = row['Model']
                    self._performance[model_name] = {
                        'rmse': row.get('RMSE', 0),
                        'mae': row.get('MAE', 0),
                        'r2': row.get('R2', 0)
                    }
                logger.info(f"✅ Loaded performance data for {len(self._performance)} models.")
            
            except Exception as e:
                logger.error(f"❌ Failed to load performance data: {str(e)}")

    def _fallback_load(self):
        """Fallback mechanism to load models if MLflow fails."""
        logger.warning("⚠️ Falling back to local model loading.")
        self._load_local_models()
        self._load_performance_data()

    def get_model(self, model_name: str):
        """Get a model by name."""
        model = self._models.get(model_name)

        if not model:
            logger.warning(f"⚠️ Model '{model_name}' not found in registry.")
        return model
    
    def list_models(self) -> List[str]:
        """List all available models."""
        return list(self._models.keys())
    
    def get_model_performance(self, model_name: str) -> Optional[Dict]:
        """Get performance metrics for a specific model."""
        return self._performance.get(model_name)
    
    def register_model(self, model_name: str, model_path: Path, metrics: Dict):
        """Register a model with MLFlow"""
        try:
            with mlflow.start_run(run_name=f"register_{model_name}"):

                # Log model
                mlflow.sklearn.log_model(
                    joblib.load(model_path),
                    artifact_path=model_name
                )

                # Log metrics
                mlflow.log_metrics(metrics)
                logger.info(f"✅ Registered model '{model_name}' with MLflow and logged metrics.")

        except Exception as e:
            logger.error(f"❌ Failed to register model '{model_name}': {str(e)}")

    def get_best_model(self, metric: str = "rmse") -> Optional[str]:
        """Get the best performing model based on a specific metric."""

        if not self._performance:
            logger.warning("⚠️ No performance data available to determine the best model.")
            return None
        
        best_model = None
        best_score = float('inf') if metric == "rmse" else float('-inf')

        for model_name, perf in self._performance.items():
            score = perf.get(metric, 0)

            if metric == "rmse" and score < best_score:
                best_score = score
                best_model = model_name
            elif metric != "rmse" and score > best_score:
                best_score = score
                best_model = model_name

        return best_model
    
# Singleton instance
model_registry = ModelRegistry()