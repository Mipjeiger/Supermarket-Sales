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
            # Set MLFlow tracking URI & experiment name
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            # Map the exact active multi-directory structures
            self._load_local_models()
            self._load_performance_data()

            logger.info("✅ ModelRegistry initialized successfully.")
            logger.info(f"📊 Loaded {len(self._models)} models")

        except Exception as e:
            logger.error(f"❌ Failed to initialize model registry: {str(e)}")
            self._fallback_load()

    def _load_local_models(self):
        """Recursively collect model binaries across nested workspace variants (.joblib and .pkl)."""
        model_dir = Path(settings.MODEL_PATH)

        if not model_dir.exists():
            logger.warning(f"⚠️ Model directory does not exist: {model_dir}. Creating it.")
            return
        
        # Scan nested sales engines
        sales_path = model_dir / "sales_ml_models"
        if sales_path.exists():
            for file_path in sales_path.glob("*_model.joblib"):
                try:
                    name_key = file_path.stem.replace("_model", "")
                    self._models[name_key] = joblib.load(file_path)
                    logger.info(f"✅ Loaded Sales model: {name_key} from {file_path}")
                
                except Exception as e:
                    logger.error(f"❌ Failed to load Sales model from {file_path}: {str(e)}")

        # Scan nested fraud engines
        fraud_path = model_dir / "fraud_ml_models"
        if fraud_path.exists():
            for file_path in fraud_path.glob("*_model.joblib"):
                try:
                    name_key = file_path.stem.replace("_model", "")
                    self._models[name_key] = joblib.load(file_path)
                    logger.info(f"✅ Loaded Fraud model: {name_key} from {file_path}")
                
                except Exception as e:
                    logger.error(f"❌ Failed to load Fraud model from {file_path}: {str(e)}")

    def _load_performance_data(self):
        """Parse structured metrics from isolated domain summary tables."""
        perf_file = Path(settings.MODEL_PATH) / "model_performance_comparison.csv"

        # Load sales metrics matrix
        sales_metrics = perf_file / "sales_models" / "model_comparison_summary.csv"
        if sales_metrics.exists():
            try:
                df = pd.read_csv(sales_metrics)

                for _, row in df.iterrows():
                    self._performance[row['Model']] = {
                        'rmse': row.get('RMSE', 0),
                        'mae': row.get('MAE', 0),
                        'r2': row.get('R2', 0)
                    }
            
            except Exception as e:
                logger.error(f"❌ Failed to load Sales performance metrics: {str(e)}")

        # Load fraud metrics matrix
        fraud_metrics = perf_file / "fraud_models" / "model_comparison_results.csv"
        if fraud_metrics.exists():
            try:
                df = pd.read_csv(fraud_metrics)

                for _, row in df.iterrows():
                    self._performance[row['Model']] = {
                        'accuracy': row.get('Accuracy', 0),
                        'f1': row.get('F1_Score', row.get('F1', 0)),
                        'precision': row.get('Precision', 0)
                    }
            
            except Exception as e:
                logger.error(f"❌ Failed to load Fraud performance metrics: {str(e)}")

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
                mlflow.sklearn.log_model(joblib.load(model_path), artifact_path=model_name)

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
        best_score = float('inf') if metric in ["rmse", "mae"] else float('-inf')

        for model_name, perf in self._performance.items():
            score = perf.get(metric)
            if score is None:
                continue

            if metric in ["rmse", "mae"] and score < best_score:
                best_score = score
                best_model = model_name
            elif metric not in ["rmse", "mae"] and score > best_score:
                best_score = score
                best_model = model_name

        return best_model
    
# Singleton instance
model_registry = ModelRegistry()