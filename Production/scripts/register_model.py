import joblib
import pandas as pd
import mlflow
from pathlib import Path
import logging

"""Register existing .pkl models with MLflow -> Run manually"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set paths
MODEL_PATH = Path("Models")
MLFLOW_PATH = Path("mlflow")

def register_models():
    """Register all .pkl models with MLflow."""

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"file://" + str(MLFLOW_PATH.absolute()))
    mlflow.set_experiment("supermarket_sales")

    # Load performance data
    perf_file = MODEL_PATH / "model_performance_comparison.csv"
    performance = {}

    if perf_file.exists():
        df = pd.read_csv(perf_file)

        for _, row in df.iterrows():
            performance[row['Model']] = {
                'rmse': row.get('RMSE', 0),
                'mae': row.get('MAE', 0),
                'r2': row.get('R2', 0)
            }

    # Register each model
    for pkl_file in MODEL_PATH.glob("model_*.pkl"):
        model_name = pkl_file.stem.replace("model_", "")
        logger.info(f"Registering model: {model_name}")

        try:
            model = joblib.load(pkl_file)

            with mlflow.start_run(run_name=f"register_{model_name}"):

                # Log model
                mlflow.sklearn.log_model(model, model_name)

                # Log performance metrics
                if model_name in performance:
                    metrics = performance[model_name]
                    mlflow.log_metrics(metrics)
                    logger.info(f"Logged metrics for {model_name}: {metrics}")
                
                else:
                    logger.warning(f"No performance metrics found for {model_name}. Skipping metric logging.")

        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {str(e)}")

    
    logger.info("✅ Model registration completed.")

# Usage
if __name__ == "__main__":
    MLFLOW_PATH.mkdir(exist_ok=True)
    (MLFLOW_PATH / "artifacts").mkdir(exist_ok=True)

    register_models()