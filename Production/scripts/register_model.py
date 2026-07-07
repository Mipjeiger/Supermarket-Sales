import os
import joblib
import pandas as pd
import mlflow
from pathlib import Path
import logging

"""Register existing .pkl models with MLflow -> Run manually"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set paths
BASE_MODEL_PATH = Path("Models")
SALES_DIR = BASE_MODEL_PATH / "sales_ml_models"
FRAUD_DIR = BASE_MODEL_PATH / "fraud_ml_models"

def parse_metrics_file(csv_path: Path) -> dict:
    """Parse to dynamically convert model comparison tables into lookup dictionaries."""
    perf_map = {}
    if not csv_path.exists():
        logger.warning(f"⚠️ Performance metrics file {csv_path} does not exist.")
        return perf_map
    
    try:
        df = pd.read_csv(csv_path)

        # Standardize string lookup variations
        if 'Model' in df.columns:
            for _, row in df.iterrows():
                model_name = str(row['Model']).strip()
                
                # Extract all numeric attributes across varying columns dynamically
                metrics_dict = {k: v for k, v in row.items() if k != 'Model' and pd.notna(v) and isinstance(v, (int, float))}
                perf_map[model_name] = metrics_dict    
    except Exception as e:
        logger.error(f"Error parsing performance metrics file: {str(e)}")

    return perf_map

def register_pipeline():
    """ML Pipeline executing tracking uploads into isolated MLflow Experiments."""

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow Tracking URI set to: {tracking_uri}")

    # ============================================================================
    # PROCESS SALES REGRESSION FAMILY
    # ============================================================================
    if SALES_DIR.exists():
        mlflow.set_experiment("supermarket_sales_pipeline")
        sales_metrics = parse_metrics_file(SALES_DIR / "model_comparison_summary.csv")

        # Crawl through joblib serialization files
        for model_file in SALES_DIR.glob("*_model.joblib"):
            # Normalize model key to look up performance metrics
            raw_name = model_file.stem.replace("_model", "")
            logger.info(f"Registering Sales Model: {raw_name} from {model_file}")

            try:
                model_artifact = joblib.load(model_file)
                with mlflow.start_run(run_name=f"sales_{raw_name.lower()}"):

                    # Explicity tag the domain scope
                    mlflow.set_tag("pipeline_tier", "production")
                    mlflow.set_tag("model_domain", "sales_regression")

                    # Log sklearn-compliant wrappers safely
                    mlflow.sklearn.log_model(model_artifact, artifact_path="model")

                    # Log associated metrics matrix matching the base string patterns
                    metrics = sales_metrics.get(raw_name, {})

                    if metrics:
                        mlflow.log_metrics(metrics)
                        logger.info(f"✅ Logged metrics for {raw_name}: {metrics}")
            
            except Exception as e:
                logger.error(f"❌ Error registering model {raw_name}: {str(e)}")

    # =========================================================================
    # PROCESS FRAUD CLASSIFICATION FAMILY
    # =========================================================================
    if FRAUD_DIR.exists():
        mlflow.set_experiment("supermarket_fraud_pipeline")
        fraud_metrics = parse_metrics_file(FRAUD_DIR / "model_comparison_results.csv")

        # Crawl through binary pickle serialization files
        for model_file in FRAUD_DIR.glob("*.pkl"):
            raw_name = model_file.stem
            logger.info(f"Processing Fraud Model: {raw_name} from {model_file}")

            try:
                model_artifact = joblib.load(model_file)