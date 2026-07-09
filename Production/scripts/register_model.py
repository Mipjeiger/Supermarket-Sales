import os
import joblib
import json
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
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Check for lowercase 'model identifier
        if 'model' in df.columns:
            for _, row in df.iterrows():
                model_name = str(row['model']).strip()
                
                # Extract all numeric attributes across varying columns dynamically
                metrics_dict = {k: v for k, v in row.items() if k != 'model' and pd.notna(v) and isinstance(v, (int, float))}
                perf_map[model_name] = metrics_dict    
        else:
            logger.warning(f"⚠️ Could not find a 'model' key column header in {csv_path.name}")
                
    except Exception as e:
        logger.error(f"❌ Error parsing performance metrics file: {str(e)}")

    return perf_map

def load_json_parameters(json_path: Path) -> dict:
    """Load JSON logs cleanly into structured parameter maps"""
    if not json_path.exists():
        logger.warning(f"⚠️ Parameters JSON file {json_path} does not exist.")
        return {}
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Scenario A: JSON is already a standard dictionary mapping model
        if isinstance(data, dict):
            return data
        
        # Scenario B: JSON is a list of dictionaries with 'model_name' keys
        if isinstance(data, list):
            logger.info(f"📋 Detecting JSON array structure in {json_path.name}. Normalizing...")
            normalized_map = {}

            for item in data:
                if isinstance(item, dict):

                    # Identify which key holds the name of the algorithm
                    model_key = None
                    for identifier in ['Model', 'model', 'model_name', 'name']:
                        if identifier in item:
                            model_key = str(item[identifier]).strip()
                            break
                    
                    if model_key:
                        if 'params' in item and isinstance(item['params'], dict):
                            normalized_map[model_key] = item['params']

                        elif 'parameters' in item and isinstance(item['parameters'], dict):
                            normalized_map[model_key] = item['parameters']

                        else:
                            # Extract everything except the identifier key itself
                            normalized_map[model_key] = {k: v for k, v in item.items() if k not in ['Model', 'model', 'model_name', 'name']}
            
            return normalized_map

    except Exception as e:
        logger.error(f"❌ Error loading JSON parameters from {json_path}: {str(e)}")
        return {}

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

        # Load core summary metrics
        sales_metrics = parse_metrics_file(SALES_DIR / "model_comparison_summary.csv")

        # Load sales parameters JSON from logs_summary
        sales_params_file = SALES_DIR / "logs_summary" / "sales_model_params.json"
        sales_params_master = load_json_parameters(sales_params_file)

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

                    # Extract hyperparameters from the JSON summary if available
                    params = sales_params_master.get(raw_name) or sales_params_master.get(raw_name.lower()) or {}

                    if params:
                        mlflow.log_params(params)
                        logger.info(f"✅ Logged parameters for {raw_name}: {params}")
                    else:
                        logger.warning(f"⚠️ No parameters found for {raw_name} in JSON summary.")
            
            except Exception as e:
                logger.error(f"❌ Error registering model {raw_name}: {str(e)}")

    # =========================================================================
    # PROCESS FRAUD CLASSIFICATION FAMILY
    # =========================================================================
    if FRAUD_DIR.exists():
        mlflow.set_experiment("supermarket_fraud_pipeline")
        fraud_metrics = parse_metrics_file(FRAUD_DIR / "model_comparison_results.csv")

        # Load Fraud Metrics & Parameters JSON from logs_summary
        fraud_metrics_file = FRAUD_DIR / "logs_summary" / "fraud_model_metrics.csv"
        fraud_params_file = FRAUD_DIR / "logs_summary" / "fraud_model_params.json"

        # Parse targeted logs summary files
        fraud_metrics = parse_metrics_file(fraud_metrics_file)
        fraud_params_master = load_json_parameters(fraud_params_file)

        # Fallback layer: if logs_summary metrics table is empty, fallback to the main model_comparison_results.csv for metrics
        if not fraud_metrics:
            logger.info("⚠️ No metrics found in logs_summary; falling back to model_comparison_results.csv for fraud metrics.")
            fraud_metrics = parse_metrics_file(FRAUD_DIR / "model_comparison_results.csv")

        # Crawl through binary pickle serialization files
        for model_file in FRAUD_DIR.glob("*.pkl"):
            raw_name = model_file.stem
            logger.info(f"Processing Fraud Model: {raw_name} from {model_file}")

            try:
                model_artifact = joblib.load(model_file)
                with mlflow.start_run(run_name=f"fraud_{raw_name.lower()}"):
                    mlflow.set_tag("pipeline_tier", "production")
                    mlflow.set_tag("model_domain", "fraud_classification")

                    mlflow.sklearn.log_model(model_artifact, artifact_path="model")

                    metrics = fraud_metrics.get(raw_name, {})
                    if metrics:
                        mlflow.log_metrics(metrics)
                        logger.info(f"✅ Logged metrics for {raw_name}: {metrics}")

                    params = fraud_params_master.get(raw_name) or fraud_params_master.get(raw_name.lower()) or {}
                    if params:
                        mlflow.log_params(params)
                        logger.info(f"✅ Logged parameters for {raw_name}: {params}")
                    else:
                        logger.warning(f"⚠️ No parameters found for {raw_name} in JSON summary.")

            except Exception as e:
                logger.error(f"❌ Error registering model {raw_name}: {str(e)}")

# Usage
if __name__ == "__main__":
    register_pipeline()
    logger.info("\n🚀 ML Pipeline execution finished! Navigate to the MLflow UI to inspect metrics.")