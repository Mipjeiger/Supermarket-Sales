import joblib
import logging
import time
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Any, Dict, Union
from pathlib import Path
from enum import Enum
from feast import FeatureStore
from app.core.config import settings
from app.monitoring.metrics import metrics_collector
from app.services.model_registry import model_registry

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Intialize Feast Feature Store ---
try:
    feast_store = FeatureStore(repo_path=settings.FEAST_REPO_PATH)
    logger.info("✅ Feast Feature Store initialized successfully.")
except Exception as e:
    feature_store = None
    logger.warning("⚠️ Failed to initialize Feast Feature Store. Ensure the repository path is correct. Error: %s", str(e))

class FraudRiskLevel(str, Enum):
    """Enumeration for fraud risk levels."""

    LOW = "Low Risk Level"
    MEDIUM = "Medium Risk Level"
    HIGH = "High Risk Level"

class FraudPredictionRequest(BaseModel):
    order_id: Optional[Union[int, str]] = Field(None, description="Order ID to retrieve pre-computed features from Feast store.")
    ship_mode: Optional[str] = None
    customer_name: Optional[str] = None
    segment: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    market: Optional[str] = None
    region: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    product_name: Optional[str] = None
    quantity: Optional[int] = None
    discount: Optional[float] = None
    profit: Optional[float] = None
    shipping_cost: Optional[float] = None
    order_priority: Optional[str] = None
    year: Optional[int] = None
    unit_price: Optional[float] = None
    profit_margin: Optional[float] = None
    sales: Optional[float] = None
    shipping_days: Optional[int] = None

def fetch_fraud_features_from_feast(order_id: Union[int, str]) -> Dict[str, Any]:
    """Retrieves online features for a given order_id from SQLite Feast store."""
    if feast_store is None:
        raise RuntimeError("Feast Feature Store is not initialized.")
    
    # Ensure order_id is integer-compatible for Feast lookup
    try:
        feast_entity_id = int(order_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid order_id '{order_id}'. Must be an integer or string representing an integer.",
        )
    
    feature_refs = [
        "fraud_features:ship_mode",
        "fraud_features:customer_name",
        "fraud_features:segment",
        "fraud_features:state",
        "fraud_features:country",
        "fraud_features:market",
        "fraud_features:region",
        "fraud_features:category",
        "fraud_features:sub_category",
        "fraud_features:product_name",
        "fraud_features:quantity",
        "fraud_features:discount",
        "fraud_features:profit",
        "fraud_features:shipping_cost",
        "fraud_features:order_priority",
        "fraud_features:year",
        "fraud_features:unit_price",
        "fraud_features:profit_margin",
        "fraud_features:sales",
        "fraud_features:shipping_days",
    ]

    response = feast_store.get_online_features(
        features=feature_refs,
        entity_rows=[{"order_id": feast_entity_id}],
    )
    df = response.to_df()

    if df.empty or df["fraud_features:sales"].isnull().all():
        raise HTTPException(
            status_code=404,
            detail=f"❌ Order ID '{order_id}' not found in Feast online feature store.",
        )
    
    # Strip the 'fraud_features:' prefix for payload building
    raw_dict = {}
    for col in df.columns:
        clean_col = col.replace("fraud_features:", "")
        raw_dict[clean_col] = df[col].iloc[0]  # Get the first row value

    return raw_dict

# --- Module-level caches - built once on first request ---
_feature_columns: Optional[List[str]] = None
_categorical_cols: Optional[List[str]] = None
_label_maps: Optional[Dict[str, Dict[str, int]]] = None

def get_encode_features() -> Optional[Path]:
    """Returns the encode features path used during training model."""
    fraud_features_path = settings.FRAUD_FEATURES_FEAST
    return fraud_features_path if fraud_features_path.exists() else None

def build_encoding_schema() -> tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """Derives the full encoding schema"""
    encode_path = get_encode_features()
    if encode_path is None:
       raise RuntimeError("Encode features path not found.")

    raw_path = settings.DATA_CLEANED.parent / "combined_sql_supermarket.parquet"
    if not raw_path.exists():
        raise RuntimeError("Raw data path not found.")

    X_reference = pd.read_parquet(encode_path)
    raw_df = pd.read_parquet(raw_path)
    feature_columns: List[str] = X_reference.columns.tolist()
    categorical_cols: List[str] = [
        col for col in feature_columns
        if col in raw_df.columns and X_reference[col].dtype == np.int64 and raw_df[col].dtype == object
    ]

    label_maps: Dict[str, Dict[str, int]] = {
        col: {v: i for i, v in enumerate(sorted(raw_df[col].dropna().unique().tolist()))}
        for col in categorical_cols}  # enumerate unique values for each categorical column

    return feature_columns, categorical_cols, label_maps

def get_encoded_features(payload_dict: Dict[str, Any]) -> pd.DataFrame:
    """Converts human-readable transaction data into encoded features for model prediction."""
    global _feature_columns, _categorical_cols, _label_maps

    if _label_maps is None:
        _feature_columns, _categorical_cols, _label_maps = build_encoding_schema()

    # Kept outside the if-block to ensure data processing on every single request
    raw = {k.lower().replace("_", ""): v for k, v in payload_dict.items()}
    encoded: Dict[str, Any] = {}

    for col in _feature_columns:
        lookup_col = col.lower().replace("_", "")
        val = raw.get(lookup_col, 0)  # Default to 0 if not found

        if col in _categorical_cols:
            label_map = _label_maps[col]
            encoded[col] = label_map.get(str(val), 0)  # Default to 0 if not found
        else:
            try:
                encoded[col] = float(val) if val is not None else 0.0
            except ValueError:
                logger.warning(f"⚠️ Non-numeric value '{val}' for column '{col}'. Defaulting to 0.0.")
                encoded[col] = 0.0

    # Return to Dataframe with a single row for prediction
    df = pd.DataFrame([encoded], columns=_feature_columns).fillna(0)
    return df


def get_available_models() -> Dict[str, Path]:
    """Returns a dictionary of available fraud detection model stem mapped to full path."""
    available_models: List[Path] = settings.FRAUD_MODEL_PATH
    if not available_models:
        return {}
    return {model.stem: model for model in available_models}

def resolve_model_name(input_name: str) -> str:
    """Loweercase model-name and simplify to stem for matching against available models."""
    stem_map = get_available_models()
    clean_input = (input_name.lower().strip().replace("classifier", "").replace("model", "").replace("_", ""))

    # Define dynamic mapping shorthand for your classification production stack
    aliases = {
        "gbc": "GradientBoostingClassifier",
        "gradientboosting": "GradientBoostingClassifier",
        "gradientboost": "GradientBoostingClassifier",
        "rfc": "RandomForestClassifier",
        "randomforest": "RandomForestClassifier",
        "xgb": "XGBClassifier",
        "xgboost": "XGBClassifier",
    }

    # Check for alias match first
    if clean_input in aliases and aliases[clean_input] in stem_map:
        return aliases[clean_input]
    
    for actual_stem in stem_map.keys(): # Fallback loose partial matching againts available model stems
        clean_actual = (actual_stem.lower().replace("classifier", "").replace("model", "").replace("_", ""))

        if clean_input == clean_actual or clean_input in clean_actual:
            return actual_stem

    raise HTTPException(status_code=400, detail=f"Model identifier '{input_name}' not found.")

def get_best_model_name() -> str:
    """Returns the default fallback best performing model name."""
    list_models = ["GradientBoostingClassifier", "RandomForestClassifier", "XGBClassifier"]
    stem_map = get_available_models()

    if not stem_map:
        raise RuntimeError("No available models found in the model registry.")

    for pref_model in list_models:
        if pref_model in stem_map:
            return pref_model
    return list(stem_map.keys())[0] if stem_map else ""

def calculate_classifier_confidence(
    model, feature_array, prediction_raw: float
) -> float:
    """
    Calculates an engineered variance confidence percentage for the classifier prediction outputs.
    Fallsback gracefully based on the model's available methods and prediction outputs.
    """
    try:
        if hasattr(model, "predict_proba") and len(model.classes_) == 2:
            proba = model.predict_proba(feature_array)
            variance = np.var(proba[:, 1])  # Variance of the positive class probability

            # Convert variance to a confidence percentage (0-100)
            confidence = 1.0 / (1.0 + variance)
            return float(np.clip(confidence * 100, 50.0, 98.5)) # Ensure within 0-98.5 range

        return float(np.clip(96.5 - abs(prediction_raw) * 0.001, 70.0, 98.5)) # Fallback confidence based on raw prediction

    except Exception:
        return 80.0  # Fallback to safe default confidence

def load_fraud_components(selected_model_name: str):
    """Loads the selected model and its associated components from the model registry."""
    actual_file_stem = resolve_model_name(selected_model_name)

    if model_registry.get_model(actual_file_stem) is None:
        model_registry.load_model(domain="fraud", model_name=actual_file_stem, ext=".pkl")

    # Ensure the model is loaded
    model = model_registry.get_model(actual_file_stem)
    if model is None:
        raise RuntimeError(f"❌ Failed to load model '{actual_file_stem}' from the registry.")

    return model, actual_file_stem

@router.post("/fraud-prediction")
async def fraud_prediction(
    payload: FraudPredictionRequest,
    model_name: Optional[str] = Query(None, description="Specify the model name to use for prediction.",),
):
    # Brdige prometheus metrics collection for this endpoint
    start_time = time.time()
    status = "success"
    model_used = model_name if model_name else get_best_model_name()
    resolved_stem = None

    """
    Endpoint to predict the fraud risk level of a transaction based on provided features."""
    try:
        # 1. Check if Feast store lookup is requested - if not, use the provided payload directly
        if payload.order_id is not None and payload.sales is None:
            logger.info(f"🔍 Fetching features from Feast store for order_id: {payload.order_id}")
            data_dict = fetch_fraud_features_from_feast(payload.order_id)
        else:
            logger.info("⚡ Using direct payload features for inference.")
            data_dict = payload.model_dump() 

        target_query = model_name if model_name else get_best_model_name()

        # 2. Encode payload (using the cached encoding schema)
        feature_vector: pd.DataFrame = get_encoded_features(data_dict)

        # 3. Load runtime artifacts for the selected model
        model, resolved_stem = load_fraud_components(target_query)

        # 4. Predict probability score directly (0.0 to 1.0)
        if hasattr(model, "predict_proba"):
            prediction_prob = float(model.predict_proba(feature_vector)[:, 1][0])  # Probability of the positive class
        else:
            prediction_prob = float(model.predict(feature_vector)[0]) # Fallback to direct prediction

        # 5. Calculate model output confidence percentage score
        confidence_percentage = calculate_classifier_confidence(model, feature_vector, prediction_prob)

        return {
            "order_id": payload.order_id,
            "model_user": resolved_stem,
            "prediction": round(prediction_prob, 2),
            "prediction_confidence_score": f"{round(confidence_percentage, 2)}%",
            "unit": "Fraud Legacy Risk Score",
            "risk_level": (
                FraudRiskLevel.HIGH
                if prediction_prob >= 0.5
                else (FraudRiskLevel.LOW if prediction_prob < 0.2 else FraudRiskLevel.MEDIUM)
            ),
        }

    except HTTPException:
        status = "error"
        raise
    except Exception as e:
        status = "error"
        logger.exception("❌ Error during fraud prediction: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during fraud prediction: {str(e)}",
        )

    finally:
        # Register metrics and pushes the metrics event to Prometheus
        duration = time.time() - start_time
        metrics_collector.track_prediction(
            latency=duration,
            model_name=resolved_stem if resolved_stem else model_used,
            status=status,
        )
