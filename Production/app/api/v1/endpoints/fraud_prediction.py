import joblib
import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Any, Dict
from pathlib import Path
from enum import Enum

from app.core.config import settings
from app.services.model_registry import model_registry

logger = logging.getLogger(__name__)
router = APIRouter()

class FraudRiskLevel(str, Enum):
    """Enumeration for fraud risk levels."""
    LOW = "Low Risk Level"
    MEDIUM = "Medium Risk Level"
    HIGH = "High Risk Level"

class FraudPredictionRequest(BaseModel):
    """Request model for fraud prediction based on transaction data."""
    ship_mode: str
    customer_name: str
    segment: str
    state: str
    country: str
    market: str
    region: str
    category: str
    sub_category: str
    product_name: str
    quantity: int
    discount: float
    profit: float
    shipping_cost: float
    order_priority: str
    year: int
    unit_price: float
    profit_margin: float
    sales: float
    shipping_days: int

# --- Module-level caches - built once on first request ---
_feature_columns: Optional[List[str]] = None
_categorical_cols: Optional[List[str]] = None
_label_maps: Optional[Dict[str, Dict[str, int]]] = None

def get_encode_features() -> Optional[Path]:
    """Returns the encode features path used during training model."""
    fraud_features_path = settings.FRAUD_FEATURES
    return fraud_features_path if fraud_features_path.exists() else None

def build_encoding_schema() -> tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """Derives the full encoding schema"""
    encode_path = get_encode_features()
    if encode_path is None:
        raise RuntimeError("Encode features path not found. Ensure the model is trained and features are saved.")
    
    raw_path = settings.DATA_CLEANED.parent / "combined_sql_supermarket.parquet"
    if not raw_path.exists():
        raise RuntimeError("Raw data path not found. Ensure the raw data is available for encoding schema derivation.")
    
    X_reference = pd.read_parquet(encode_path)
    raw_df = pd.read_parquet(raw_path)

    feature_columns: List[str] = X_reference.columns.tolist()
    
    categorical_cols: List[str] = [
        col for col in feature_columns if col in raw_df.columns
        and X_reference[col].dtype == np.int64
        and raw_df[col].dtype == object
    ]

    label_maps: Dict[str, Dict[str, int]] = {
        col: {v: i for i, v in enumerate(sorted(raw_df[col].dropna().unique().tolist()))}
        for col in categorical_cols
    } # enumerate unique values for each categorical column

    logger.info(f"✅ Encoding schema built with {len(feature_columns)} features, {len(categorical_cols)} categorical columns.")
    return feature_columns, categorical_cols, label_maps

def get_encoded_features(payload: FraudPredictionRequest) -> pd.DataFrame:
    """Converts human-readable transaction data into encoded features for model prediction."""
    global _feature_columns, _categorical_cols, _label_maps

    if _label_maps is None:
        _feature_columns, _categorical_cols, _label_maps = build_encoding_schema()

    # Kept outside the if-block to ensure data processing on every single request
    raw = {k.lower().replace("_",""): v for k, v in payload.model_dump().items()}
    encoded: Dict[str, Any] = {}

    for col in _feature_columns:
        lookup_col = col.lower().replace("_","")

        if lookup_col not in raw:
            logger.warning(f"⚠️ Column '{lookup_col}' not found in payload. Defaulting to 0 (safely).")
            val = 0
        else:
            val = raw[lookup_col]

        if col in _categorical_cols:
            label_map = _label_maps[col]
            if str[val] not in label_map:
                encoded[col] = 0 # Fallback to 0
            else:
                encoded[col] = label_map[str(val)]

        else:
            try:
                encoded[col] = float(val) if val is not None else 0.0
            except ValueError:
                logger.warning(f"⚠️ Non-numeric value '{val}' for column '{col}'. Defaulting to 0.0.")
                encoded[col] = 0.0

    # Return to Dataframe with a single row for prediction
    df = pd.DataFrame([encoded], columns=_feature_columns)
    df = df.fillna(0)  # Ensure no NaN values
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
    clean_input = input_name.lower().strip().replace("classifier", "").replace("model", "").replace("_", "")

    # Define dynamic mapping shorthand for your classification production stack
    aliases = {
        "gbc": "GradientBoostingClassifier",
        "gradientboosting": "GradientBoostingClassifier",
        "gradientboost": "GradientBoostingClassifier",
        "rfc": "RandomForestClassifier",
        "randomforest": "RandomForestClassifier",
        "xgb": "XGBClassifier",
        "xgboost": "XGBClassifier"
    }

    # Check for alias match first
    if clean_input in aliases:
        target_stem = aliases[clean_input]
        if target_stem in stem_map:
            return target_stem
        
    # Fallback loose partial matching againts available model stems
    for actual_stem in stem_map.keys():
        clean_actual = actual_stem.lower().replace("classifier", "").replace("model", "").replace("_", "")

        if clean_input == clean_actual or clean_input in clean_actual:
            return actual_stem
        
    # If no match found, raise HTTP exception with available options
    available_options = ["gbc", "randomforest", "xgboost"]
    raise HTTPException(
        status_code=400,
        detail=f"Model identifier '{input_name}' not found. Available options: {available_options}"
    )

def get_best_model_name() -> str:
    """Returns the default fallback best performing model name."""
    list_models = [
        "GradientBoostingClassifier",
        "RandomForestClassifier",
        "XGBClassifier"
    ]

    stem_map = get_available_models()
    if not stem_map:
        raise RuntimeError("No available models found in the model registry.")
    
    for pref_model in list_models:
        if pref_model in stem_map:
            return pref_model
        
    return list(stem_map.keys())[0] if stem_map else ""

def calculate_classifier_confidence(model, feature_array, prediction_raw: float) -> float:
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
            return float(np.clip(confidence * 100, 50.0, 98.5))  # Ensure within 0-98.5 range
        
        return float(np.clip(96.5 - abs(prediction_raw) * 0.001, 70.0, 98.5))  # Fallback confidence based on raw prediction
    
    except Exception:
        return 85.0  # Fallback to safe default confidence
    
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
    model_name: Optional[str] = Query(
        None,
        description="Specify the model name to use for prediction. If not provided, the best available model will be used."
    )
):
    """
    Endpoint to predict the fraud risk level of a transaction based on provided features."""
    try:
        # 1. Fallback to default model strategy if parameter is not provided
        target_query = model_name if model_name else get_best_model_name()

        # 2. Encode payload (using the cached encoding schema)
        feature_vector: pd.DataFrame = get_encoded_features(payload)

        # 3. Load runtime artifacts for the selected model
        model, resolved_stem = load_fraud_components(target_query)

        # 4. Predict fraud risk using the ML model
        prediction = model.predict(feature_vector)
        prediction_raw = model.predict_proba(feature_vector)[:, 1][0] if hasattr(model, "predict_proba") else prediction[0]

        # Apply exponential scale conversion inversion logic to restore currency metrics if available in the payload
        prediction_actual = float(np.expm1(prediction_raw))
        if prediction_actual < 0 or prediction_actual is None:
            prediction_actual = float(prediction_raw)  # Fallback to raw prediction if expm1 fails

        # Calculate model output confidence percentage score
        confidence_percentage = calculate_classifier_confidence(model, feature_vector, prediction_raw)

        return {
            "model_user": resolved_stem,
            "prediction": round(prediction_actual, 2),
            "prediction_confidence_score": f"{round(confidence_percentage, 2)}%",
            "unit": "Fraud Legcay Risk Score",
            "risk_level": FraudRiskLevel.HIGH if prediction_actual > 0.5 else FraudRiskLevel.LOW if prediction_actual < 0.3 else FraudRiskLevel.MEDIUM
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Error during fraud prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error during fraud prediction: {str(e)}")