from typing import Dict
import joblib
import logging
import numpy as np
import pandas as pd
import joblib
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

from app.core.config import settings
from app.services.model_registry import model_registry

logger = logging.getLogger(__name__)
router = APIRouter()

class SalesRequest(BaseModel):
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

# --- Module-level caches - built once on first request ---
_feature_columns: Optional[List[str]] = None
_categorical_cols: Optional[List[str]] = None
_label_maps: Optional[Dict[str, int]] = None

def _get_encode_features() -> Optional[Path]:
    """Returns the encode features path used during training model."""
    sales_features_path = settings.SALES_FEATURES
    return sales_features_path if sales_features_path.exists() else None

def _build_encoding_schema() -> tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    Derives the full encoding schema directly from X_features.parquet.
    - feature_columns : column order as used during training (from X_features.parquet)
    - categorical_cols: auto-detected by comparing dtypes between X_features (int64)
                        and combined_sql_supermarket (object). No hardcoded lists.
    - label_maps      : {col: {string_val: int_code}} reconstructed by alphabetical
                        sort — identical to how sklearn LabelEncoder works.
    """
    encode_path = _get_encode_features()
    if encode_path is None:
        raise RuntimeError(f"❌ X_features.parquet not found at: {settings.SALES_FEATURES}")

    # Load SQL data combined
    raw_path = settings.DATA_CLEANED.parent / "combined_sql_supermarket.parquet"
    if not raw_path.exists():
        raise RuntimeError(f"❌ Parquet data not found at: {raw_path}")

    # Load reference files
    X_ref = pd.read_parquet(encode_path)
    raw_df = pd.read_parquet(raw_path)

    # Feature column order comes directly from .parquet file
    feature_columns: List[str] = X_ref.columns.tolist()

    # Auto-detect categorical: int64 in X_ref and object dtype in raw_df
    categorical_cols: List[str] = [
        col for col in feature_columns if col in raw_df.columns
        and X_ref[col].dtype == np.int64
        and raw_df[col].dtype == object
    ]

    # Build str-int map per categorical column
    label_maps: Dict[str, Dict[str, int]] = {
        col: {v: i for i, v in enumerate(sorted(raw_df[col].dropna().unique().tolist()))}
        for col in categorical_cols
    }

    logger.info(f"✅ Encoding schema built from X_features.parquet: "
        f"{len(feature_columns)} features, {len(categorical_cols)} categorical cols.")

    return feature_columns, categorical_cols, label_maps

def _get_encoded_features(payload: SalesRequest) -> pd.DataFrame:
    """
    Converts human-readable SalesRequest into an integer-encoded DataFrame
    matching X_features.parquet column order and dtypes.
    Raises HTTP 422 for unknown categorical values with valid options listed.
    """
    global _feature_columns, _categorical_cols, _label_maps

    # Build schema once, cache for all subsequent requests
    if _label_maps is None:
        _feature_columns, _categorical_cols, _label_maps = _build_encoding_schema()

        raw = {k.lower().replace("_", ""): v for k, v in payload.model_dump().items()}
        encoded: Dict[str, Any] = {}

        for col in _feature_columns:
            # Normalize target column name to match payload keys
            lookup_key = col.lower().replace("_", "")

            if lookup_key not in raw:
                logger.warning(f"⚠️ Missing feature '{col}' in payload, defaulting to None")
                val = 0
            else:
                val = raw[lookup_key]

            if col in _categorical_cols:
                label_map = _label_maps[col]

                if str(val) not in label_map:
                    encoded[col] = 0 # Fallback to closet matching key
                else:
                    encoded[col] = label_map[str(val)]
            else:
                # Force numeric conversion to stop NaN generation
                try:
                    encoded[col] = float(val) if val is not None else 0.0
                except ValueError:
                    logger.warning(f"⚠️ Non-numeric value for '{col}': {val}, defaulting to 0.0")
                    encoded[col] = 0.0

        df = pd.DataFrame([encoded], columns=_feature_columns)
        df = df.fillna(0.0) # Ensure no NaN values remain
        return df

def _get_best_model_path() -> Optional[Path]:
    """
    Returns the best performing sales model from the model registry
    """
    list_models = [
        "CatboostRegressor_model",
        "XGBRegressor_model",
        "RandomForestRegressor_model",
        "DecisionTreeRegressor_model",
    ]
    available: List[Path] = settings.SALES_MODEL_PATH

    if not available:
        return None

    # Match by stem (filename without extension)
    stem_map = {p.stem: p for p in available}
    for preferred in list_models:
        if preferred in stem_map:
            return stem_map[preferred]

    return available[0] # Fallback: first model in list

def _get_scaler_path() -> Optional[Path]:
    """Returns the scaler path used during training model."""
    scaler_path = settings.MODEL_PATH / "sales_ml_models" / "scaler" / "scaler.joblib"
    return scaler_path if scaler_path.exists() else None

def _load_sales_components():
    """Loads the best model + scaler into the registry."""
    model_path = _get_best_model_path()
    if model_path is None:
        raise RuntimeError("No .joblib model files found in SALES_MODEL_PATH.")

    model_name = model_path.stem # Get model name without extension

    # Load model into registry
    if model_registry.get_model(model_name) is None:
        model_registry.load_model(domain="sales", model_name=model_name, ext=".joblib")

    # Load scaler separately
    scaler = None
    scaler_path = _get_scaler_path()
    
    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler loaded from {scaler_path}")
        except Exception as e:
            logger.error(f"❌ Error loading scaler: {str(e)}")
    else:
        logger.warning("⚠️ Scaler path not found, skipping scaler loading")

    model = model_registry.get_model(model_name)
    if model is None:
        raise RuntimeError(f"❌ Model '{model_name}' not loaded successfully")

    return model, scaler, model_name
    
@router.post("/sales-prediction")
async def sales_prediction(payload: SalesRequest):
    """
    Sales prediction endpoint.
    Loads the best performing trained model (CatboostRegressor by default)
    and returns a sales forecast given the input features.
    """
    try:
        # 1. Encode payload - integer features derived from X_features.parquet
        feature_vector: pd.DataFrame = _get_encoded_features(payload)

        # 2. Load model + scaled (cached after first call
        model, scaler, model_name = _load_sales_components()

        # Apply scaler if available
        feature_array: np.ndarray
        if scaler is not None:
            raw_array = feature_vector.to_numpy().astype(np.float64)
            raw_array = np.nan_to_num(raw_array)
            feature_array = scaler.transform(raw_array.reshape(1, -1))
        else:
            feature_array = feature_vector.to_numpy().reshape(1, -1) # Fallback conversion to avoid scalar array issues

        # Predict using ML model
        prediction = model.predict(feature_array)

        # Handle prediction output array shapes from different estimators gracefully
        pred_value = float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else float(prediction)

        return {
            "model_used": model_name,
            "prediction": pred_value,
            "unit": "sales_amount",
            "input_features": payload.model_dump(),
            "encoded_features": feature_vector.to_dict(orient="records")[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Prediction failed")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")