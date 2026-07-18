from typing import Dict
import joblib
import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Any
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
_label_maps: Optional[Dict[str, Dict[str, int]]] = None

def _get_encode_features() -> Optional[Path]:
    """Returns the encode features path used during training model."""
    sales_features_path = settings.SALES_FEATURES
    return sales_features_path if sales_features_path.exists() else None

def _build_encoding_schema() -> tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    Derives the full encoding schema directly from X_features.parquet.
    """
    encode_path = _get_encode_features()
    if encode_path is None:
        raise RuntimeError(f"❌ X_features.parquet not found at: {settings.SALES_FEATURES}")

    raw_path = settings.DATA_CLEANED.parent / "combined_sql_supermarket.parquet"
    if not raw_path.exists():
        raise RuntimeError(f"❌ Parquet data not found at: {raw_path}")

    X_ref = pd.read_parquet(encode_path)
    raw_df = pd.read_parquet(raw_path)

    feature_columns: List[str] = X_ref.columns.tolist()

    categorical_cols: List[str] = [
        col for col in feature_columns if col in raw_df.columns
        and X_ref[col].dtype == np.int64
        and raw_df[col].dtype == object
    ]

    label_maps: Dict[str, Dict[str, int]] = {
        col: {v: i for i, v in enumerate(sorted(raw_df[col].dropna().unique().tolist()))}
        for col in categorical_cols
    }

    logger.info(f"✅ Encoding schema built from X_features.parquet: {len(feature_columns)} features.")
    return feature_columns, categorical_cols, label_maps

def _get_encoded_features(payload: SalesRequest) -> pd.DataFrame:
    """
    Converts human-readable SalesRequest into an integer-encoded DataFrame.
    """
    global _feature_columns, _categorical_cols, _label_maps

    if _label_maps is None:
        _feature_columns, _categorical_cols, _label_maps = _build_encoding_schema()

    # ⚡ FIX: Kept OUTSIDE the if-block so it processes data on EVERY single request
    raw = {k.lower().replace("_", ""): v for k, v in payload.model_dump().items()}
    encoded: Dict[str, Any] = {}

    for col in _feature_columns:
        lookup_key = col.lower().replace("_", "")

        if lookup_key not in raw:
            logger.warning(f"⚠️ Missing feature '{col}' in payload, defaulting to 0")
            val = 0
        else:
            val = raw[lookup_key]

        if col in _categorical_cols:
            label_map = _label_maps[col]
            if str(val) not in label_map:
                encoded[col] = 0  # Fallback
            else:
                encoded[col] = label_map[str(val)]
        else:
            try:
                encoded[col] = float(val) if val is not None else 0.0
            except ValueError:
                logger.warning(f"⚠️ Non-numeric value for '{col}': {val}, defaulting to 0.0")
                encoded[col] = 0.0

    df = pd.DataFrame([encoded], columns=_feature_columns)
    df = df.fillna(0.0)
    return df

def _get_available_models() -> Dict[str, Path]:
    """Returns a dictionary of available model stems mapped to their full path."""
    available: List[Path] = settings.SALES_MODEL_PATH
    if not available:
        return {}
    return {p.stem: p for p in available}

def _get_best_model_name() -> str:
    """Returns the default fallback best performing model name."""
    list_models = [
        "CatboostRegressor_model",
        "XGBRegressor_model",
        "RandomForestRegressor_model",
        "DecisionTreeRegressor_model",
    ]
    stem_map = _get_available_models()
    if not stem_map:
        raise RuntimeError("No models found in SALES_MODEL_PATH.")

    for preferred in list_models:
        if preferred in stem_map:
            return preferred
    return list(stem_map.keys())[0]

def _get_scaler_path() -> Optional[Path]:
    """Returns the scaler path used during training model."""
    scaler_path = settings.MODEL_PATH / "sales_ml_models" / "scaler" / "scaler.joblib"
    return scaler_path if scaler_path.exists() else None

def _load_sales_components(selected_model_name: str):
    """Loads the specifically requested model + scaler into the registry."""
    stem_map = _get_available_models()
    
    if selected_model_name not in stem_map:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{selected_model_name}' not available. Choose from: {list(stem_map.keys())}"
        )

    # Lazy-load model into model_registry if not already initialized
    if model_registry.get_model(selected_model_name) is None:
        model_registry.load_model(domain="sales", model_name=selected_model_name, ext=".joblib")

    # Load shared scaler 
    scaler = None
    scaler_path = _get_scaler_path()
    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"❌ Error loading scaler: {str(e)}")

    model = model_registry.get_model(selected_model_name)
    if model is None:
        raise RuntimeError(f"❌ Model '{selected_model_name}' failed to load from registry.")

    return model, scaler

@router.post("/sales-prediction")
async def sales_prediction(
    payload: SalesRequest,
    model_name: Optional[str] = Query(
        None, 
        description="Specify which model to run. If omitted, defaults to the best available model."
    )
):
    """
    Sales prediction endpoint with dynamic model selection.
    """
    try:
        # 1. Fallback to default model strategy if parameter isn't provided
        target_model = model_name if model_name else _get_best_model_name()

        # 2. Encode payload (Runs securely on every hit)
        feature_vector: pd.DataFrame = _get_encoded_features(payload)

        # 3. Load runtime artifacts
        model, scaler = _load_sales_components(target_model)

        # 4. Input processing array alignment
        feature_array: np.ndarray
        if scaler is not None:
            raw_array = feature_vector.to_numpy().astype(np.float64)
            raw_array = np.nan_to_num(raw_array)
            feature_array = scaler.transform(raw_array.reshape(1, -1))
        else:
            feature_array = feature_vector.to_numpy().reshape(1, -1)

        # 5. Predict using ML model
        prediction = model.predict(feature_array)
        pred_value = float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else float(prediction)

        return {
            "model_used": target_model,
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