import joblib
import logging
import numpy as np
import time
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
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

# --- Initialize Feast Feature Store ---
try:
    feast_store = FeatureStore(repo_path=str(settings.FEAST_REPO_PATH))
except Exception as e:
    logger.error(f"❌ Failed to initialize Feast Feature Store: {str(e)}")
    feast_store = None

# -- Decalartive Enums for UI selection toggles --
class PipelineMode(str, Enum):
    HISTORICAL = "Historical Verification"
    SIMULATION = "Inference Simulation"

class SalesRequest(BaseModel):
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

def fetch_sales_features(order_id: Union[int, str]) -> Dict[str, Any]:
    """Retrieves online features for a given order_id from Feast feature store."""
    if feast_store is None:
        raise RuntimeError("Feast Feature Store is not initialized.")
    
    # Ensure order_id is integer-compatible for Feast lookup
    try:
        feast_entity_id = int(order_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid order_id '{order_id}'. Must be an integer.")

    feature_refs = [
        "sales_features:ship_mode",
        "sales_features:customer_name",
        "sales_features:segment",
        "sales_features:state",
        "sales_features:country",
        "sales_features:market",
        "sales_features:region",
        "sales_features:category",
        "sales_features:sub_category",
        "sales_features:product_name",
        "sales_features:quantity",
        "sales_features:discount",
        "sales_features:profit",
        "sales_features:shipping_cost",
        "sales_features:order_priority",
        "sales_features:year",
        "sales_features:unit_price",
        "sales_features:profit_margin"
    ]

    response = feast_store.get_online_features(features=feature_refs, entity_rows=[{"order_id": feast_entity_id}])
    df = response.to_df()

    if df.empty or df["sales_features:ship_mode"].isnull().all():
        raise HTTPException(status_code=404, detail=f"No features found for order_id '{order_id}'.")
    
    # Strip the 'sales_features:' prefix for payload building
    raw_dict = {}
    for col in df.columns:
        clean_col = col.replace("sales_features:", "")
        raw_dict[clean_col] = df[col].iloc[0]  # Extract the first row value for each feature

    return raw_dict

# --- Module-level caches - built once on first request ---
_feature_columns: Optional[List[str]] = None
_categorical_cols: Optional[List[str]] = None
_label_maps: Optional[Dict[str, Dict[str, int]]] = None

def _get_encode_features() -> Optional[Path]:
    """Returns the encode features path used during training model."""
    sales_features_path = settings.SALES_FEATURES_FEAST
    return sales_features_path if sales_features_path.exists() else None


def _build_encoding_schema() -> tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    Derives the full encoding schema directly from X_features.parquet.
    """
    encode_path = _get_encode_features()
    if encode_path is None:
        raise RuntimeError(f"❌ X_features.parquet not found at: {settings.SALES_FEATURES_FEAST}")

    raw_path = settings.DATA_CLEANED.parent / "combined_sql_supermarket.parquet"
    if not raw_path.exists():
        raise RuntimeError(f"❌ Parquet data not found at: {raw_path}")

    X_ref = pd.read_parquet(encode_path)
    raw_df = pd.read_parquet(raw_path)

    feature_columns: List[str] = X_ref.columns.tolist()

    categorical_cols: List[str] = [
        col for col in feature_columns 
        if col in raw_df.columns and X_ref[col].dtype == np.int64 and raw_df[col].dtype == object
    ]

    label_maps: Dict[str, Dict[str, int]] = {
        col: {v: i for i, v in enumerate(sorted(raw_df[col].dropna().unique().tolist()))}
        for col in categorical_cols
    }

    logger.info(f"✅ Encoding schema built from X_features.parquet: {len(feature_columns)} features.")
    return feature_columns, categorical_cols, label_maps


def _get_encoded_features(payload_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts human-readable SalesRequest into an integer-encoded DataFrame.
    """
    global _feature_columns, _categorical_cols, _label_maps

    if _label_maps is None:
        _feature_columns, _categorical_cols, _label_maps = _build_encoding_schema()

    # FIX: Kept OUTSIDE the if-block so it processes data on EVERY single request
    raw = {k.lower().replace("_", ""): v for k, v in payload_dict.items()}
    encoded: Dict[str, Any] = {}

    for col in _feature_columns:
        lookup_key = col.lower().replace("_", "")

        val = raw.get(lookup_key, 0)  # Default to 0 if not provided

        if col in _categorical_cols:
            label_map = _label_maps[col]
            encoded[col] = label_map.get(str(val), 0)
        else:
            try:
                encoded[col] = float(val) if val is not None else 0.0
            except ValueError:
                logger.warning(f"⚠️ Non-numeric value for '{col}': {val}, defaulting to 0.0")
                encoded[col] = 0.0

    df = pd.DataFrame([encoded], columns=_feature_columns).fillna(0)
    return df

def _get_available_models() -> Dict[str, Path]:
    """Returns a dictionary of available model stems mapped to their full path."""
    available: List[Path] = settings.SALES_MODEL_PATH
    if not available:
        return {}
    return {p.stem: p for p in available}

def _resolve_model_name(input_name: str) -> str:
    """
    Maps a clean, user-friendly model name string to the actual file stem.
    Supports flexible aliases (case-insensitive, handles missing '_model' or 'regressor').
    """
    stem_map = _get_available_models()
    clean_input = (input_name.lower().strip().replace("regressor", "").replace("model", ""))

    # Define mapping dictionary for clean inputs -> actual stems
    """Models aliaes for query in API calls"""
    aliases = {
        "catboost": "CatboostRegressor_model",
        "xgboost": "XGBRegressor_model",
        "xgb": "XGBRegressor_model",
        "randomforest": "RandomForestRegressor_model",
        "rf": "RandomForestRegressor_model",
        "decisiontree": "DecisionTreeRegressor_model",
        "dt": "DecisionTreeRegressor_model",
    }

    if clean_input in aliases and aliases[clean_input] in stem_map:
        return aliases[clean_input]
    
    for actual_stem in stem_map.keys(): # Fallback loose partial matching againts available model stems
        clean_actual = (actual_stem.lower().replace("regressor", "").replace("model", "")).replace("_", "")
        if clean_input == clean_actual or clean_input in clean_actual:
            return actual_stem

    # If no match found, raise HTTP exception with available options
    readable_options = ["catboost", "xgboost", "randomforest", "decisiontree"]
    raise HTTPException(
        status_code=400,
        detail=f"Model identifier '{input_name}' not recognized. Please choose from: {readable_options}",
    )

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
    return list(stem_map.keys())[0] if stem_map else ""


def _get_scaler_path() -> Optional[Path]:
    """Returns the scaler path used during training model."""
    scaler_path = settings.MODEL_PATH / "sales_ml_models" / "scaler" / "scaler.joblib"
    return scaler_path if scaler_path.exists() else None

def _calculate_regression_confidence(
    model, feature_array, prediction_raw: float
) -> float:
    """
    Calculates an engineered variance confidence percentage for continuous regression outputs.
    Falls back gracefully based on tree variance or expected model boundaries.
    """
    try:
        if hasattr(model, "estimators_") and len(model.estimators_) > 0:
            preds = [est.predict(feature_array)[0] for est in model.estimators_]
            variance = np.var(preds)

            # Convert variance to a bounded percentage inverse ratio
            confidence = 1.0 / (1.0 + variance)
            return float(
                np.clip(confidence * 100, 55.0, 99.5)
            )  # Confidence between 55% and 99.5%

        # General model structural fallback using target bound
        return float(np.clip(95.2 - (abs(prediction_raw) * 0.001), 70.0, 98.5))

    except Exception:
        return 90.0  # Safe pipeline fallback confidence

def _load_sales_components(selected_model_name: str):
    """Loads the specifically requested model + scaler into the registry."""
    actual_file_stem = _resolve_model_name(selected_model_name)

    if model_registry.get_model(actual_file_stem) is None:
        model_registry.load_model(domain="sales", model_name=actual_file_stem, ext=".joblib")

    # Load shared scaler
    scaler = None
    scaler_path = _get_scaler_path()

    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"❌ Error loading scaler: {str(e)}")

    model = model_registry.get_model(actual_file_stem)
    if model is None:
        raise RuntimeError(f"❌ Model '{actual_file_stem}' failed to load from registry.")

    return model, scaler, actual_file_stem

@router.post("/sales-prediction")
async def sales_prediction(
    payload: SalesRequest,
    model_name: Optional[str] = Query(
        None,
        description="Specify which model to run. If omitted, defaults to the best available model.",
    ),
    pipeline_mode: PipelineMode = Query(
        PipelineMode.HISTORICAL,
        description="Select runtime engine behavior. 'Historical Verification' runs data as-is. 'Inference Simulation' activates scenario forecasting multipliers.",
    ),
    quantity_multiplier: float = Query(
        1,
        description="Simulate change in transaction sales volume (Only applies in Simulation mode).",
    ),
    discount_multiplier: float = Query(
        0.5,
        description="Simulate changes in promotional discounting activity (Only applies in Simulation mode).",
    ),
):
    # Bridge prometheus metrics collection with request lifecycle
    start_time = time.time()
    status = "success"
    model_used = model_name if model_name else _get_best_model_name()
    resolved_stem = None
    """
    Enhanced Sales Prediction API with explicit pipeline mode choices.
    """
    try:
        # 1. Check if Feast store lookup is requested - if not. use the provided payload directly
        if payload.order_id is not None and payload.sales is None:
            logger.info(f"🔍 Fetching features for order_id: {payload.order_id} from Feast store.")
            data_dict = fetch_sales_features(payload.order_id)
        else:
            logger.info("⚡ Using provided payload directly for prediction.")
            data_dict = payload.model_dump()
        
        # 2. Fallback to default model strategy if parameter isn't provided
        target_query = model_name if model_name else _get_best_model_name()

        # 3. Encode payload (Runs securely on every hit)
        feature_vector: pd.DataFrame = _get_encoded_features(data_dict)

        # Enchanced Policy: Evaluate explicit execution selection choices for pipeline mode
        q_mult = 1.0
        d_mult = 1.0

        if pipeline_mode == PipelineMode.SIMULATION:
            q_mult = quantity_multiplier
            d_mult = discount_multiplier

            # Apply feature-scpace transformations for simulation multipliers
            for col in feature_vector.columns:
                if "quantity" in col.lower():
                    feature_vector[col] = feature_vector[col] * q_mult
                if "discount" in col.lower():
                    feature_vector[col] = feature_vector[col] * d_mult
        else:
            logger.info(
                "⚡ Pipeline operating in Historical mode. Forecasting multipliers bypassed."
            )

        # 4. Load runtime artifacts
        model, scaler, resolved_stem = _load_sales_components(target_query)

        # 5. Pipeline Scale Normalization Realignment
        if scaler is not None:
            raw_array = np.nan_to_num(feature_vector.to_numpy().astype(np.float64)) 
            feature_array = scaler.transform(raw_array.reshape(1, -1))
        else:
            feature_array = feature_vector.to_numpy().reshape(1, -1)

        # 6. Predict using ML model & log scale
        prediction = model.predict(feature_array)
        pred_raw = (float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else float(prediction))

        # Apply exponential scale conversion inversion logic to restore currency metrics
        prediction_actual = float(np.expm1(pred_raw))
        if prediction_actual < 0:
            prediction_actual = 0.0  # Ensure no negative sales predictions

        # Calculate Model output confidence score
        confidence_percentage = _calculate_regression_confidence(model, feature_array, pred_raw)

        return {
            "model_used": resolved_stem,
            "prediction": round(prediction_actual, 2),
            "prediction_confidence_score": f"{round(confidence_percentage, 2)}%",
            "unit": "Rp",
            "applied_multipliers": {
                "quantity_multiplier": q_mult,
                "discount_multiplier": d_mult,
            },
        }

    except HTTPException:
        status = "error"
        raise
    except Exception as e:
        status = "error"
        logger.exception("❌ Prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )

    finally:
        # Register metrics for Prometheus monitoring
        duration = time.time() - start_time
        metrics_collector.track_prediction(
            latency=duration,
            model_name=resolved_stem if resolved_stem else model_used,
            status=status,
        )
