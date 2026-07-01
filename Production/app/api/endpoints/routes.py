from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, Dict

class PredictionRequest(BaseModel):
    """Prediction request model for the API endpoint."""
    model_name: str = Field(..., description="Model name to use for prediction")
    features: Dict = Field(..., description="Features for prediction")
    horizon: Optional[int] = Field(24, description="Prediction horizon in hours")

class PredictionResponse(BaseModel):
    """Prediction response model for the API endpoint."""
    prediction: float
    model_name: str
    timestamp: str
    confidence_interval: Optional[Dict] = None

class ForecastResponse(BaseModel):
    """Forecast response model for the API endpoint."""
    predictions: list[float]
    timestamps: list[str]
    model_name: str
    rmse: Optional[float] = None
    data_through: str