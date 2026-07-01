import time
import logging
from typing import Optional, Dict
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.model_registry import model_registry
from app.services.forecasting.prediction_service import PredictionService
from app.monitoring.metrics import MetricsCollector
from app.monitoring.slack_notifier import slack_notifier
from app.api.endpoints.routes import PredictionRequest, PredictionResponse, ForecastResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a sales prediction using the specified model."""
    start_time = time.time()

    try:
        # Get model from the registry
        model = model_registry.get_model(request.model_name)

        if not model:
            available_models = model_registry.list_models()
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found. Available models: {available_models}")
        
        # Make prediction
        prediction = await PredictionService.predict(model, request.features)

        # Calculate confidence interaval
        confidence_interval = {
            "lower": prediction * 0.9,
            "upper": prediction * 1.1
        }

        # Track metrics
        latency = time.time() - start_time
        MetricsCollector.track_prediction(latency, request.model_name)

        # Background task for latency alerts
        if latency > 1.0:  # Threshold for latency alert
            background_tasks.add_task(
                slack_notifier.send_message,
                f"⚠️ High latency detected: {latency:.2f}s for model {request.model_name}",
                color="#ffaa00"
            )
        
        return PredictionResponse(
            prediction=prediction,
            model_name=request.model_name,
            timestamp=datetime.now().isoformat(),
            confidence_interval=confidence_interval
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")

        await slack_notifier.send_error_alert(f"❌ Prediction failed for model {request.model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(model_name: str, horizon: int = 24, start_date: Optional[str] = None):
    """Generate a time series forecast for the next N hours."""
    try:
        # Get model
        model = model_registry.get_model(model_name)

        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
        
        # Get historical data from database
        historical_data = await PredictionService.get_historical_data(
            lookback_hours=168,  # Last 7 days
            end_date=start_date
        )

        # Generate forecast
        forecast, timestamps = await PredictionService.generate_forecast(
            model=model,
            historical_data=historical_data,
            horizon=horizon
        )

        # Calculate RMSE if actuals available
        rmse = None
        if len(historical_data) > horizon:
            rmse = await PredictionService.calculate_rmse(
                actual=historical_data[-horizon:],
                predicted=forecast[:horizon]
            )

        return ForecastResponse(
            predictions=forecast,
            timestamps=[t.isoformat() for t in timestamps],
            model_name=model_name,
            rmse=rmse,
            data_through=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/models")
async def list_models():
    """List all available models with their performance metrics."""
    try:
        models = model_registry.list_models()
        performance = {}

        for model in models:
            perf = model_registry.get_model_performance(model)

            if perf:
                performance[model] = perf
        
        return {
            "models": models,
            "performance": performance,
            "best_model": model_registry.get_best_model()
        }
    
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/models/{model_name}/performance")
async def get_model_performance(model_name: str):
    """Get performance metrics for a specific model."""
    try:
        performance = model_registry.get_model_performance(model_name)

        if not performance:
            raise HTTPException(status_code=404, detail=f"Performance metrics for model '{model_name}' not found.")
        
        return {
            "model_name": model_name,
            **performance
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get performance for model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))