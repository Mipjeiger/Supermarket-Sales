import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sqlalchemy import create_engine, text
from app.config.config import settings

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for making predictions and forecasts."""

    @staticmethod
    async def predict(model, features: dict) -> float:
        """Make a single prediction using the given model"""
        try:
            # Convert features to numpy array or dataframe
            X = PredictionService.__prepare__features(features)
            prediction = model.predict(X)[0]

            return float(prediction)

        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            raise

    @staticmethod
    async def generate_forecast(
        model,
        historical_data: pd.DataFrame,
        horizon: int = 24
    ) -> Tuple[List[float], List[datetime]]:
        """Generate a forecast for the next N hours or days or any specified horizon."""
        try:
            predictions = []
            timestamps = []

            # Use the last data point as starting point for forecasting
            last_data = historical_data.iloc[-1].copy()

            for i in range(horizon):
                # Prepare features for prediction
                features = PredictionService._prepare_forecast_features(last_data, i + 1)

                # Make prediction
                pred = model.predict(features)[0]
                predictions.append(float(pred))

                # Create timestamp for the prediction
                next_time = datetime.now() + timedelta(hours=i + 1)
                timestamps.append(next_time)

                # Update last_data for next iteration
                if i < horizon - 1:
                    last_data = last_data.iloc[-1].copy()
                    last_data['target'] = pred  # Update target for next prediction
                    last_data['timestamp'] = next_time

            return predictions, timestamps
        
        except Exception as e:
            logger.error(f"❌ Forecast generation failed: {str(e)}")
            raise

    @staticmethod
    async def get_historical_data(
        lookback_hours: int = 168,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get historical data from database"""
        try:
            engine = create_engine(settings.POSTGRES_URL)

            if end_date:
                query = f"""
                    SELECT * FROM engineering.supermarket
                    WHERE order_date <= '{end_date}'
                    ORDER BY order_date DESC
                    LIMIT {lookback_hours}
                """
            else:
                query = f"""
                    SELECT * FROM engineering.supermarket
                    ORDER BY order_date DESC
                    LIMIT {lookback_hours}
                """

            df = pd.read_sql(query, engine)
            return df
        
        except Exception as e:
            logger.error(f"❌ Failed to fetch historical data: {str(e)}")
            raise

    @staticmethod
    async def calculate_rmse(actual: List[float], predicted: List[float]) -> float:
        """Calculate RMSE between actual and predicted values."""
        try:
            return float(np.sqrt(mean_squared_error(actual, predicted)))
        
        except Exception as e:
            logger.error(f"❌ RMSE calculation failed: {str(e)}")
            return None
        
    @staticmethod
    async def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate various performance metrics."""
        try:
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
                "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            }

            return metrics
        
        except Exception as e:
            logger.error(f"❌ Metrics calculation failed: {str(e)}")
            return {}
        
    @staticmethod
    def __prepare__features(features: dict) -> np.ndarray:
        """Prepare features for prediction"""
        return np.array([list(features.values())])
    
    @staticmethod
    def _prepare_forecast_features(last_data: pd.DataFrame, hour_offset: int) -> np.ndarray:
        """Prepare features for forecasting"""

        features = last_data[['target']].values.flatten()
        return features.reshape(1, -1)