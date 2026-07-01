from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FraudRiskStackingEnsemble:
    """Stacking ensemble for fraud risk prediction combining multiple ML models"""

    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'xgboost': XGBRegressor(n_estimators=200, random_state=42, max_depth=10, learning_rate=0.05),
            'decision_tree': DecisionTreeRegressor(max_depth=10, random_state=42, min_samples_split=5, min_samples_leaf=2)
        }
        self.stacking_model = None
        self.meta_model = None
        self.is_trained = False

    def _create_stacking_model(self):
        """Create stacking ensemble with meta-model."""
        # Define base models
        base_models = [
            ('linear', self.models['linear']),
            ('random_forest', self.models['random_forest']),
            ('xgboost', self.models['xgboost']),
            ('decision_tree', self.models['decision_tree'])
        ]
        
        # Meta-model (learns best combination)
        meta_model = LinearRegression()
        
        # Create stacking ensemble
        self.stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            stack_method='predict'
        )
        
        return self.stacking_model
    
    def predict(self, X):
        """
        Make predictions using the stacking ensemble.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.stacking_model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities for risk scoring.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.predict(X)

        # Convert to probabilities using sigmoid (simplified)
        proba = 1 / (1 + np.exp(-predictions))
        return proba
    
    def explain_predictions(self, X, feature_names):
        """
        Explain predictions using SHAP or feature importance.
        """
        # Simplified explanation - in production use SHAP
        importance = self._calculate_feature_importance()
        return {
            'feature_importance': dict(zip(feature_names, importance)),
            'top_features': sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from base models."""
        importances = []
        
        if hasattr(self.stacking_model, 'estimators_'):
            for name, model in self.stacking_model.estimators_:
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
                elif hasattr(model, 'coef_'):
                    importances.append(np.abs(model.coef_))
        
        if importances:
            return np.mean(importances, axis=0)
        else:
            return np.ones(10)  # Placeholder
    
    def save(self, path):
        """Save the trained model."""
        joblib.dump(self.stacking_model, path)
        logger.info(f"💾 Model saved to {path}")
    
    def load(self, path):
        """Load a trained model."""
        self.stacking_model = joblib.load(path)
        self.is_trained = True
        logger.info(f"📂 Model loaded from {path}")
        return self.stacking_model