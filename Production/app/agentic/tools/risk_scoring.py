import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import joblib
from pathlib import Path

from app.config.config import settings

logger = logging.getLogger(__name__)

"""
Risk Scoring Tool for Agentic System
Calculates risk scores using ML models and rules.
"""

class RiskScoringTool:
    """
    Tool for calculating risk scores:
    - ML-based risk scoring
    - Rule-based risk assessment
    - Composite risk calculation
    """
    
    def __init__(self):
        self.models = {}
        self.rules = self._load_rules()
        self.weights = {
            'transaction': 0.3,
            'user': 0.25,
            'behavior': 0.25,
            'context': 0.2
        }
        self._load_models()

    def _load_models(self):
        """Load risk scoring models"""
        try:
            model_path = Path(settings.MODEL_PATH)

            # Try loading .pkl models
            for model_file in model_path.glob("model_*.pkl"):
                model_name = model_file.stem.replace("model_", "")
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded model: {model_name} from {model_file}")

            if not self.models:
                logger.warning("⚠️ No models found in the specified path.")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def _load_rules(self) -> List[Dict]:
        """Load risk scoring rules."""
        return [
            {
                'id': 'R001',
                'name': 'High Transaction Amount',
                'condition': lambda x: x.get('sales', 0) > 1000,
                'weight': 0.2,
                'description': 'Transaction amount exceeds threshold'
            },
            {
                'id': 'R002',
                'name': 'Unusual Time',
                'condition': lambda x: self._is_unusual_time(x.get('order_date')),
                'weight': 0.15,
                'description': 'Transaction at unusual time'
            },
            {
                'id': 'R003',
                'name': 'High Discount',
                'condition': lambda x: x.get('discount', 0) > 0.3,
                'weight': 0.1,
                'description': 'Discount rate exceeds threshold'
            },
            {
                'id': 'R004',
                'name': 'Unusual Category',
                'condition': lambda x: self._is_unusual_category(x.get('category')),
                'weight': 0.15,
                'description': 'Unusual product category for user'
            },
            {
                'id': 'R005',
                'name': 'Frequent Transactions',
                'condition': lambda x: x.get('frequency', 0) > 10,
                'weight': 0.1,
                'description': 'High transaction frequency'
            },
            {
                'id': 'R006',
                'name': 'Negative Profit',
                'condition': lambda x: x.get('profit', 0) < 0,
                'weight': 0.1,
                'description': 'Transaction with negative profit'
            },
            {
                'id': 'R007',
                'name': 'New User',
                'condition': lambda x: x.get('user_tenure', 365) < 7,
                'weight': 0.1,
                'description': 'New user with less than 7 days tenure'
            },
            {
                'id': 'R008',
                'name': 'Express Shipping',
                'condition': lambda x: x.get('ship_mode') == 'Express',
                'weight': 0.05,
                'description': 'Express shipping mode'
            }
        ]
    
    def run(self, data: str) -> Dict:
        """
        Main entry point for the tool.
        
        Args:
            data: JSON string or dict with transaction data
            
        Returns:
            Dict containing risk scores
        """
        try:
            # Parse data if string
            if isinstance(data, str):
                data = json.loads(data)
            
            # Calculate risk scores
            ml_score = self._calculate_ml_score(data)
            rule_score = self._calculate_rule_score(data)
            
            # Combine scores
            composite_score = self._calculate_composite_score(
                ml_score,
                rule_score,
                data
            )
            
            # Determine risk level
            risk_level = self._get_risk_level(composite_score)
            
            # Generate risk factors
            risk_factors = self._identify_risk_factors(data, ml_score, rule_score)
            
            return {
                'risk_score': composite_score,
                'risk_level': risk_level,
                'ml_score': ml_score,
                'rule_score': rule_score,
                'risk_factors': risk_factors,
                'recommendations': self._get_recommendations(risk_level, risk_factors),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Risk scoring failed: {str(e)}")
            return {'error': str(e)}
        
    def _calculate_ml_score(self, data: Dict) -> float:
        """Calculate risk score using ML models."""
        try:
            if not self.models:
                return self._fallback_ml_score(data)
            
            # Prepare features for ML model
            features = self._prepare_features(data)

            # Get predictions from all models
            scores = []
            for model_name, model in self.models.items():
                try:
                    pred = model.predict([features])[0]
                    scores.append(pred)
                
                except Exception as e:
                    logger.warning(f"⚠️ Model {model_name} prediction failed: {e}")

                    continue

            if scores:
                return float(np.mean(scores))
            else:
                return self._fallback_ml_score(data)
            
        except Exception as e:
            logger.error(f"❌ ML score calculation failed: {e}")
            return 0.5 # Neutral score if ML fails
        
    def _fallback_ml_score(self, data: Dict) -> float:
        """Fallback ML scoring when models are not available or fail."""
        score = 0.3

        # Increase score based on risk factors
        if data.get('sales', 0) > 1000:
            score += 0.2
        if data.get('discount', 0) > 0.3:
            score += 0.15
        if data.get('profit', 0) < 0:
            score += 0.15
        if self._is_unusual_time(data.get('order_date')):
            score += 0.1

        return min(score, 1.0)
    
    def _calculate_rule_score(self, data: Dict) -> float:
        """Calculate risk score using rules."""
        score = 0.0
        triggered_rules = []

        for rule in self.rules:
            try:
                if rule['condition'](data):
                    score += rule['weight']
                    triggered_rules.append(rule['id'])
                
            except Exception as e:
                logger.warning(f"⚠️ Rule {rule['id']} evaluation failed: {e}")
                continue

        return min(score, 1.0)
    
    def _calculate_composite_score(self, ml_score: float, rule_score: float, data: Dict) -> float:
        """Calculate composite risk score."""
        # Dynamic weighting based on data availability
        ml_weight = 0.6 if self.models else 0.0
        rule_weight = 0.4 if ml_weight > 0 else 0.8

        # Adjust weights if ML score is available
        if ml_score > 0:
            composite = (ml_score * ml_weight) + (rule_score * rule_weight)
        
        else:
            composite = rule_score

        return round(min(composite, 1.0), 3)
    
    def _prepare_features(self, data: Dict) -> np.ndarray:
        """Prepare features for ML models"""
        # Extract features for ML models
        features = [
            data.get('sales', 0),
            data.get('quantity', 0),
            data.get('discount', 0),
            data.get('profit', 0),
            data.get('order_frequency', 0),
            data.get('avg_transaction_value', 0),
            data.get('user_tenure', 365),
            data.get('seasonal_index', 0),
            data.get('category_risk', 0),
            data.get('payment_method_risk', 0)
        ]

        return np.array(features)
    
    def _is_unusual_time(self, order_date: Any) -> bool:
        """Check if transaction is at an unusual time."""
        if not order_date:
            return False
        
        try:
            if isinstance(order_date, str):
                order_date = pd.to_datetime(order_date)
            
            hour = order_date.hour
            # Unusual hours: 11 PM - 5 AM
            return hour >= 23 or hour <= 5
        except:
            return False
        
    def _is_unusual_category(self, category: str) -> bool:
        """Check if category is unusual for the user."""
        unusual_categories = ['Electronics', 'Jewelry', 'Luxury'] # Need to checkup based on database
        return category in unusual_categories
    
    def _get_risk_level(self, score: float) -> str:
        """
        Determine risk level based on score.
        """
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.3:
            return 'medium'
        else:
            return 'low'
        
    def _identify_risk_factors(self, data: Dict, ml_score: float, rule_score: float) -> List[Dict]:
        """
        Identify specific risk factors based on analysis. 
        """
        risk_factors = [] 
        
        # Check condition on analysis factors
        if data.get('sales', 0) > 2400000:
            risk_factors.append({
                'factor': 'High transaction amount',
                'severity': 'high',
                'value': data.get('sales')
            })
        
        if data.get('discount', 0) > 0.35:
            risk_factors.append({
                'factor': 'High discount rate',
                'severity': 'medium',
                'value': data.get('discount')
            })
        
        if data.get('profit', 0) < 0:
            risk_factors.append({
                'factor': 'Negative profit margin',
                'severity': 'high',
                'value': data.get('profit')
            })
        
        if self._is_unusual_time(data.get('order_date')):
            risk_factors.append({
                'factor': 'Unusual transaction time',
                'severity': 'medium',
                'value': data.get('order_date')
            })
        
        if ml_score > 0.7:
            risk_factors.append({
                'factor': 'High ML model risk score',
                'severity': 'high',
                'value': ml_score
            })
        
        if rule_score > 0.5:
            risk_factors.append({
                'factor': 'Multiple risk rules triggered',
                'severity': 'medium',
                'value': len([r for r in self.rules if r['condition'](data)])
            })
        
        return risk_factors
    
    def _get_recommendations(self, risk_level: str, risk_factors: List[Dict]) -> List[str]:
        """Get recommendations based on risk assesment"""
        recommendations = []

        if risk_level in ['critical', 'high']:
            recommendations.extend([
                "Immediately flag transaction for manual review.",
                "Contact customer for verification.",
                "Check for duplicate transactions",
                "Review user acount history"
            ])

        if risk_level == 'medium':
            recommendations.extend([
                "Flag for additional monitoring.",
                "Review transaction patterns",
                "Check for unusual user behavior"
            ])

        # Add specific recommendations based on risk factors
        for factor in risk_factors:
            if 'High transaction amount' in factor['factor']:
                recommendations.append("Verify large transaction with customer.")
            elif 'High discount rate' in factor['factor']:
                recommendations.append("Review discount authorization.")
            elif 'Negative profit' in factor['factor']:
                recommendations.append("Investigate negative margin transactions.")

        return list(dict.fromkeys(recommendations))  # Remove duplicates while preserving order
    
    def adjust_weights(self, new_weights: Dict):
        """Adjust risk scoring weights"""
        self.weights.update(new_weights)
        logger.info(f"✅ Risk weights updated: {new_weights}")