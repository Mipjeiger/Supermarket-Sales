import logging
import json
import pandas as pd
from typing import Dict, Any
from app.services.model_registry import model_registry
from app.services.llm_provider import llm_provider
from app.prompts.anomaly_templates import ANOMALY_SYSTEM_PROMPT, ANOMALY_USER_PROMPT

logger = logging.getLogger(__name__)

class AnomalyAgentEngine:
    def __init__(self):
        self.classifier_name = "XGBClassifier"

    def evaluate_transaction_risk(self, feature_vector: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate the risk of a transaction using the XGBoost classifier."""
        try:
            model = model_registry.get_model(self.classifier_name)

            if model is None:
                return {"flag": 0, "probability": 0.05}
            
            # Predict the risk flag and probability
            prediction = model.predict(feature_vector)[0]
            probability = float(model.predict_proba(feature_vector)[0][1]) if hasattr(model, "predict_proba") else 0.5
            return {"flag": int(prediction), "probability": probability}
        
        except Exception as e:
            logger.error(f"XGBoost scoring extraction error: {e}")
            return {"flag": 0, "probability": 0.0}  # Default to safe with low probability
        
    def generate_security_brief(self, streaming_db_row: pd.DataFrame, risk_metrics: Dict[str, Any]) -> str:
        """Injects clean structured transactional rows directly into the analysis loop."""

        # Run inference scoring via XGBoost classifier
        ml_eval = self.evaluate_transaction_risk(streaming_db_row)

        # Extract and format row variables explicitly
        row_dict = streaming_db_row.iloc[0].to_dict()
        audit_keys = ["order_id", "sales", "quantity", "discount", "profit", "shipping_cost", "profit_margin"]
        structured_audit_json = {k: row_dict[k] for k in audit_keys if k in row_dict}

        # Bind features to structural templates
        user_prompt = ANOMALY_USER_PROMPT.format(
            database_row_json=json.dumps(structured_audit_json, indent=2, default=str),
            flag=ml_eval["flag"],
            probability=ml_eval["probability"],
            abuse_score=risk_metrics.get("abuse_score", 0.0)
        )

        # Route to ultra-low-latency Groq hardware clusters for grounded LLM inference
        return llm_provider.generate_grounded_text(
            prompt=user_prompt,
            system_instruction=ANOMALY_SYSTEM_PROMPT,
            hf_model="meta-llama/Llama-3.1-8B-Instruct",
            groq_model="llama-3.3-70b-versatile",
            temperature=0.0
        )
    
# Singleton initialization for global anomaly agent access
anomaly_agent = AnomalyAgentEngine()