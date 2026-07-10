import asyncio
import logging
import json
import pandas as pd
from typing import Dict, Any
from app.services.model_registry import model_registry
from app.services.llm_provider import llm_provider
from app.prompts.anomaly_templates import ANOMALY_SYSTEM_PROMPT, ANOMALY_USER_PROMPT
from app.monitoring.slack_notifier import slack_notifier
from app.core.config import settings

logger = logging.getLogger(__name__)


class AnomalyAgentEngine:
    def __init__(self):
        self.classifier_name = "GradientBoostingClassifier"

    def _send_slack_alert_sync(
        self, error_message: str, channel: str, severity: str = "critical"
    ) -> None:
        """Run the async Slack alert from synchronous code."""
        try:
            asyncio.run(
                slack_notifier.send_error_alert(
                    error_message=error_message,
                    severity=severity,
                )
            )

        except RuntimeError:
            loop = asyncio.new_event_loop()

            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    slack_notifier.send_error_alert(
                        error_message=error_message,
                        severity=severity,
                    )
                )
            finally:
                loop.close()

    def evaluate_transaction_risk(self, feature_vector: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate the risk of a transaction using the GradientBoostingClassifier."""
        try:
            model = model_registry.get_model(self.classifier_name)

            if model is None:
                return {"flag": 0, "probability": 0.05}

            # Predict the risk flag and probability
            prediction = model.predict(feature_vector)[0]
            probability = (
                float(model.predict_proba(feature_vector)[0][1])
                if hasattr(model, "predict_proba")
                else 0.5
            )
            return {"flag": int(prediction), "probability": probability}

        except Exception as e:
            logger.error(f"GradientBoostingClassifier scoring extraction error: {e}")
            return {
                "flag": 0,
                "probability": 0.0,
            }  # Default to safe with low probability

    def evaluate_and_notify_stream(
        self, streaming_db_row: pd.DataFrame, risk_metrics: Dict[str, Any]
    ) -> str:
        """Injects clean structured transactional rows directly into the analysis loop."""

        # Run inference scoring via GradientBoostingClassifier
        ml_eval = self.evaluate_transaction_risk(streaming_db_row)

        # Extract and format row variables explicitly
        row_dict = streaming_db_row.iloc[0].to_dict()
        audit_keys = [
            "order_id",
            "sales",
            "quantity",
            "discount",
            "profit",
            "shipping_cost",
            "profit_margin",
        ]
        structured_audit_json = {k: row_dict[k] for k in audit_keys if k in row_dict}

        # Bind features to structural templates
        user_prompt = ANOMALY_USER_PROMPT.format(
            database_row_json=json.dumps(structured_audit_json, indent=2, default=str),
            flag=ml_eval["flag"],
            probability=ml_eval["probability"],
            abuse_score=risk_metrics.get("abuse_score", 0.0),
        )

        # Generate gorunded security assessment summary via LLM provider
        brief = llm_provider.generate_grounded_text(
            prompt=user_prompt,
            system_instruction=ANOMALY_SYSTEM_PROMPT,
            hf_model="meta-llama/Llama-3.1-8B-Instruct",
            groq_model="llama-3.3-70b-versatile",
            temperature=0.0,
        )

        # Conditionally send Slack notification if risk is flagged
        if ml_eval["flag"] == 1:
            slack_alert_body = (
                f"🚨 *CRITICAL MALICIOUS FRAUD DETECTED* 🚨\n"
                f"• *Order Reference:* `{structured_audit_json.get('order_id', 'N/A')}`\n"
                f"• *Customer Name:* {structured_audit_json.get('customer_name', 'Unknown')}\n"
                f"• *Financial Hit (Sales):* ${structured_audit_json.get('sales', 0):,.2f}\n"
                f"• *CatBoost Confidence:* {ml_eval['probability']:.2%}\n\n"
                f"*AI Decision Executive Brief:*\n{brief}"
            )
            self._send_slack_alert_sync(
                error_message=slack_alert_body,
                channel=settings.SLACK_CHANNEL,
                severity="critical",
            )

        return brief


# Singleton initialization for global anomaly agent access
anomaly_agent = AnomalyAgentEngine()
