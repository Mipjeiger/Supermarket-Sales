import logging
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from app.services.model_registry import model_registry
from app.services.llm_provider import llm_provider
from app.prompts.behavior_templates import BEHAVIOR_SYSTEM_PROMPT, BEHAVIOR_USER_PROMPT

logger = logging.getLogger(__name__)


class BehaviorAnalystEngine:
    def __init__(self):
        self.model_name = "CatboostRegressor_model"

    def predict_spend_ceiling(self, historical_features: pd.DataFrame) -> float:
        try:
            model = model_registry.get_model(self.model_name)

            if model is None:
                return 500.0  # Default to baseline fallback
            raw_pred = model.predict(historical_features)
            return float(
                np.expm1(raw_pred[0])
            )  # Convert log-transformed prediction back to original scale

        except Exception as e:
            logger.error(f"❌ Upstream regressor failed: {e}")
            return 500.0

    def generate_personalized_offers(
        self, customer_id: str, last_transaction_df: pd.DataFrame
    ) -> str:
        """Fuse structural database columns with ML thresholds to ground the LLM."""

        # Calculate spending boundary using the ML model
        spend_ceiling = self.predict_spend_ceiling(last_transaction_df)

        # Extract structural values from the Dataframe row matching the DB schema
        row_dict = last_transaction_df.iloc[0].to_dict()

        segment = row_dict.get("segment", "Consumer")
        region = row_dict.get("region", "Global")

        # Isolate column to pass as verified history context
        grounded_keys = [
            "order_id",
            "category",
            "sub_category",
            "product_name",
            "sales",
            "quantity",
            "discount",
            "profit_margin",
            "unit_price",
        ]
        history_summary = {k: row_dict[k] for k in grounded_keys if k in row_dict}

        # Compile the structural prompt parameters
        user_prompt = BEHAVIOR_USER_PROMPT.format(
            customer_id=customer_id,
            segment=segment,
            region=region,
            historical_rows_json=json.dumps(history_summary, indent=2, default=str),
            spend_ceiling=spend_ceiling,
        )

        # Invoke LLM via provider recommended
        return llm_provider.generate_grounded_text(
            prompt=user_prompt,
            system_instruction=BEHAVIOR_SYSTEM_PROMPT,
            hf_model="meta-llama/Llama-3.1-8B-Instruct",
            groq_model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
