from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import pandas as pd
from app.services.behavior_analyst import behavior_analyst

router = APIRouter()

class RecommendationRequest(BaseModel):
    order_id: str = Field(..., description="Unique identifier for the customer.")
    transaction_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary mapping exactly to your engineering.supermarket table columns",
    )


@router.post("/generate-offer")
async def generate_offer(payload: RecommendationRequest):
    """
    Accepts historical user feature states, evaluates financial ceilings using CatBoost,
    and streams back a grounded cross-sell report via serverless LLM layers.
    """
    try:
        if not payload.transaction_data:
            raise HTTPException(status_code=400, detail="Transaction data is required.")

        # Standardize flat row attributes into a Dataframe instance
        df_features = pd.DataFrame([payload.transaction_data])

        report = behavior_analyst.generate_personalized_offers(
            order_id=payload.order_id, last_transaction_df=df_features
        )

        return {
            "status": "success",
            "order_id": payload.order_id,
            "recommendation_report": report,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
