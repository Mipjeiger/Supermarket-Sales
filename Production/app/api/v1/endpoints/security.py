from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import pandas as pd
from app.services.anomaly_agent import anomaly_agent

router = APIRouter()


class SecurityStreamRequest(BaseModel):
    streaming_db_row: Dict[str, Any] = Field(
        ...,
        description="Real-time transaction features corresponding to your database columns",
    )
    risk_metrics: Dict[str, Any] = Field(
        ...,
        description="Upstream risk context elements, including baseline pipeline abuse velocity scores",
    )


@router.post("/analyze-stream")
async def analyze_stream(payload: SecurityStreamRequest):
    """
    Evaluates transactional risk via XGBoost, extracts structural data fields,
    and streams down a concise terminal enforcement brief.
    """
    try:
        if not payload.streaming_db_row:
            raise HTTPException(
                status_code=400, detail="Streaming transaction data payload is empty."
            )

        # Parse into structural framework array representation
        df_row = pd.DataFrame([payload.streaming_db_row])

        brief = anomaly_agent.evaluate_and_notify_stream(
            streaming_db_row=df_row, risk_metrics=payload.risk_metrics
        )
        return {"status": "processed", "decision_brief": brief}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Security stream processing blackout event triggered: {str(e)}",
        )
