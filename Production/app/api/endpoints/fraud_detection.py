import logging
from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

# Import from main using absolute import
from app.api.dependencies import get_fraud_agent, get_llm_judge
from app.agentic.agents.fraud_detector import FraudDetectionAgent
from app.agentic.evaluators.llm_as_judge import LLMAsJudge
from app.monitoring.metrics import MetricsCollector
from app.monitoring.slack_notifier import slack_notifier

logger = logging.getLogger(__name__)
router = APIRouter()

"""
Fraud Detection Endpoints with Agentic Integration
"""

class TransactionData(BaseModel):
    """Transaction data for fraud detection."""
    transaction_id: str = Field(..., description="Unique transaction ID")
    user_id: str = Field(..., description="User or customer ID")
    amount: float = Field(..., description="Transaction amount")
    timestamp: str = Field(..., description="Transaction timestamp")
    merchant_id: str = Field(..., description="Merchant ID")
    category: str = Field(..., description="Transaction category")
    from app.api.dependencies import get_fraud_agent, get_llm_judge
    device_info: Dict = Field(..., description="Device information")
    additional_data: Optional[Dict] = Field(default={})

class FraudDetectionResponse(BaseModel):
    """Fraud detection response."""
    transaction_id: str
    fraud_score: float
    risk_level: str
    is_fraudulent: bool
    investigation_id: Optional[str]
    confidence: float
    quality_score: Optional[float]
    recommendations: List[str]
    timestamp: str

@router.post("/detect", response_model=FraudDetectionResponse)
async def detect_fraud(
    transaction: TransactionData,
    background_tasks: BackgroundTasks,
    fraud_agent: FraudDetectionAgent = Depends(get_fraud_agent),
    llm_judge: LLMAsJudge = Depends(get_llm_judge)
):
    """
    Detect fraud using agentic framework.
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"🔍 Detecting fraud for transaction: {transaction.transaction_id}")
        
        # Run fraud investigation
        investigation_result = await fraud_agent.investigate_fraud(
            transaction.dict()
        )
        
        # Evaluate quality using LLM-as-Judge
        quality_score = None
        if llm_judge:
            evaluation = await llm_judge.evaluate(
                investigation_result,
                criteria=["accuracy", "thoroughness", "evidence_quality", "actionability"]
            )
            quality_score = evaluation.get('overall_score', 0)
        
        # Extract risk assessment
        risk_assessment = investigation_result.get('risk_assessment', {})
        fraud_score = risk_assessment.get('score', 0)
        
        # Determine risk level
        risk_level = _get_risk_level(fraud_score)
        
        # Generate recommendations
        recommendations = _get_recommendations(risk_level)
        
        # Track metrics
        latency = (datetime.now() - start_time).total_seconds()
        MetricsCollector.track_fraud_detection(latency, risk_level)
        
        # Send alerts for high-risk transactions
        if risk_level in ['high', 'critical']:
            background_tasks.add_task(
                slack_notifier.send_message,
                f"🚨 Critical Fraud Alert! Transaction: {transaction.transaction_id}, Risk: {risk_level}",
                color="#ff0000"
            )
        
        return FraudDetectionResponse(
            transaction_id=transaction.transaction_id,
            fraud_score=fraud_score,
            risk_level=risk_level,
            is_fraudulent=risk_level in ['high', 'critical'],
            investigation_id=investigation_result.get('investigation_id'),
            confidence=investigation_result.get('confidence', 0.5),
            quality_score=quality_score,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Fraud detection failed: {str(e)}")
        await slack_notifier.send_error_alert(f"Fraud detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/investigation/{investigation_id}")
async def get_fraud_investigation(
    investigation_id: str,
    fraud_agent: FraudDetectionAgent = Depends(get_fraud_agent)
):
    """
    Get details of a specific fraud investigation.
    """
    try:
        # Search investigation history
        for record in fraud_agent.investigation_history:
            if record.get('investigation_id') == investigation_id:
                return record
        
        raise HTTPException(status_code=404, detail="Investigation not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get investigation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_fraud_history(
    limit: int = 10,
    fraud_agent: FraudDetectionAgent = Depends(get_fraud_agent)
):
    """
    Get fraud investigation history.
    """
    try:
        history = fraud_agent.investigation_history[-limit:]
        return {
            "total": len(fraud_agent.investigation_history),
            "recent": history,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_risk_level(score: float) -> str:
    """Determine risk level from score."""
    if score >= 0.8:
        return 'critical'
    elif score >= 0.6:
        return 'high'
    elif score >= 0.3:
        return 'medium'
    else:
        return 'low'

def _get_recommendations(risk_level: str) -> List[str]:
    """Get recommendations based on risk level."""
    recommendations = {
        'critical': [
            "Immediately block transaction",
            "Alert security team",
            "Freeze user account",
            "Initiate incident response"
        ],
        'high': [
            "Flag for manual review",
            "Restrict user actions",
            "Notify fraud team",
            "Collect additional evidence"
        ],
        'medium': [
            "Monitor closely",
            "Review transaction patterns",
            "Check user history"
        ],
        'low': [
            "No immediate action required",
            "Continue monitoring"
        ]
    }
    return recommendations.get(risk_level, ["No recommendations available"])