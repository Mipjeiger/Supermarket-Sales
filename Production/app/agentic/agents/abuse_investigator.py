"""
Abuse Detection Endpoints with Agentic Integration
"""

import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from app.main import get_abuse_agent, get_llm_judge
from app.agentic.agents.abuse_investigator import AbuseInvestigationAgent
from app.agentic.evaluators.llm_as_judge import LLMAsJudge
from app.monitoring.metrics import MetricsCollector
from app.monitoring.slack_notifier import slack_notifier

logger = logging.getLogger(__name__)
router = APIRouter()

class AbuseReportRequest(BaseModel):
    """Request for abuse investigation."""
    user_id: str = Field(..., description="User ID to investigate")
    reason: str = Field(..., description="Reason for investigation")
    evidence: List[Dict] = Field(default=[], description="Supporting evidence")
    priority: str = Field("medium", description="Priority: low, medium, high, critical")

class AbuseInvestigationResponse(BaseModel):
    """Response from abuse investigation."""
    user_id: str
    abuse_detected: bool
    abuse_type: Optional[str]
    severity: str
    confidence: float
    quality_score: Optional[float]
    findings: List[Dict]
    recommendations: List[str]
    action_required: bool
    timestamp: str

@router.post("/investigate", response_model=AbuseInvestigationResponse)
async def investigate_abuse(
    request: AbuseReportRequest,
    background_tasks: BackgroundTasks,
    abuse_agent: AbuseInvestigationAgent = Depends(get_abuse_agent),
    llm_judge: LLMAsJudge = Depends(get_llm_judge)
):
    """
    Conduct abuse investigation using agentic framework.
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"🔍 Starting abuse investigation for user: {request.user_id}")
        
        # Prepare user data
        user_data = {
            "user_id": request.user_id,
            "reason": request.reason,
            "evidence": request.evidence,
            "priority": request.priority
        }
        
        # Run abuse investigation
        investigation_result = await abuse_agent.investigate_abuse(user_data)
        
        # Evaluate quality
        quality_score = None
        if llm_judge:
            evaluation = await llm_judge.evaluate(
                investigation_result,
                criteria=["detection_accuracy", "classification", "severity_assessment", "actionability"]
            )
            quality_score = evaluation.get('overall_score', 0)
        
        # Send alerts for high-severity abuse
        severity = investigation_result.get('severity', 'low')
        if severity in ['high', 'critical']:
            background_tasks.add_task(
                slack_notifier.send_message,
                f"🚨 Abuse Alert: {severity.upper()} severity detected for user {request.user_id}",
                color="#ff0000"
            )
        
        # Track metrics
        latency = (datetime.now() - start_time).total_seconds()
        MetricsCollector.track_agent_execution("abuse_investigation", latency, True)
        
        return AbuseInvestigationResponse(
            user_id=request.user_id,
            abuse_detected=investigation_result.get('abuse_detected', False),
            abuse_type=investigation_result.get('abuse_type'),
            severity=severity,
            confidence=investigation_result.get('confidence', 0),
            quality_score=quality_score,
            findings=investigation_result.get('findings', []),
            recommendations=investigation_result.get('recommendations', []),
            action_required=severity in ['high', 'critical'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Abuse investigation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))