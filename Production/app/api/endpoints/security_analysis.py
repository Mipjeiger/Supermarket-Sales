"""
Security Analysis Endpoints with Agentic Integration
"""

import logging
from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from app.api.dependencies import get_security_agent, get_llm_judge
from app.agentic.agents.security_analyst import SecurityAnalystAgent
from app.agentic.evaluators.llm_as_judge import LLMAsJudge
from app.monitoring.metrics import MetricsCollector
from app.monitoring.slack_notifier import slack_notifier

logger = logging.getLogger(__name__)
router = APIRouter()

class SecurityAnalysisRequest(BaseModel):
    """Request for security analysis."""
    system_data: Dict = Field(..., description="System data for analysis")
    analysis_type: str = Field("comprehensive", description="Analysis type")
    include_recommendations: bool = Field(True)

class SecurityAnalysisResponse(BaseModel):
    """Response from security analysis."""
    security_score: float
    threat_level: str
    threats_identified: List[Dict]
    vulnerabilities: List[Dict]
    posture_level: str
    quality_score: Optional[float]
    recommendations: List[Dict]
    action_required: bool
    timestamp: str

@router.post("/analyze", response_model=SecurityAnalysisResponse)
async def analyze_security(
    request: SecurityAnalysisRequest,
    background_tasks: BackgroundTasks,
    security_agent: SecurityAnalystAgent = Depends(get_security_agent),
    llm_judge: LLMAsJudge = Depends(get_llm_judge)
):
    """
    Conduct security analysis using agentic framework.
    """
    start_time = datetime.now()
    
    try:
        logger.info("🔒 Starting security analysis...")
        
        # Run security analysis
        analysis_result = await security_agent.analyze_threat(request.system_data)
        
        # Evaluate quality
        quality_score = None
        if llm_judge:
            evaluation = await llm_judge.evaluate(
                analysis_result,
                criteria=["threat_detection", "vulnerability_assessment", "risk_analysis", "actionability"]
            )
            quality_score = evaluation.get('overall_score', 0)
        
        # Extract results
        security_posture = analysis_result.get('security_posture', {})
        threats = analysis_result.get('threats_identified', [])
        vulnerabilities = analysis_result.get('vulnerabilities', [])
        recommendations = analysis_result.get('recommendations', [])
        
        # Determine if action is required
        action_required = (
            security_posture.get('level') in ['weak', 'critical'] or
            len(threats) > 0 or
            len(vulnerabilities) > 0
        )
        
        # Send alerts for critical issues
        if security_posture.get('level') == 'critical':
            background_tasks.add_task(
                slack_notifier.send_message,
                f"🔴 Critical Security Issues Detected! Score: {security_posture.get('score', 0)}",
                color="#ff0000"
            )
        
        # Track metrics
        latency = (datetime.now() - start_time).total_seconds()
        MetricsCollector.track_agent_execution("security_analysis", latency, True)
        
        return SecurityAnalysisResponse(
            security_score=security_posture.get('score', 0),
            threat_level=security_posture.get('level', 'unknown'),
            threats_identified=threats[:10],  # Limit to 10
            vulnerabilities=vulnerabilities[:10],  # Limit to 10
            posture_level=security_posture.get('level', 'unknown'),
            quality_score=quality_score,
            recommendations=recommendations[:10],  # Limit to 10
            action_required=action_required,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Security analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_security_status(
    security_agent: SecurityAnalystAgent = Depends(get_security_agent)
):
    """
    Get overall security status.
    """
    try:
        status = await security_agent.get_security_status()
        return status
    except Exception as e:
        logger.error(f"❌ Failed to get security status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))