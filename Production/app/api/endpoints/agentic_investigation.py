import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio

from app.api.dependencies import (
    get_fraud_agent,
    get_abuse_agent,
    get_security_agent,
    get_llm_judge
)
from app.agentic.agents.fraud_detector import FraudDetectionAgent
from app.agentic.agents.abuse_investigator import AbuseInvestigationAgent
from app.agentic.agents.security_analyst import SecurityAnalystAgent
from app.agentic.evaluators.llm_as_judge import LLMAsJudge
from app.monitoring.slack_notifier import slack_notifier

logger = logging.getLogger(__name__)
router = APIRouter()

"""
Multi-Agent Investigation Endpoints
Orchestrates multiple agents for comprehensive investigations
"""

class ComprehensiveInvestigationRequest(BaseModel):
    """Request for comprehensive investigation."""
    incident_data: Dict = Field(..., description="Incident data to investigate")
    investigation_types: List[str] = Field(
        ["fraud", "abuse", "security"],
        description="Types of investigations to run"
    )
    depth: str = Field("standard", description="Investigation depth: quick, standard, thorough")
    include_quality_evaluation: bool = Field(True)

class ComprehensiveInvestigationResponse(BaseModel):
    """Response from comprehensive investigation."""
    investigation_id: str
    findings: Dict
    overall_risk_level: str
    confidence_score: float
    quality_scores: Dict
    recommendations: List[str]
    action_required: bool
    timestamp: str

@router.post("/comprehensive", response_model=ComprehensiveInvestigationResponse)
async def comprehensive_investigation(
    request: ComprehensiveInvestigationRequest,
    background_tasks: BackgroundTasks,
    fraud_agent: FraudDetectionAgent = Depends(get_fraud_agent),
    abuse_agent: AbuseInvestigationAgent = Depends(get_abuse_agent),
    security_agent: SecurityAnalystAgent = Depends(get_security_agent),
    llm_judge: LLMAsJudge = Depends(get_llm_judge)
):
    """
    Run comprehensive investigation using multiple agents.
    """
    investigation_id = f"INV-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    try:
        logger.info(f"🔍 Starting comprehensive investigation: {investigation_id}")
        
        # Initialize investigation tasks
        investigation_tasks = []
        
        if "fraud" in request.investigation_types:
            investigation_tasks.append(
                fraud_agent.investigate_fraud(request.incident_data)
            )
        
        if "abuse" in request.investigation_types:
            investigation_tasks.append(
                abuse_agent.investigate_abuse(request.incident_data)
            )
        
        if "security" in request.investigation_types:
            investigation_tasks.append(
                security_agent.analyze_threat(request.incident_data)
            )
        
        # Run all investigations in parallel
        results = await asyncio.gather(
            *investigation_tasks,
            return_exceptions=True
        )
        
        # Process results
        processed_results = await _process_investigation_results(
            results,
            request.investigation_types
        )
        
        # Evaluate quality if requested
        quality_scores = {}
        if request.include_quality_evaluation and llm_judge:
            for result in results:
                if not isinstance(result, Exception):
                    eval_result = await llm_judge.evaluate(
                        result,
                        criteria=["thoroughness", "accuracy", "actionability"]
                    )
                    quality_scores[result.get('type', 'unknown')] = eval_result.get('overall_score', 0)
        
        # Generate overall assessment
        overall_assessment = await _generate_overall_assessment(
            processed_results,
            quality_scores
        )
        
        # Send alerts for high-risk findings
        if overall_assessment.get('overall_risk_level') in ['high', 'critical']:
            background_tasks.add_task(
                slack_notifier.send_message,
                f"🚨 Comprehensive Investigation {investigation_id}: {overall_assessment['overall_risk_level'].upper()} risk detected!",
                color="#ff0000"
            )
        
        return ComprehensiveInvestigationResponse(
            investigation_id=investigation_id,
            findings=processed_results,
            overall_risk_level=overall_assessment.get('overall_risk_level', 'low'),
            confidence_score=overall_assessment.get('confidence_score', 0),
            quality_scores=quality_scores,
            recommendations=overall_assessment.get('recommendations', []),
            action_required=overall_assessment.get('action_required', False),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Comprehensive investigation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_investigation_results(results: List, types: List[str]) -> Dict:
    """Process and combine investigation results."""
    processed = {
        "fraud": {},
        "abuse": {},
        "security": {},
        "summary": {}
    }
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed[types[i]] = {"error": str(result)}
            continue
        
        result_type = result.get('type', types[i])
        processed[result_type] = {
            "status": result.get('status', 'completed'),
            "detected": result.get('fraud_detected', result.get('abuse_detected', result.get('threats_identified', []))),
            "severity": result.get('severity', result.get('risk_level', result.get('security_posture', {}).get('level', 'low'))),
            "confidence": result.get('confidence', result.get('security_score', 0)),
            "findings": result.get('findings', result.get('threats_identified', result.get('vulnerabilities', [])))
        }
    
    return processed

async def _generate_overall_assessment(results: Dict, quality_scores: Dict) -> Dict:
    """Generate overall assessment from all results."""
    # Calculate overall risk level
    risk_levels = []
    for key, value in results.items():
        if key in ['fraud', 'abuse', 'security']:
            severity = value.get('severity', 'low')
            risk_levels.append(severity)
    
    # Get highest risk level
    risk_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    max_risk = max(risk_levels, key=lambda x: risk_order.get(x, 0)) if risk_levels else 'low'
    
    # Calculate confidence score
    confidence_scores = []
    for key, value in results.items():
        if key in ['fraud', 'abuse', 'security']:
            confidence = value.get('confidence', 0)
            if isinstance(confidence, (int, float)):
                confidence_scores.append(confidence)
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    # Generate recommendations
    recommendations = []
    if max_risk in ['high', 'critical']:
        recommendations.extend([
            "Immediate investigation required",
            "Alert all relevant teams",
            "Preserve all evidence",
            "Initiate incident response protocol"
        ])
    elif max_risk == 'medium':
        recommendations.extend([
            "Schedule detailed review",
            "Collect additional data",
            "Monitor closely"
        ])
    else:
        recommendations.append("Continue standard monitoring")
    
    return {
        "overall_risk_level": max_risk,
        "confidence_score": avg_confidence,
        "recommendations": recommendations,
        "action_required": max_risk in ['high', 'critical']
    }