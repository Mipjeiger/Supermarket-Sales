from typing import Dict, List, Any
from datetime import datetime
import asyncio
import json
import logging

from app.agentic.agents.fraud_detector import FraudDetectionAgent
from app.agentic.agents.abuse_investigator import AbuseInvestigationAgent
from app.agentic.agents.security_analyst import SecurityAnalystAgent
from app.agentic.evaluators.llm_as_judge import LLMAsJudge

logger = logging.getLogger(__name__)

class MultiStepInvestigationWorkflow:
    """
    Orchestrates multi-step agentic investigations for fraud and abuse.
    """
    def __init__(self, llm_model, vector_store):
        self.fraud_agent = FraudDetectionAgent(llm_model, vector_store)
        self.abuse_agent = AbuseInvestigationAgent(llm_model, vector_store)
        self.security_agent = SecurityAnalystAgent(llm_model, vector_store)
        self.llm_judge = LLMAsJudge(llm_model)
        self.investigation_pipeline = []

    async def run_investigation(self, incident_data: Dict) -> Dict:
        """
        Run a complete multi-step investigation.
        """
        investigation_id = f"INV-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"🔍 Starting investigation {investigation_id}")
        
        try:
            # Step 1: Initial triage
            triage_result = await self._perform_triage(incident_data)
            
            # Step 2: Parallel investigation
            investigation_tasks = [
                self.fraud_agent.investigate_fraud(incident_data),
                self.abuse_agent.investigate_abuse(incident_data),
                self.security_agent.analyze_threat(incident_data)
            ]
            
            investigation_results = await asyncio.gather(
                *investigation_tasks,
                return_exceptions=True
            )
            
            # Step 3: Synthesize findings
            synthesized_findings = await self._synthesize_findings(
                triage_result,
                investigation_results
            )
            
            # Step 4: Quality evaluation
            quality_scores = await self._evaluate_quality(synthesized_findings)
            
            # Step 5: Generate comprehensive report
            final_report = await self._generate_comprehensive_report(
                investigation_id,
                incident_data,
                triage_result,
                synthesized_findings,
                quality_scores
            )
            
            # Step 6: Automate actions based on severity
            actions = await self._determine_actions(synthesized_findings)
            
            return {
                "investigation_id": investigation_id,
                "status": "completed",
                "findings": synthesized_findings,
                "quality_scores": quality_scores,
                "report": final_report,
                "actions": actions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Investigation failed: {str(e)}")
            return {
                "investigation_id": investigation_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
    async def _perform_triage(self, incident_data: Dict) -> Dict:
        """Perform initial triage to determine priority and approach."""
        
        # Logic for initial triage
        return {
            "priority": "high",
            "severity": "critical",
            "requires_immediate_action": True,
            "affected_systems": ["payment_gateway", "user_accounts"]
        }
    
    async def _synthesize_findings(self, triage: Dict, results: List) -> Dict:
        """Synthesize findings from multiple investigation agents."""
        synthesized = {
            "fraud_indicators": [],
            "abuse_patterns": [],
            "security_threats": [],
            "risk_assessment": {},
            "recommendations": []
        }
        
        # Process each result
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Agent failed: {str(result)}")
                continue
                
            if "fraud" in result.get("type", ""):
                synthesized["fraud_indicators"].extend(
                    result.get("findings", [])
                )
            elif "abuse" in result.get("type", ""):
                synthesized["abuse_patterns"].extend(
                    result.get("patterns", [])
                )
            elif "security" in result.get("type", ""):
                synthesized["security_threats"].extend(
                    result.get("threats", [])
                )
        
        # Calculate overall risk
        synthesized["risk_assessment"] = self._calculate_risk(synthesized)
        
        return synthesized
    
    def _calculate_risk(self, synthesized: Dict) -> Dict:
        """Calculate overall risk score."""
        # In production, use ML models for risk scoring
        risk_score = 0
        risk_score += len(synthesized["fraud_indicators"]) * 3
        risk_score += len(synthesized["abuse_patterns"]) * 2
        risk_score += len(synthesized["security_threats"]) * 5
        
        return {
            "overall_risk_score": min(risk_score, 100),
            "level": "critical" if risk_score > 70 else "high" if risk_score > 40 else "medium",
            "factors": {
                "fraud": len(synthesized["fraud_indicators"]),
                "abuse": len(synthesized["abuse_patterns"]),
                "security": len(synthesized["security_threats"])
            }
        }
    
    async def _evaluate_quality(self, findings: Dict) -> Dict:
        """Evaluate quality of investigation findings."""
        return await self.llm_judge.evaluate(
            json.dumps(findings),
            criteria=["thoroughness", "accuracy", "actionability", "completeness"]
        )
    
    async def _generate_comprehensive_report(self, 
                                             investigation_id: str,
                                             incident_data: Dict,
                                             triage: Dict,
                                             findings: Dict,
                                             quality: Dict) -> Dict:
        """Generate comprehensive investigation report."""
        return {
            "investigation_id": investigation_id,
            "executive_summary": f"Investigation completed with {quality.get('score', 0)}/10 quality score",
            "triage_assessment": triage,
            "detailed_findings": findings,
            "quality_assessment": quality,
            "recommendations": findings.get("recommendations", []),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _determine_actions(self, findings: Dict) -> List[Dict]:
        """Determine automated actions based on findings."""
        actions = []
        
        risk_level = findings.get("risk_assessment", {}).get("level", "low")
        
        if risk_level == "critical":
            actions.append({
                "type": "immediate_block",
                "description": "Block suspicious transactions",
                "priority": "critical"
            })
            actions.append({
                "type": "alert",
                "description": "Alert security team immediately",
                "priority": "critical"
            })
        elif risk_level == "high":
            actions.append({
                "type": "flag_review",
                "description": "Flag for manual review",
                "priority": "high"
            })
        else:
            actions.append({
                "type": "monitor",
                "description": "Add to monitoring list",
                "priority": "medium"
            })
        
        return actions