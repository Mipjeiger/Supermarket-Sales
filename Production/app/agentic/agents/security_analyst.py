"""
Security Analyst Agent
Analyzes security threats, vulnerabilities, and provides security recommendations.
Handles:
- Threat detection
- Vulnerability assessment
- Security posture analysis
- Incident response
- Compliance monitoring
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

from app.agentic.tools.data_retrieval import DataRetrievalTool
from app.agentic.tools.pattern_analysis import PatternAnalysisTool
from app.agentic.tools.risk_scoring import RiskScoringTool
from app.agentic.tools.report_generator import ReportGeneratorTool
from app.agentic.evaluators.llm_as_judge import LLMAsJudge

logger = logging.getLogger(__name__)

class SecurityAnalystAgent:
    """
    Specialized agent for security analysis and threat detection.
    Combines ML models with LLM reasoning for comprehensive security assessment.
    """
    
    def __init__(self, llm_model=None, vector_store=None):
        self.llm = llm_model
        self.vector_store = vector_store
        self.tools = self._initialize_tools()
        self.agent = self._create_agent()
        self.llm_judge = LLMAsJudge(llm_model)
        self.security_findings = []
        self.threat_intelligence = self._load_threat_intelligence()
        
    def _initialize_tools(self) -> List[Tool]:
        """Initialize specialized tools for security analysis."""
        return [
            Tool(
                name="DataRetrieval",
                func=DataRetrievalTool().run,
                description="Retrieves system logs, security events, and access data"
            ),
            Tool(
                name="PatternAnalysis",
                func=PatternAnalysisTool().run,
                description="Analyzes patterns in security events and anomalies"
            ),
            Tool(
                name="RiskScoring",
                func=RiskScoringTool().run,
                description="Calculates security risk scores"
            ),
            Tool(
                name="ReportGenerator",
                func=ReportGeneratorTool().run,
                description="Generates detailed security analysis reports"
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent with ReAct framework."""
        prompt = PromptTemplate.from_template("""
        You are an expert security analyst agent for an enterprise system.
        
        Available tools:
        {tools}
        
        Tool names: {tool_names}
        
        Objective: Analyze security threats, vulnerabilities, and provide recommendations.
        
        Security Analysis Process:
        1. Identify potential threats and vulnerabilities
        2. Assess risk levels and impact
        3. Investigate using available tools
        4. Recommend mitigation strategies
        5. Document findings and recommendations
        
        Input: {input}
        
        Agent Scratchpad:
        {agent_scratchpad}
        """)
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def _load_threat_intelligence(self) -> Dict:
        """Load threat intelligence data."""
        return {
            'known_threats': {
                'sql_injection': {
                    'severity': 'critical',
                    'indicators': ['sql_error', 'database_error', 'query_manipulation'],
                    'mitigation': 'input_validation_parameterized_queries'
                },
                'xss': {
                    'severity': 'high',
                    'indicators': ['script_tags', 'javascript_events', 'html_injection'],
                    'mitigation': 'input_sanitization_content_security_policy'
                },
                'brute_force': {
                    'severity': 'medium',
                    'indicators': ['multiple_failures', 'rapid_requests', 'auth_errors'],
                    'mitigation': 'rate_limiting_account_lockout'
                },
                'data_exfiltration': {
                    'severity': 'critical',
                    'indicators': ['large_downloads', 'unusual_queries', 'data_transfer'],
                    'mitigation': 'dll_monitoring_access_control'
                },
                'privilege_escalation': {
                    'severity': 'critical',
                    'indicators': ['admin_actions', 'permission_changes', 'sensitive_access'],
                    'mitigation': 'principle_least_privilege_audit_trails'
                }
            },
            'vulnerability_classes': {
                'owasp_top_10': [
                    'broken_access_control',
                    'cryptographic_failures',
                    'injection',
                    'insecure_design',
                    'security_misconfiguration',
                    'vulnerable_components',
                    'identification_auth_failures',
                    'software_data_integrity',
                    'security_logging_monitoring',
                    'server_side_request_forgery'
                ]
            }
        }
    
    async def analyze_threat(self, security_data: Dict) -> Dict:
        """
        Analyze security threats and vulnerabilities.
        
        Args:
            security_data: Dict containing security events, logs, and system data
            
        Returns:
            Dict with security analysis results
        """
        try:
            logger.info("🔒 Starting security analysis...")
            
            # Step 1: Initial threat assessment
            threat_assessment = await self._initial_threat_assessment(security_data)
            
            # Step 2: Run agentic analysis
            analysis_result = await self.agent.arun(
                input=json.dumps(security_data)
            )
            
            # Step 3: LLM-as-Judge evaluation
            quality_score = await self.llm_judge.evaluate(
                analysis_result,
                criteria=[
                    "threat_detection_accuracy",
                    "vulnerability_identification",
                    "risk_assessment",
                    "recommendation_quality"
                ]
            )
            
            # Step 4: Identify threats
            identified_threats = self._identify_threats(security_data, analysis_result)
            
            # Step 5: Assess vulnerabilities
            vulnerabilities = self._assess_vulnerabilities(security_data)
            
            # Step 6: Calculate security posture
            security_posture = self._calculate_security_posture(
                identified_threats,
                vulnerabilities
            )
            
            # Step 7: Generate recommendations
            recommendations = self._generate_security_recommendations(
                identified_threats,
                vulnerabilities
            )
            
            # Step 8: Store findings
            self.security_findings.append({
                "timestamp": datetime.now().isoformat(),
                "threats_identified": len(identified_threats),
                "security_score": security_posture.get("score", 0),
                "quality_score": quality_score
            })
            
            return {
                "status": "completed",
                "threats_identified": identified_threats,
                "vulnerabilities": vulnerabilities,
                "security_posture": security_posture,
                "recommendations": recommendations,
                "quality_score": quality_score,
                "analysis_summary": self._generate_security_summary(
                    identified_threats,
                    vulnerabilities,
                    security_posture
                ),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Security analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _initial_threat_assessment(self, security_data: Dict) -> Dict:
        """Perform initial threat assessment."""
        assessment = {
            "threat_level": "low",
            "indicators": [],
            "immediate_actions": []
        }
        
        # Check for critical indicators
        if security_data.get('failed_auth_attempts', 0) > 100:
            assessment["threat_level"] = "critical"
            assessment["indicators"].append("multiple authentication failures")
            assessment["immediate_actions"].append("block_source_ip")
        
        if security_data.get('sql_errors', 0) > 10:
            assessment["threat_level"] = "high"
            assessment["indicators"].append("sql error patterns detected")
            assessment["immediate_actions"].append("enable_sql_firewall")
        
        if security_data.get('unusual_downloads', False):
            assessment["threat_level"] = "high"
            assessment["indicators"].append("unusual data download activity")
            assessment["immediate_actions"].append("investigate_download_patterns")
        
        if security_data.get('admin_actions', 0) > 5:
            assessment["threat_level"] = "medium"
            assessment["indicators"].append("unusual admin activity")
            assessment["immediate_actions"].append("audit_admin_actions")
        
        return assessment
    
    def _identify_threats(self, security_data: Dict, analysis: Dict) -> List[Dict]:
        """Identify specific threats from data."""
        threats = []
        
        # Check against known threat intelligence
        for threat_name, threat_info in self.threat_intelligence['known_threats'].items():
            indicators_found = []
            for indicator in threat_info['indicators']:
                if indicator in str(security_data).lower() or indicator in str(analysis).lower():
                    indicators_found.append(indicator)
            
            if indicators_found:
                threats.append({
                    'name': threat_name,
                    'severity': threat_info['severity'],
                    'indicators': indicators_found,
                    'mitigation': threat_info['mitigation'],
                    'confidence': len(indicators_found) / len(threat_info['indicators'])
                })
        
        return threats
    
    def _assess_vulnerabilities(self, security_data: Dict) -> List[Dict]:
        """Assess system vulnerabilities."""
        vulnerabilities = []
        
        # Check for common vulnerabilities
        if security_data.get('old_software_version', False):
            vulnerabilities.append({
                'name': 'outdated_software',
                'description': 'System running outdated software versions',
                'severity': 'high',
                'recommendation': 'update_to_latest_version'
            })
        
        if security_data.get('missing_encryption', False):
            vulnerabilities.append({
                'name': 'missing_encryption',
                'description': 'Data not encrypted at rest',
                'severity': 'critical',
                'recommendation': 'enable_encryption'
            })
        
        if security_data.get('weak_authentication', False):
            vulnerabilities.append({
                'name': 'weak_authentication',
                'description': 'Weak authentication mechanisms in place',
                'severity': 'high',
                'recommendation': 'implement_mfa'
            })
        
        if security_data.get('no_logging', False):
            vulnerabilities.append({
                'name': 'missing_audit_logs',
                'description': 'No audit logging implemented',
                'severity': 'medium',
                'recommendation': 'enable_audit_logging'
            })
        
        if security_data.get('open_ports', False):
            vulnerabilities.append({
                'name': 'open_ports',
                'description': 'Unnecessary open ports detected',
                'severity': 'medium',
                'recommendation': 'close_unused_ports'
            })
        
        return vulnerabilities
    
    def _calculate_security_posture(self, threats: List, vulnerabilities: List) -> Dict:
        """Calculate overall security posture score."""
        # Base score starts at 100
        score = 100
        
        # Deduct for threats
        threat_severity_weights = {
            'critical': 20,
            'high': 15,
            'medium': 10,
            'low': 5
        }
        
        for threat in threats:
            severity = threat.get('severity', 'low')
            deduction = threat_severity_weights.get(severity, 5) * threat.get('confidence', 0.5)
            score -= deduction
        
        # Deduct for vulnerabilities
        vuln_severity_weights = {
            'critical': 25,
            'high': 15,
            'medium': 10,
            'low': 5
        }
        
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'low')
            score -= vuln_severity_weights.get(severity, 5)
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        # Determine posture level
        if score >= 80:
            level = 'strong'
        elif score >= 60:
            level = 'good'
        elif score >= 40:
            level = 'moderate'
        elif score >= 20:
            level = 'weak'
        else:
            level = 'critical'
        
        return {
            'score': round(score, 1),
            'level': level,
            'threats_impact': 100 - score,
            'vulnerabilities_count': len(vulnerabilities),
            'threats_count': len(threats)
        }
    
    def _generate_security_recommendations(self, threats: List, vulnerabilities: List) -> List[Dict]:
        """Generate security recommendations."""
        recommendations = []
        
        # Recommendations for threats
        for threat in threats:
            recommendations.append({
                'type': 'threat_mitigation',
                'priority': threat.get('severity', 'medium'),
                'action': threat.get('mitigation', 'monitor'),
                'description': f"Mitigate {threat.get('name')} threat",
                'details': threat
            })
        
        # Recommendations for vulnerabilities
        for vuln in vulnerabilities:
            recommendations.append({
                'type': 'vulnerability_fix',
                'priority': vuln.get('severity', 'medium'),
                'action': vuln.get('recommendation', 'fix'),
                'description': f"Fix {vuln.get('name')} vulnerability",
                'details': vuln
            })
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return recommendations
    
    def _generate_security_summary(self, threats: List, vulnerabilities: List, posture: Dict) -> str:
        """Generate security analysis summary."""
        summary = f"""
        Security Analysis Summary:
        - Security Score: {posture.get('score', 0)}/100 ({posture.get('level', 'unknown')})
        - Threats Identified: {len(threats)}
        - Vulnerabilities Found: {len(vulnerabilities)}
        - Top Threat: {threats[0].get('name') if threats else 'None'}
        - Top Vulnerability: {vulnerabilities[0].get('name') if vulnerabilities else 'None'}
        
        Recommendations: {len(self._generate_security_recommendations(threats, vulnerabilities))} actions recommended.
        """
        return summary.strip()
    
    async def get_security_status(self) -> Dict:
        """Get overall security status."""
        if not self.security_findings:
            return {
                'status': 'unknown',
                'message': 'No security analysis performed yet'
            }
        
        latest = self.security_findings[-1]
        return {
            'status': 'analyzed',
            'last_analysis': latest.get('timestamp'),
            'security_score': latest.get('security_score', 0),
            'threats_found': latest.get('threats_identified', 0)
        }
    
    def clear_findings(self):
        """Clear security findings."""
        self.security_findings = []
        logger.info("🧹 Security findings cleared")