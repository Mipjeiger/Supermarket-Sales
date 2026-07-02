"""
Agentic Framework for Enterprise Security & Fraud Detection
"""

from app.agentic.agents.fraud_detector import FraudDetectionAgent
from app.agentic.agents.abuse_investigator import AbuseInvestigationAgent
from app.agentic.agents.security_analyst import SecurityAnalystAgent
from app.agentic.evaluators.llm_as_judge import LLMAsJudge
from app.agentic.tools.data_retrieval import DataRetrievalTool
from app.agentic.tools.pattern_analysis import PatternAnalysisTool
from app.agentic.tools.risk_scoring import RiskScoringTool
from app.agentic.tools.report_generator import ReportGeneratorTool

__version__ = "1.0.0"

__all__ = [
    'FraudDetectionAgent',
    'AbuseInvestigationAgent',
    'SecurityAnalystAgent',
    'LLMAsJudge',
    'DataRetrievalTool',
    'PatternAnalysisTool',
    'RiskScoringTool',
    'ReportGeneratorTool'
]