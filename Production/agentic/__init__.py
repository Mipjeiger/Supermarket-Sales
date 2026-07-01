"""
Agentic Framework for Enterprise Security & Fraud Detection
"""

from agentic.agents.fraud_detector import FraudDetectionAgent
from agentic.agents.abuse_investigator import AbuseInvestigationAgent
from agentic.agents.security_analyst import SecurityAnalystAgent
from agentic.evaluators.llm_as_judge import LLMAsJudge
from agentic.tools.data_retrieval import DataRetrievalTool
from agentic.tools.pattern_analysis import PatternAnalysisTool
from agentic.tools.risk_scoring import RiskScoringTool
from agentic.tools.report_generator import ReportGeneratorTool

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