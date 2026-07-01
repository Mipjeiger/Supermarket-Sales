from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
import json
import logging
from datetime import datetime

from agentic.tools.data_retrieval import DataRetrievalTool
from agentic.tools.pattern_analysis import PatternAnalysisTool
from agentic.tools.risk_scoring import RiskScoringTool
from agentic.tools.report_generator import ReportGeneratorTool
from agentic.evaluators.llm_as_judge import LLMAsJudge