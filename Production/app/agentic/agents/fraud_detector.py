from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
import json
import logging
from datetime import datetime
from typing import Dict, List

from app.agentic.tools.data_retrieval import DataRetrievalTool
from app.agentic.tools.pattern_analysis import PatternAnalysisTool
from app.agentic.tools.risk_scoring import RiskScoringTool
from app.agentic.tools.report_generator import ReportGeneratorTool
from app.agentic.evaluators.llm_as_judge import LLMAsJudge

logger = logging.getLogger(__name__)

class FraudDetectionAgent:
    """Multi-step agentic system for fraud detection and investigation.
    Combines LLM reasoning with specialized tools.
    """
    
    def __init__(self, llm_model, vector_store):
        self.llm = llm_model
        self.vector_store = vector_store
        self.tools = self._initialize_tools()
        self.agent = self._create_agent()
        self.llm_judge = LLMAsJudge()
        self.investigation_history = []  # Store investigation steps and results

    def _initialize_tools(self) -> List[Tool]:
        """Initialize specialized tools for fraud detection."""
        return [
            Tool(
                name="DataRetrieval",
                func=DataRetrievalTool().run,
                description="Retrieves transaction data, user history, and patterns"
            ),
            Tool(
                name="PatternAnalysis",
                func=PatternAnalysisTool().run,
                description="Analyzes patterns in transaction data for anomalies"
            ),
            Tool(
                name="RiskScoring",
                func=RiskScoringTool().run,
                description="Calculates risk scores using ML models"
            ),
            Tool(
                name="ReportGenerator",
                func=ReportGeneratorTool().run,
                description="Generates detailed fraud investigation reports"
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent with React framework"""
        prompt = PromptTemplate.from_template("""
        You are an expert fraud detection agent for a large supermarket chain.
                                              
        Available tools:
        {tools}
                                              
        Tool names: {tool_names}
                                              
        Objective: Detect and investigate potential fraud cases.
                                              
        Follow this process:
        1. Analyze the transaction data
        2. Identify suspicious patterns
        3. Investigate using available tools
        4. Make a judgment
        5. Generate a detailed report
                                              
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
    
    async def investigate_fraud(self, transaction_data: Dict) -> Dict:
        """Conduct a multi-step fraud investigation"""
        try:
            # 1. Initial screening
            logger.info("🔍 Starting fraud investigation...")

            # 2. Run agentic investigation
            investigation_result = await self.agent.arun(input=json.dumps(transaction_data))

            # 3. LLM-as-Judge evaluation
            quality_score = await self.llm_judge.evaluate(
                investigation_result,
                criteria=["thoroughness", "reasoning", "evidence", "actionability"]
            )

            # 4. Store investigation history
            self.investigation_history.append({
                "timestamp": datetime.now().isoformat(),
                "transaction_id": transaction_data.get("transaction_id"),
                "result": investigation_result,
                "quality_score": quality_score
            })

            # 5. Generate report
            report = await self._generate_report(
                transaction_data, investigation_result, quality_score
            )

            return {
                "status": "completed",
                "investigation_result": investigation_result,
                "quality_score": quality_score,
                "report": report,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Error during fraud investigation: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
    async def _generate_report(self, data: Dict, result: str, quality: float) -> Dict:
        """Generate comprehensive investigation report."""
        return {
            "investigation_id": f"INV-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "transaction_id": data.get("transaction_id"),
            "fraud_score": result.get("risk_score", 0),
            "findings": result.get("findings", []),
            "recommendations": result.get("recommendations", []),
            "evidence": result.get("evidence", []),
            "confidence": quality,
            "timestamp": datetime.now().isoformat(),
            "investigator": "FraudDetectionAgent_v1"
        }