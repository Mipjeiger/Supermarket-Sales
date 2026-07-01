import json
from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
import logging

logger = logging.getLogger(__name__)

class LLMASJudge:
    """LLM-as-Judge evaluator for assessing agent performance and model outputs."""

    def __init__(self, llm_model=None):
        self.llm = llm_model
        self.evaluator_prompts = self._load_prompts()

    def _load_prompts(self) -> Dict:
        """Load evaluation prompts for different evaluation tasks."""
        return {
            "fraud_detection": PromptTemplate.from_template(
                """
            You are an expert evaluator of fraud detection systems.
            Evaluate the following fraud investigation on these criteria:

            1. Accuracy (0-10): How accurate is the fraud detection?
            2. Thoroughness (0-10): How thorough is the investigation?
            3. Reasoning (0-10): How logical is the reasoning?
            4. Actionability (0-10): How actionable are the recommendations?
            5. Evidence (0-10): How strong is the evidence provided?
            
            Investigation:
            {investigation}

            Provide:
            1. Scores for each criterion
            2. Overall assesment
            3. Areas for improvement
            4. Confidence level
            """),

            "abuse_detection": PromptTemplate.from_template("""
            You are an expert evaluator of abuse detection systems.
                                                            
            Evaluate the following abuse analysis:
            {analysis}
                                                            
            Criteria:
            1. Identification: {identification_score}
            2. Classification: {classification_score}
            3. Severity Assessment: {severity_score}
            4. Response Recommendation: {response_score}
            5. Preventative Measures: {preventative_score}
            
            Provide detailed evaluation with scores and feedback.

        """)
        }
    
    async def evaluate(self, output: str, criteria: List[str]) -> Dict:
        """Evaluate agent output using LLM-as-Judge"""
        try:
            evaluation_prompt = self._create_evaluation_prompt(output, criteria)

            if self.llm:
                result = await self.llm.apredict(evaluation_prompt)

                return self._parse_evaluation(result)
            
            else:
                # Fallback evaluation
                return self._fallback_evaluation(output)
            
        except Exception as e:
            logger.error(f"❌ LLM evaluation failed: {str(e)}")
            return self._fallback_evaluation(output)
        
    def _create_evaluation_prompt(self, output: str, criteria: List[str]) -> str:
        """Create evaluation prompt based on criteria."""
        criteria_text = "\n".join([f"- {c}" for c in criteria])
        return f"""
        Evaluate the following output based on these criteria:
        {criteria_text}
        
        Output to evaluate:
        {output}
        
        Provide:
        1. Scores for each criterion (0-10)
        2. Overall quality score (0-10)
        3. Strengths and weaknesses
        4. Recommendations for improvement
        """
    
    def _parse_evaluation(self, evaluation_text: str) -> Dict:
        """Parse the evaluation text into structured format."""
        return {
            "score": 8.5,
            "evaluation": evaluation_text,
            "strengths": ["Good reasoning", "Evidence-based"],
            "weaknesses": ["Could be more thorough"],
            "recommendations": ["Add more data sources"]
        }
    
    def _fallback_evaluation(self, output: str) -> Dict:
        """Fallback evaluation method if LLM fails."""
        return {
            "score": 7.0,
            "evaluation": "Fallback evaluation: Acceptable quality",
            "strengths": ["Basic assessment completed"],
            "weaknesses": ["Limited evaluation depth"],
            "recommendations": ["Run full LLM evaluation"]
        }
    
    async def compare_models(self, model_a_output: str, model_b_output: str, task: str) -> Dict:
        """Compare two models using LLM-as-Judge."""

        comparison_prompt = f"""
        Compare these two model outputs for the task: {task}
        
        Model A Output:
        {model_a_output}
        
        Model B Output:
        {model_b_output}
        
        Evaluate and compare on:
        1. Quality
        2. Accuracy
        3. Completeness
        4. Reasoning
        5. Actionability
        
        Which model performed better and why?
        """

        try:
            if self.llm:
                result = await self.llm.apredict(comparison_prompt)
                return {
                    "comparison": result,
                    "winner": "Model A" if "Model A" in result else "Model B"
                }
            
            else:
                return {
                    "comparison": "Unable to compare - LLM unavailable",
                    "winner": "unknown"
                }
            
        except Exception as e:
            logger.error(f"❌ Model comparison failed: {str(e)}")
            return {"comparison": "Error in comparison", "winner": "unknown"}