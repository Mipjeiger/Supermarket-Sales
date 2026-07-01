"""
LLM-as-Judge Evaluator
Evaluates agent outputs, model responses, and investigation quality using LLM.
Provides consistent, scalable evaluation for:
- Agent performance
- Investigation quality
- Model outputs
- System responses
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import LLMChain
import numpy as np

logger = logging.getLogger(__name__)

class LLMAsJudge:
    """
    Evaluator that uses LLM to judge quality, accuracy, and performance.
    Supports multiple evaluation criteria and provides detailed feedback.
    """
    
    def __init__(self, llm_model=None):
        self.llm = llm_model
        self.evaluation_history = []
        self.evaluation_templates = self._load_evaluation_templates()
        self.quality_thresholds = {
            'excellent': 9.0,
            'good': 7.0,
            'acceptable': 5.0,
            'poor': 3.0,
            'failing': 0.0
        }
    
    def _load_evaluation_templates(self) -> Dict:
        """Load evaluation prompt templates for different criteria."""
        return {
            'agent_performance': PromptTemplate.from_template("""
            You are an expert evaluator of AI agent performance.
            
            Evaluate the agent's performance based on these criteria:
            
            1. Task Completion (0-10): Did the agent complete the assigned task?
            2. Accuracy (0-10): How accurate were the agent's outputs?
            3. Efficiency (0-10): How efficiently did the agent work?
            4. Reasoning (0-10): How logical and thorough was the reasoning?
            5. Actionability (0-10): How actionable are the recommendations?
            
            Agent Output:
            {output}
            
            Task Context:
            {context}
            
            Provide:
            1. Scores for each criterion
            2. Overall performance score (0-10)
            3. Key strengths identified
            4. Areas for improvement
            5. Confidence in evaluation (0-10)
            
            Format your response as JSON.
            """),
            
            'investigation_quality': PromptTemplate.from_template("""
            You are an expert investigator evaluating the quality of an investigation.
            
            Evaluate the investigation based on:
            
            1. Thoroughness (0-10): How comprehensive was the investigation?
            2. Evidence Quality (0-10): How strong and relevant is the evidence?
            3. Reasoning (0-10): How logical is the investigative reasoning?
            4. Conclusions (0-10): How well-supported are the conclusions?
            5. Actionability (0-10): How actionable are the findings?
            
            Investigation Results:
            {output}
            
            Investigation Context:
            {context}
            
            Provide:
            1. Scores for each criterion
            2. Overall quality score (0-10)
            3. Strongest aspects
            4. Weakest aspects
            5. Recommendations for improvement
            6. Confidence in evaluation (0-10)
            
            Format your response as JSON.
            """),
            
            'model_output': PromptTemplate.from_template("""
            You are an expert evaluating model outputs for quality and accuracy.
            
            Evaluate the model output based on:
            
            1. Accuracy (0-10): How accurate is the output?
            2. Completeness (0-10): How complete is the response?
            3. Clarity (0-10): How clear and understandable is the output?
            4. Relevance (0-10): How relevant is it to the query?
            5. Utility (0-10): How useful is the information?
            
            Model Output:
            {output}
            
            Query/Context:
            {context}
            
            Provide:
            1. Scores for each criterion
            2. Overall quality score (0-10)
            3. Strengths
            4. Weaknesses
            5. Suggestions for improvement
            6. Confidence in evaluation (0-10)
            
            Format your response as JSON.
            """),
            
            'abuse_detection': PromptTemplate.from_template("""
            You are an expert evaluator of abuse detection systems.
            
            Evaluate the abuse detection based on:
            
            1. Detection Accuracy (0-10): How accurate was the abuse detection?
            2. Classification (0-10): How well was the abuse type classified?
            3. Severity Assessment (0-10): How accurate is the severity assessment?
            4. Response Recommendation (0-10): How appropriate are the recommendations?
            5. Evidence Quality (0-10): How strong is the evidence provided?
            
            Detection Results:
            {output}
            
            Context:
            {context}
            
            Provide:
            1. Scores for each criterion
            2. Overall detection quality score (0-10)
            3. Key findings
            4. Areas for improvement
            5. Confidence in evaluation (0-10)
            
            Format your response as JSON.
            """),
            
            'fraud_detection': PromptTemplate.from_template("""
            You are an expert evaluator of fraud detection systems.
            
            Evaluate the fraud detection based on:
            
            1. Detection Accuracy (0-10): How accurately was fraud detected?
            2. Risk Assessment (0-10): How accurate is the risk assessment?
            3. Investigation Quality (0-10): How thorough was the investigation?
            4. Recommendations (0-10): How appropriate are the recommendations?
            5. Evidence Quality (0-10): How strong is the evidence?
            
            Detection Results:
            {output}
            
            Context:
            {context}
            
            Provide:
            1. Scores for each criterion
            2. Overall quality score (0-10)
            3. Strengths
            4. Weaknesses
            5. Recommendations for improvement
            6. Confidence in evaluation (0-10)
            
            Format your response as JSON.
            """),
            
            'security_analysis': PromptTemplate.from_template("""
            You are an expert security analyst evaluating security analysis quality.
            
            Evaluate the security analysis based on:
            
            1. Threat Identification (0-10): How well were threats identified?
            2. Vulnerability Assessment (0-10): How accurate is vulnerability assessment?
            3. Risk Analysis (0-10): How comprehensive is the risk analysis?
            4. Recommendations (0-10): How effective are the recommendations?
            5. Actionability (0-10): How actionable are the findings?
            
            Analysis Results:
            {output}
            
            Context:
            {context}
            
            Provide:
            1. Scores for each criterion
            2. Overall security analysis score (0-10)
            3. Key insights
            4. Areas needing attention
            5. Confidence in evaluation (0-10)
            
            Format your response as JSON.
            """)
        }
    
    async def evaluate(self, output: Union[str, Dict], criteria: List[str], context: Optional[Dict] = None) -> Dict:
        """
        Evaluate output using LLM-as-Judge.
        
        Args:
            output: The output to evaluate (string or dict)
            criteria: List of evaluation criteria
            context: Optional context for evaluation
            
        Returns:
            Dict with evaluation results
        """
        try:
            # Determine evaluation type based on criteria
            eval_type = self._determine_evaluation_type(criteria)
            
            # Get appropriate template
            template = self.evaluation_templates.get(eval_type)
            if not template:
                template = self.evaluation_templates['agent_performance']
            
            # Prepare evaluation prompt
            output_str = json.dumps(output, indent=2) if isinstance(output, dict) else str(output)
            context_str = json.dumps(context, indent=2) if context else "No additional context provided"
            
            prompt = template.format(
                output=output_str,
                context=context_str
            )
            
            # Get evaluation from LLM
            if self.llm:
                try:
                    chain = LLMChain(llm=self.llm, prompt=template)
                    eval_result = await chain.apredict(
                        output=output_str,
                        context=context_str
                    )
                except Exception:
                    eval_result = await self.llm.apredict(prompt)
            else:
                # Fallback evaluation without LLM
                eval_result = self._fallback_evaluation(output, criteria)
            
            # Parse evaluation
            parsed_result = self._parse_evaluation(eval_result, eval_type)
            
            # Store evaluation history
            self.evaluation_history.append({
                'timestamp': datetime.now().isoformat(),
                'eval_type': eval_type,
                'criteria': criteria,
                'results': parsed_result
            })
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"❌ LLM evaluation failed: {str(e)}")
            return self._fallback_evaluation(output, criteria)
    
    def _determine_evaluation_type(self, criteria: List[str]) -> str:
        """Determine evaluation type based on criteria."""
        criteria_str = ' '.join(criteria).lower()
        
        if any(word in criteria_str for word in ['abuse', 'detection', 'abuse_detection']):
            return 'abuse_detection'
        elif any(word in criteria_str for word in ['fraud', 'fraud_detection']):
            return 'fraud_detection'
        elif any(word in criteria_str for word in ['security', 'threat', 'vulnerability']):
            return 'security_analysis'
        elif any(word in criteria_str for word in ['investigation', 'investigate']):
            return 'investigation_quality'
        elif any(word in criteria_str for word in ['model', 'output', 'response']):
            return 'model_output'
        else:
            return 'agent_performance'
    
    def _parse_evaluation(self, eval_result: str, eval_type: str) -> Dict:
        """Parse evaluation result from LLM."""
        try:
            # Try to parse as JSON
            if '{' in eval_result and '}' in eval_result:
                start_idx = eval_result.find('{')
                end_idx = eval_result.rfind('}') + 1
                json_str = eval_result[start_idx:end_idx]
                return json.loads(json_str)
            
            # Parse text-based evaluation
            return self._parse_text_evaluation(eval_result, eval_type)
            
        except Exception as e:
            logger.error(f"❌ Failed to parse evaluation: {str(e)}")
            return {
                'overall_score': 7.0,
                'raw_evaluation': eval_result,
                'confidence': 6.0,
                'parse_error': str(e)
            }
    
    def _parse_text_evaluation(self, text: str, eval_type: str) -> Dict:
        """Parse text-based evaluation."""
        scores = {}
        score_lines = []
        
        # Extract scores from text
        import re
        score_pattern = r'(\d+\.?\d*)\s*(?:/\s*10)?'
        
        lines = text.split('\n')
        for line in lines:
            if ':' in line and re.search(score_pattern, line):
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value_match = re.search(score_pattern, parts[1])
                    if value_match:
                        try:
                            scores[key] = float(value_match.group(1))
                        except:
                            pass
                    score_lines.append(line.strip())
        
        # Calculate overall score if not explicitly provided
        overall_score = scores.get('overall', 0)
        if overall_score == 0 and scores:
            overall_score = sum(scores.values()) / len(scores)
        
        return {
            'scores': scores,
            'overall_score': round(overall_score, 1),
            'raw_evaluation': text,
            'score_lines': score_lines,
            'confidence': scores.get('confidence', 7.0) if scores else 7.0,
            'evaluation_type': eval_type,
            'timestamp': datetime.now().isoformat()
        }
    
    def _fallback_evaluation(self, output: Union[str, Dict], criteria: List[str]) -> Dict:
        """
        Fallback evaluation when LLM is unavailable.
        Provides basic evaluation based on heuristics.
        """
        logger.warning("⚠️ Using fallback evaluation (LLM not available)")
        
        # Basic heuristic scoring
        output_str = str(output)
        score = 5.0  # Starting neutral score
        
        # Positive indicators
        positive_indicators = [
            'success', 'completed', 'found', 'detected', 'identified',
            'recommended', 'analyzed', 'investigated', 'verified'
        ]
        for indicator in positive_indicators:
            if indicator in output_str.lower():
                score += 0.5
        
        # Negative indicators
        negative_indicators = [
            'error', 'failed', 'unknown', 'unable', 'cannot',
            'missing', 'invalid', 'exception', 'null'
        ]
        for indicator in negative_indicators:
            if indicator in output_str.lower():
                score -= 0.5
        
        # Clamp score
        score = max(0, min(10, score))
        
        return {
            'overall_score': round(score, 1),
            'scores': {
                'accuracy': min(10, score + 0.5),
                'completeness': min(10, score - 0.5),
                'clarity': min(10, score + 0.5),
                'relevance': min(10, score - 0.5),
                'utility': min(10, score + 0.5)
            },
            'confidence': 3.0,  # Low confidence for fallback
            'evaluation_method': 'fallback',
            'fallback': True,
            'timestamp': datetime.now().isoformat()
        }
    
    async def compare_responses(self, response_a: str, response_b: str, criteria: List[str]) -> Dict:
        """
        Compare two responses and determine which is better.
        
        Args:
            response_a: First response to compare
            response_b: Second response to compare
            criteria: Criteria for comparison
            
        Returns:
            Dict with comparison results
        """
        try:
            comparison_prompt = PromptTemplate.from_template("""
            Compare these two responses and determine which is better.
            
            Criteria: {criteria}
            
            Response A:
            {response_a}
            
            Response B:
            {response_b}
            
            Provide:
            1. Which response is better (A or B)
            2. Scores for each response on the criteria
            3. Reasons for the decision
            4. Key differences
            
            Format as JSON.
            """)
            
            if self.llm:
                try:
                    chain = LLMChain(llm=self.llm, prompt=comparison_prompt)
                    comparison_result = await chain.apredict(
                        criteria=", ".join(criteria),
                        response_a=response_a,
                        response_b=response_b
                    )
                except Exception:
                    comparison_result = await self.llm.apredict(
                        comparison_prompt.format(
                            criteria=", ".join(criteria),
                            response_a=response_a,
                            response_b=response_b
                        )
                    )
                parsed = self._parse_comparison(comparison_result)
            else:
                parsed = self._fallback_comparison(response_a, response_b)
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Response comparison failed: {str(e)}")
            return {
                'better': 'unknown',
                'error': str(e)
            }
    
    def _parse_comparison(self, comparison_text: str) -> Dict:
        """Parse comparison result from LLM."""
        try:
            if '{' in comparison_text and '}' in comparison_text:
                start_idx = comparison_text.find('{')
                end_idx = comparison_text.rfind('}') + 1
                json_str = comparison_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Simple text parsing
                lines = comparison_text.split('\n')
                better = None
                scores_a = {}
                scores_b = {}
                
                for line in lines:
                    if 'better' in line.lower() and 'A' in line:
                        better = 'A'
                    elif 'better' in line.lower() and 'B' in line:
                        better = 'B'
                    elif 'A' in line and ':' in line:
                        parts = line.split(':')
                        if len(parts) == 2:
                            try:
                                scores_a[parts[0].strip()] = float(parts[1].strip())
                            except:
                                pass
                    elif 'B' in line and ':' in line:
                        parts = line.split(':')
                        if len(parts) == 2:
                            try:
                                scores_b[parts[0].strip()] = float(parts[1].strip())
                            except:
                                pass
                
                return {
                    'winner': better,
                    'scores_a': scores_a,
                    'scores_b': scores_b,
                    'raw_comparison': comparison_text
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to parse comparison: {str(e)}")
            return {
                'winner': 'unknown',
                'raw_comparison': comparison_text
            }
    
    def _fallback_comparison(self, response_a: str, response_b: str) -> Dict:
        """Fallback comparison when LLM is unavailable."""
        # Simple heuristic: compare lengths and complexity
        score_a = len(response_a.split())
        score_b = len(response_b.split())
        
        # More words doesn't always mean better, but it's a fallback
        better = 'A' if score_a > score_b else 'B' if score_b > score_a else 'tie'
        
        return {
            'winner': better,
            'scores_a': {'length_score': min(10, score_a / 10)},
            'scores_b': {'length_score': min(10, score_b / 10)},
            'method': 'fallback',
            'message': 'Used fallback comparison based on response length'
        }
    
    def get_evaluation_statistics(self, eval_type: Optional[str] = None) -> Dict:
        """
        Get statistics from evaluation history.
        
        Args:
            eval_type: Optional filter by evaluation type
            
        Returns:
            Dict with evaluation statistics
        """
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        # Filter evaluations
        evals = self.evaluation_history
        if eval_type:
            evals = [e for e in evals if e.get('eval_type') == eval_type]
        
        if not evals:
            return {'message': f'No evaluations of type {eval_type} found'}
        
        # Calculate statistics
        scores = [e['results'].get('overall_score', 0) for e in evals]
        avg_score = np.mean(scores) if scores else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        std_score = np.std(scores) if len(scores) > 1 else 0
        
        return {
            'total_evaluations': len(evals),
            'avg_score': round(avg_score, 2),
            'min_score': round(min_score, 2),
            'max_score': round(max_score, 2),
            'std_deviation': round(std_score, 2),
            'eval_types': [e.get('eval_type') for e in evals],
            'latest_evaluation': evals[-1] if evals else None
        }
    
    def get_quality_level(self, score: float) -> str:
        """
        Get quality level based on score.
        
        Args:
            score: Quality score (0-10)
            
        Returns:
            Quality level string
        """
        if score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif score >= self.quality_thresholds['good']:
            return 'good'
        elif score >= self.quality_thresholds['acceptable']:
            return 'acceptable'
        elif score >= self.quality_thresholds['poor']:
            return 'poor'
        else:
            return 'failing'
    
    def clear_history(self):
        """Clear evaluation history."""
        self.evaluation_history = []
        logger.info("🧹 Evaluation history cleared")