import os
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.services.sales_llm_investigation import SalesInvestigationRequest
from app.services.fraud_llm_investigation import FraudInvestigationRequest
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parents[4]
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

router = APIRouter(prefix="/llm", tags=["LLM Intelligence Gateway"])

# ================================================================
# 1. Pydantic Models for LLM Investigation Requests
# ===============================================================
class InvestigationResponse(BaseModel):
    """Unified response schema for LLM-driven investigation insights."""
    status: str = Field(..., example="success")
    investigation_type: str = Field(..., example="fraud")
    provider_used: str = Field(..., example="Groq (llama-3.3-70b-versatile)")
    target_entity: str = Field(..., example="John Lee")
    investigation_date: str = Field(..., example="2026-07-25")
    metrics_summary: Dict[str, Any]
    analysis_summary: str

# ===============================================================
# 2. Async Provider Invocation Handlers
# =============================================================
async def _invoke_groq(prompt: str) -> Optional[str]:
    """Invoke Groq LLM asynchronously"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    model_name = "llama-3.3-70b-versatile"

    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.2,
        )
        return completion.choices[0].message.content

    except Exception as e:
        print(f"❌ Error invoking Groq LLM: {str(e)}")
        return None

async def _invoke_huggingface(prompt: str) -> Optional[str]:
    """Invoke HuggingFace LLM asynchronously"""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return None

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 128,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", str(result))
                elif isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"]
                return str(result)
            else:
                print(f"⚠️ HuggingFace API Error {response.status_code}: {response.text}")
                return None
    except Exception as e:
        print(f"⚠️ HuggingFace Invocation Error: {str(e)}")
        return None

# ==============================================================================
# 3. Rule-Based Fallback Generators
# ==============================================================================

def _fraud_rule_fallback(req: FraudInvestigationRequest) -> str:
    """Rule-based fallback summary when LLM services are offline."""
    return (
        f"1. Pattern Summary: Customer '{req.customer_name}' order ({req.order_id}) in category '{req.category}/{req.sub_category}' "
        f"shows risk level {req.risk_level} with profit margin {req.profit_margin:.2f}.\n"
        f"2. Likely Root Cause: Shipping latency ({req.shipping_days} days) combined with discount {req.discount:.0%} resulted in profit degradation.\n"
        f"3. Immediate Action Required: Perform standard manual verification if abuse score ({req.abuse_score:.2f}) exceeds 0.50.\n"
        f"4. Long-term Prevention: Review shipping SLAs and discount capping rules for {req.category}."
    )

def _sales_rule_fallback(req: SalesInvestigationRequest) -> str:
    """Rule-based fallback summary for sales variance analysis."""
    return (
        f"1. Performance Summary: Sales for target '{req.customer_target}' show variance of {req.sales_difference:.2f} "
        f"(Actual: {req.actual_sales:.2f} vs Predicted: {req.predicted_sales:.2f}). Overall trend is {req.sales_trend}.\n"
        f"2. Drivers & Factors: Discount applied at {req.discount:.0%} across {req.quantity} units with {req.shipping_days} days shipping delay.\n"
        f"3. Strategic Action: Recalibrate demand forecasting for sub-category '{req.sub_category or 'General'}'.\n"
        f"4. Revenue Optimization: Adjust promo pricing and address fulfillment lag to close variance gap."
    )

# ==============================================================================
# 4. FastAPI Endpoints
# ==============================================================================

@router.post(
    "/investigate-fraud",
    response_model=InvestigationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate LLM-driven Fraud Audit Briefing based on Supermarket DB features",
)
async def investigate_fraud_endpoint(payload: FraudInvestigationRequest):
    """
    Evaluates transactional anomalies, negative margins, and shipping delays 
    using enriched supermarket database features.
    """
    try:
        formatted_context = (
            f"Order ID: {payload.order_id}\n"
            f"Customer Name: {payload.customer_name}\n"
            f"Category/Sub-Category: {payload.category} / {payload.sub_category}\n"
            f"Sales: {payload.sales:,.2f} | Quantity: {payload.quantity} | Unit Price: {payload.unit_price:,.2f}\n"
            f"Discount: {payload.discount:.2%} | Profit: {payload.profit:,.2f} | Profit Margin: {payload.profit_margin:.2%}\n"
            f"Shipping Days: {payload.shipping_days} | Order Priority: {payload.order_priority}\n"
            f"Fraud Flag: {payload.fraud_flag} | Risk Level: {payload.risk_level} | Abuse Score: {payload.abuse_score:.2f}\n"
            f"Derived Context: {payload.custom_context}\n"
        )
        if payload.custom_context:
            formatted_context += f"Custom Context: {payload.custom_context}\n"

        prompt = f"""
        You are an enterprise MLOps & Fraud Security Agent auditing supermarket transactional records.

        Database Context Metrics:
        {formatted_context}

        Provide a concise, highly analytical 4-bullet investigation briefing:
        1. Pattern Summary
        2. Likely Root Cause
        3. Immediate Action Required
        4. Long-term Prevention
        """

        # Step 1: Try HuggingFace
        hf_res = await _invoke_huggingface(prompt)
        if hf_res:
            return InvestigationResponse(
                status="success",
                investigation_type="fraud",
                provider_used=f"HuggingFace ({os.getenv('HF_MODEL_NAME', 'meta-llama/Llama-3.2-3B-Instruct')})",
                target_entity=payload.customer_name,
                investigation_date=payload.investigation_date,
                metrics_summary=payload.dict(),
                analysis_summary=hf_res.strip(),
            )

        # Step 2: Fallback to Groq
        groq_res = await _invoke_groq(prompt)
        if groq_res:
            return InvestigationResponse(
                status="success",
                investigation_type="fraud",
                provider_used=f"Groq ({os.getenv('GROQ_MODEL_NAME', 'llama-3.3-70b-versatile')})",
                target_entity=payload.customer_name,
                investigation_date=payload.investigation_date,
                metrics_summary=payload.dict(),
                analysis_summary=groq_res.strip(),
            )

        # Step 3: Rule-based fallback
        return InvestigationResponse(
            status="fallback",
            investigation_type="fraud",
            provider_used="Rule-Engine (Offline Mode)",
            target_entity=payload.customer_name,
            investigation_date=payload.investigation_date,
            metrics_summary=payload.dict(),
            analysis_summary=_fraud_rule_fallback(payload),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fraud investigation execution failure: {str(e)}",
        )

@router.post(
    "/investigate-sales",
    response_model=InvestigationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate LLM-driven Sales Variance & Market Intelligence Briefing",
)
async def investigate_sales_endpoint(payload: SalesInvestigationRequest):
    """
    Analyzes sales variance (Actual vs Predicted), discounting impacts, 
    and regional product demand trends.
    """
    try:
        formatted_context = (
            f"Target Customer: {payload.customer_target}\n"
            f"Product: {payload.product_name} ({payload.category} / {payload.sub_category})\n"
            f"Market / Segment: {payload.market} / {payload.segment}\n"
            f"Predicted Sales: {payload.predicted_sales:,.2f} | Actual Sales: {payload.actual_sales:,.2f}\n"
            f"Variance Gap: {payload.sales_difference:,.2f} | Sales Trend: {payload.sales_trend}\n"
            f"Quantity Sold: {payload.quantity} | Discount: {payload.discount:.2%}\n"
            f"Shipping Days: {payload.shipping_days}\n"
        )
        if payload.custom_context:
            formatted_context += f"Custom Context: {payload.custom_context}\n"

        prompt = f"""
        You are a Senior Commercial Data Analyst investigating sales revenue variance and market trends.

        Context & Variance Metrics:
        {formatted_context}

        Provide a concise, executive-level 4-bullet investigation summary:
        1. Performance & Variance Summary
        2. Key Revenue Drivers & Root Cause
        3. Immediate Commercial Recommendation
        4. Long-term Sales & Demand Strategy
        """

        # Step 1: Try HuggingFace
        hf_res = await _invoke_huggingface(prompt)
        if hf_res:
            return InvestigationResponse(
                status="success",
                investigation_type="sales",
                provider_used=f"HuggingFace ({os.getenv('HF_MODEL_NAME', 'meta-llama/Llama-3.2-3B-Instruct')})",
                target_entity=payload.customer_target,
                investigation_date=payload.investigation_date,
                metrics_summary=payload.dict(),
                analysis_summary=hf_res.strip(),
            )

        # Step 2: Fallback to Groq
        groq_res = await _invoke_groq(prompt)
        if groq_res:
            return InvestigationResponse(
                status="success",
                investigation_type="sales",
                provider_used=f"Groq ({os.getenv('GROQ_MODEL_NAME', 'llama-3.3-70b-versatile')})",
                target_entity=payload.customer_target,
                investigation_date=payload.investigation_date,
                metrics_summary=payload.dict(),
                analysis_summary=groq_res.strip(),
            )
    
        # Step 3: Rule-based fallback
        return InvestigationResponse(
            status="fallback",
            investigation_type="sales",
            provider_used="Rule-Engine (Offline Mode)",
            target_entity=payload.customer_target,
            investigation_date=payload.investigation_date,
            metrics_summary=payload.dict(),
            analysis_summary=_sales_rule_fallback(payload),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sales investigation execution failure: {str(e)}",
        )