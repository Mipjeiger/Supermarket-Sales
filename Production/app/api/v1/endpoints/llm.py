import os
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter,  HTTPException, status
from pydantic import BaseModel, Field
from app.services.sales_llm_investigation import sales_investigation_request
from app.services.fraud_llm_investigation import fraud_investigation_request
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
            temperature=0.2
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
            "return_full_text": False
        }
    }