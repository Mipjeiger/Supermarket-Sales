import logging
import requests
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMProvider:
    """Orhecstrates serverless LLM token processing with automated rate-limit fallbacks."""

    def __init__(self):
        self.hf_key = settings.HUGGINGFACE_API_KEY
        self.groq_key = settings.GROQ_API_KEY

        # Base Endpoints (Utilizing OpenAI-compatible chat architectures)
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.hf_base_url = "https://api-inference.huggingface.co/models"

    def query_huggingface_chat(
        self,
        prompt: str,
        system_instruction: str,
        model_repo: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Executes text generation against Hugging Face Serverless Architecture endpoints."""
        if not self.hf_key:
            raise ValueError("Hugging Face API key is not configured. Please set HUGGINGFACE_API_KEY in your environment.")
        
        endpoint = f"{self.hf_base_url}/{model_repo}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_repo,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=15)

        # Explicity buble up rate limit errors or bad requests
        if response.status_code == 429:
            raise ResourceWarning("Hugging Face API rate limit exceeded. Please retry after some time.")
        elif response.status_code != 200:
            raise RuntimeError(f"Hugging Face API request failed with status {response.status_code}: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]
    
    def query_groq_chat(
        self,
        prompt: str,
        system_instruction: str,
        model_name: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Executes ultra-fast failover token generation using Groq LPU acceleration layers."""
        if not self.groq_key:
            raise ValueError("Missing GROQ_API_KEY configuration.")

        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(self.groq_url, headers=headers, json=payload, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"Groq API request failed with status {response.status_code}: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]
    
    def generate_grounded_text(
        self,
        prompt: str,
        system_instruction: str,
        hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        groq_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 256
    ) -> str:
        """
        Primary interface method for compound pipelines. Tries Hugging Face first. 
        Automatically shifts execution execution context to Groq upon token or network depletion.
        """
        try:
            logger.info(f"Sending context payload to Hugging Face serverless layer model: {hf_model}")
            return self.query_huggingface_chat(
                prompt=prompt,
                system_instruction=system_instruction,
                model_repo=hf_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        except (ResourceWarning, Exception) as e:
            logger.warning(f"⚠️ Hugging Face pipeline restricted or exhausted. Error: {str(e)}")
            logger.info(f"🔄 Executing failover sequence. Routing directly to Groq hardware clusters ({groq_model})...")

            try:
                return self.query_groq(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model_name=groq_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            except Exception as critical_error:
                logger.error(f"❌ Complete pipeline blackout. Both LLM providers exhausted: {str(critical_error)}")
                raise RuntimeError(f"🚨 System blacked out. Provider fallback failure chain: {str(critical_error)}")
            
# Singleton initialization for global LLM provider access
llm_provider = LLMProvider()