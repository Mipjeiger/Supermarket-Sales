import os
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEndpoint
from app.core.config import settings
from app.core.redis import redis_cache


class SecurityAgentEngine:
    """Bridge between Qdrant vector database and HuggingFace LLM for security analysis."""

    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
        )

        # Load fine-tuned HuggingFace LLM interface
        self.llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            max_new_tokens=256,
            temperature=0.1,
            huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        )

    async def chat_investigate(
        self, session_id: str, user_query: str, velocity_context: str
    ) -> str:
        """Processes complex user prompts with streaming context injection and Redis memory storage."""
        # Pull historical chat sequences from Redis cache layer
        memory_key = f"chat_history:{session_id}"
        raw_history = await redis_cache.client.get(memory_key)
        chat_history = raw_history if raw_history else "No prior context available."

        # Construct the prompt for the LLM
        system_prompt = f"""
            You are an expert MLOps Threat Intelligence Agent. Assist the security team in investigating system exceptions and transaction frauds.

            [SYSTEM REAL-TIME TRANSACTION METRICS]:
            {velocity_context}

            [CONVERSATIONAL SESSION HISTORY]:
            {chat_history}

            User Request: {user_query}
            Analyze cleanly, look for malicious vectors, and provide tactical next actions.
            """

        # Generate response from the LLM
        response = await self.llm.invoke(system_prompt)

        # Commit updated dialogue frame back to Redis cache (expiring in 12 hours)
        updated_history = f"{chat_history}\nUser: {user_query}\nAgent: {response}"
        await redis_cache.client.setex(
            memory_key, 43200, updated_history
        )  # 12 hours in seconds

        return response


# Singleton instance for application-wide use
security_agent_engine = SecurityAgentEngine()
