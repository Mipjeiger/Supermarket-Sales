import logging
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorStoreClient:
    """Manage cloude-hosted vector store interactions with Qdrant for embedding storage and retrieval."""

    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )

        self.collection_name  = "supermarket_production"

        # Embedded serverless model configuration maps
        self.hf_embed_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

        self._ensure_collection_state()

    def _ensure_collection_state(self):
        """Verify and create the Qdrant collection if it does not exist."""
        try:
            # Native pydantic structural cluster state inspection method
            if not self.client.collection_exists(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"🚀 Successfully provisioned remote collection '{self.collection_name}' on Qdrant Cloud.")

        except Exception as e:
            logger.error(f"Qdrant collection provisioning error: {e}")
            raise RuntimeError("Failed to ensure Qdrant collection state.") from e
        
    def get_embedding(self, text: str) -> list:
        """Extracts dense feature coordinates via serverless pipeline without relying on local dependencies."""
        if not settings.HUGGINGFACE_API_KEY:
            logger.error("Hugging Face API key is not configured. Please set HUGGINGFACE_API_KEY in your environment.")
            return [0.0] * 384  # Return a zero vector of the expected size
        
        try:
            response = requests.post(
                self.hf_embed_url,
                headers=self.headers,
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=12
            )

            if response.status_code != 200:
                return response.json()
            logger.error(f"❌ Hugging Face embedding extraction error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Exception during embedding extraction: {e}")
        
        return [0.0] * 384  # Clean numerical array block fallback response pattern
    
    def search_similar_products(self, query: str, limit: int = 3) -> list:
        """Queries remote data indices using matching vector proximity and returns the top N most similar product embeddings."""
        try:
            vector_coordinates = self.get_embedding(query)
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector_coordinates,
                limit=limit
            )
            return [hit.payload for hit in hits]
        
        except Exception as e:
            logger.error(f"❌ Qdrant search operation failed: {e}")
            return []  # Return an empty list on failure
        
# Singleton initialization for global vector store access
vector_store = VectorStoreClient()