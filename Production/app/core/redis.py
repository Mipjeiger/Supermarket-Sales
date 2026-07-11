import redis.asyncio as aioredis
from app.core.config import settings
import logging

logger = logging.getLogger("app.core.redis")


class RedisCacheManager:
    """Enterprise-grade async Redis connection manager for caching and pub/sub operations."""

    def __init__(self):
        self.client: aioredis.Redis | None = None

    def initialize(self) -> None:
        """Instantiate the Redis client with connection pooling."""
        self.client = aioredis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        logger.info("✅ Redis client async connection pool initialized successfully.")

    async def close(self) -> None:
        """Gracefully terminate connection pools during app shutdown."""
        if self.client:
            await self.client.close()
            logger.info("🔒 Redis client connection pool closed successfully.")


# Singleton
redis_cache = RedisCacheManager()
