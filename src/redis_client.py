"""One shared, lazily-created async Redis client."""
from redis.asyncio import Redis
from src.config import Settings, get_settings
_client: Redis | None = None

def get_redis(settings: Settings | None = None) -> Redis:
    global _client
    if _client is None:
        _client = Redis.from_url((settings or get_settings()).redis_url, decode_responses=True)
    return _client

async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
