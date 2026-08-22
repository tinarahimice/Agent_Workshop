import pytest
from fakeredis.aioredis import FakeRedis
from src.cache import RagCache,cache_key

def test_key_is_deterministic_versioned_and_private():
    assert cache_key("  Return  POLICY ","v1")==cache_key("return policy","v1")
    assert cache_key("return policy","v1")!=cache_key("return policy","v2")
    assert "return policy" not in cache_key("return policy","v1")
@pytest.mark.asyncio
async def test_cache_hit_and_ttl():
    redis=FakeRedis(decode_responses=True); cache=RagCache(redis,ttl_seconds=30)
    assert await cache.get("q","v") is None
    await cache.set("q","v",{"answer":"a"})
    assert await cache.get("q","v")=={"answer":"a"}
    assert await redis.ttl(cache_key("q","v"))>0
