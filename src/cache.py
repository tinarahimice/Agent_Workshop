import hashlib
import json
import logging
from typing import Any
from src.namespaces import CACHE_PREFIX
log = logging.getLogger("CACHE")

def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())

def cache_key(question: str, index_version: str) -> str:
    digest = hashlib.sha256(f"{index_version}\0{normalize_question(question)}".encode()).hexdigest()
    return f"{CACHE_PREFIX}:{index_version}:{digest}"

class RagCache:
    def __init__(self, redis: Any, enabled: bool = True, ttl_seconds: int = 600):
        self.redis, self.enabled, self.ttl_seconds = redis, enabled, ttl_seconds
    async def get(self, question: str, version: str) -> dict[str, Any] | None:
        if not self.enabled: return None
        raw = await self.redis.get(cache_key(question, version))
        log.info("CACHE HIT" if raw else "CACHE MISS")
        return json.loads(raw) if raw else None
    async def set(self, question: str, version: str, value: dict[str, Any]) -> None:
        if self.enabled:
            await self.redis.set(cache_key(question, version), json.dumps(value), ex=self.ttl_seconds)
            log.info("CACHE SET")
