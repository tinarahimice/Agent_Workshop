"""A small reliable Redis list queue with durable job hashes."""
import json
import logging
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4
from pydantic import BaseModel
from src.namespaces import JOB_PREFIX, PENDING_QUEUE, PROCESSING_QUEUE
log=logging.getLogger("QUEUE")
class JobType(StrEnum): OCR="ocr"; INGEST="ingest"; REINDEX="reindex"
class JobState(StrEnum): PENDING="pending"; PROCESSING="processing"; COMPLETED="completed"; FAILED="failed"
class Job(BaseModel):
    id: str
    type: JobType
    file_path: str | None = None
    created_at: str
    attempt: int = 0
    state: JobState = JobState.PENDING
    error: str = ""

def job_key(job_id: str) -> str: return f"{JOB_PREFIX}:{job_id}"
class JobQueue:
    def __init__(self, redis: Any, max_retries: int=3): self.redis,self.max_retries=redis,max_retries
    async def enqueue(self, job_type: JobType, file_path: Path | None=None) -> Job:
        job=Job(id=str(uuid4()),type=job_type,file_path=str(file_path) if file_path else None,created_at=datetime.now(timezone.utc).isoformat())
        await self._save(job); await self.redis.lpush(PENDING_QUEUE,job.model_dump_json()); log.info("JOB ENQUEUED id=%s",job.id); return job
    async def claim(self, timeout: int=2) -> Job | None:
        # BLMOVE atomically moves a payload so a crash does not silently lose it.
        raw=await self.redis.blmove(PENDING_QUEUE,PROCESSING_QUEUE,timeout,"RIGHT","LEFT")
        if not raw: return None
        job=Job.model_validate_json(raw); job.state=JobState.PROCESSING; await self._save(job)
        log.info("JOB CLAIMED id=%s",job.id); return job
    async def complete(self, job: Job) -> None:
        job.state=JobState.COMPLETED; job.error=""; await self._remove_processing(job.id); await self._save(job); log.info("JOB COMPLETED id=%s",job.id)
    async def fail(self, job: Job, error: str) -> None:
        await self._remove_processing(job.id); job.attempt += 1; job.error=error
        if job.attempt <= self.max_retries:
            job.state=JobState.PENDING; await self._save(job); await self.redis.lpush(PENDING_QUEUE,job.model_dump_json()); log.warning("JOB RETRY id=%s attempt=%d",job.id,job.attempt)
        else:
            job.state=JobState.FAILED; await self._save(job); log.error("JOB FAILED id=%s error=%s",job.id,error)
    async def get(self, job_id: str) -> Job | None:
        data=await self.redis.hgetall(job_key(job_id)); return Job.model_validate(data) if data else None
    async def _save(self, job: Job) -> None:
        values={k:(v.value if isinstance(v,StrEnum) else v) for k,v in job.model_dump().items() if v is not None}
        await self.redis.hset(job_key(job.id),mapping=values)
    async def _remove_processing(self, job_id: str) -> None:
        for raw in await self.redis.lrange(PROCESSING_QUEUE,0,-1):
            if json.loads(raw).get("id")==job_id: await self.redis.lrem(PROCESSING_QUEUE,1,raw); break
