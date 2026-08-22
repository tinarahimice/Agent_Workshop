import pytest
from fakeredis.aioredis import FakeRedis
from src.queue import JobQueue,JobState,JobType
@pytest.mark.asyncio
async def test_job_lifecycle(monkeypatch):
    redis=FakeRedis(decode_responses=True); queue=JobQueue(redis,max_retries=1)
    job=await queue.enqueue(JobType.INGEST)
    # fakeredis versions may not implement BLMOVE; exercise equivalent atomic move.
    async def move(*args): return await redis.lmove(args[0],args[1],"RIGHT","LEFT")
    monkeypatch.setattr(redis,"blmove",move)
    claimed=await queue.claim(); assert claimed and claimed.state==JobState.PROCESSING
    await queue.complete(claimed); assert (await queue.get(job.id)).state==JobState.COMPLETED
@pytest.mark.asyncio
async def test_retry_then_fail(monkeypatch):
    redis=FakeRedis(decode_responses=True); queue=JobQueue(redis,max_retries=1); await queue.enqueue(JobType.INGEST)
    async def move(*args): return await redis.lmove(args[0],args[1],"RIGHT","LEFT")
    monkeypatch.setattr(redis,"blmove",move)
    job=await queue.claim(); await queue.fail(job,"first"); assert (await queue.get(job.id)).state==JobState.PENDING
    job=await queue.claim(); await queue.fail(job,"again"); stored=await queue.get(job.id)
    assert stored.state==JobState.FAILED and stored.attempt==2 and stored.error=="again"
