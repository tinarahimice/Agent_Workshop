import asyncio
import logging
import signal
from pathlib import Path
from src.config import Settings, get_settings
from src.ingest import build_index
from src.ocr import process_image
from src.queue import Job, JobQueue, JobType
from src.redis_client import get_redis
log=logging.getLogger("WORKER")
async def execute(job: Job, settings: Settings) -> None:
    if job.type is JobType.OCR:
        if not job.file_path: raise ValueError("OCR job has no file_path")
        await process_image(Path(job.file_path),settings=settings)
        await asyncio.to_thread(build_index,False,settings)
    elif job.type is JobType.INGEST: await asyncio.to_thread(build_index,False,settings)
    elif job.type is JobType.REINDEX: await asyncio.to_thread(build_index,True,settings)
async def run_worker(settings: Settings | None=None) -> None:
    settings=settings or get_settings(); queue=JobQueue(get_redis(settings),settings.job_max_retries); stopping=asyncio.Event()
    loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM): loop.add_signal_handler(sig,stopping.set)
    log.info("worker started")
    while not stopping.is_set():
        try:
            job=await queue.claim()
            if job:
                try: await execute(job,settings); await queue.complete(job)
                except Exception as exc: await queue.fail(job,str(exc))
        except Exception as exc:
            log.error("Redis/worker error: %s",exc); await asyncio.sleep(2)
    log.info("worker stopped")
