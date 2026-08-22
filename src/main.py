import argparse
import asyncio
import json
import logging
from pathlib import Path
from src.agent import run_agent
from src.config import get_settings
from src.ingest import build_index
from src.logging_config import configure_logging
from src.ocr import process_all
from src.queue import JobQueue, JobType
from src.rag import query_knowledge_base
from src.redis_client import close_redis, get_redis
from src.worker import run_worker

def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="llm-agent-workshop",description="NovaTech OCR, RAG, and agent workshop")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("generate-data"); sub.add_parser("ocr"); sub.add_parser("ingest"); sub.add_parser("reindex"); sub.add_parser("worker"); sub.add_parser("health")
    for name in ("rag","agent"):
        item=sub.add_parser(name); item.add_argument("question")
    item=sub.add_parser("enqueue-ocr"); item.add_argument("file_path",type=Path)
    item=sub.add_parser("enqueue-ingest"); item.add_argument("--reindex",action="store_true")
    item=sub.add_parser("job-status"); item.add_argument("job_id")
    return p
async def dispatch(args: argparse.Namespace) -> None:
    settings=get_settings()
    if args.command=="generate-data":
        from scripts.generate_mock_data import main as text_main
        from scripts.generate_scanned_docs import main as scan_main
        text_main(); scan_main()
    elif args.command=="ocr": print("\n".join(map(str,await process_all(settings))))
    elif args.command in {"ingest","reindex"}: print("Index version:",await asyncio.to_thread(build_index,args.command=="reindex",settings))
    elif args.command=="rag": print(query_result_json(await query_knowledge_base(args.question,settings)))
    elif args.command=="agent": print(await run_agent(args.question,settings))
    elif args.command=="worker": await run_worker(settings)
    elif args.command=="enqueue-ocr":
        from src.ocr import validate_scanned_path
        path=validate_scanned_path(args.file_path,settings); print((await JobQueue(get_redis(settings),settings.job_max_retries).enqueue(JobType.OCR,path)).model_dump_json(indent=2))
    elif args.command=="enqueue-ingest":
        kind=JobType.REINDEX if args.reindex else JobType.INGEST; print((await JobQueue(get_redis(settings),settings.job_max_retries).enqueue(kind)).model_dump_json(indent=2))
    elif args.command=="job-status":
        job=await JobQueue(get_redis(settings)).get(args.job_id); print(job.model_dump_json(indent=2) if job else "Job not found")
    elif args.command=="health": await health(settings)
def query_result_json(result) -> str: return result.model_dump_json(indent=2)
async def health(settings) -> None:
    checks={
        "redis": False,
        "llm_provider": settings.llm_provider,
        "llm_configured": (
            bool(settings.openai_api_key)
            if settings.llm_provider == "openai"
            else bool(settings.ollama_base_url and settings.ollama_model)
        ),
        "jina_configured": bool(settings.jina_api_key),
        "index_exists": (settings.index_dir / "index_version.json").exists(),
    }
    try: checks["redis"]=bool(await get_redis(settings).ping())
    except Exception as exc: checks["redis_error"]=f"Unavailable: {exc}"
    print(json.dumps(checks,indent=2))
async def amain() -> int:
    settings=get_settings(); configure_logging(settings.log_level)
    try: await dispatch(parser().parse_args()); return 0
    except Exception as exc:
        logging.getLogger("CLI").error("%s",exc,exc_info=settings.log_level.upper()=="DEBUG"); return 1
    finally: await close_redis()
def main() -> None: raise SystemExit(asyncio.run(amain()))
if __name__=="__main__": main()
