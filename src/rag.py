import logging
import hashlib
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from src.cache import RagCache
from src.config import Settings, get_settings
from src.ingest import configure_embedding, qdrant_vector_store, read_index_version
from src.llm import create_llm
from src.redis_client import get_redis
from src.rerank import FastEmbedRerank
log=logging.getLogger("RAG")
class RagResult(BaseModel):
    answer: str
    sources: list[str]
    cached: bool = False


def rag_cache_version(index_version: str, settings: Settings) -> str:
    """Invalidate answers when either indexed data or retrieval models change."""
    retrieval_config = (
        f"{settings.embedding_provider}:{settings.embedding_model}:{settings.ollama_embedding_model}:"
        f"{settings.reranker_provider}:{settings.reranker_model}:{settings.fastembed_reranker_model}:"
        f"{settings.sparse_model}:{settings.hybrid_alpha}:{settings.qdrant_collection}:"
        f"{settings.retrieval_top_k}:{settings.rerank_top_n}:"
        f"{settings.llm_provider}:{settings.llm_model}:{settings.ollama_model}"
    )
    suffix = hashlib.sha256(retrieval_config.encode()).hexdigest()[:8]
    return f"{index_version}-{suffix}"

async def query_knowledge_base(question: str, settings: Settings | None = None, cache: RagCache | None = None) -> RagResult:
    settings=settings or get_settings()
    if not question.strip(): raise ValueError("Question must not be empty")
    version=rag_cache_version(read_index_version(settings.index_dir), settings)
    cache=cache or RagCache(get_redis(settings),settings.cache_enabled,settings.cache_ttl_seconds)
    cached=await cache.get(question,version)
    if cached: return RagResult(**cached,cached=True)
    configure_embedding(settings)
    index=VectorStoreIndex.from_vector_store(qdrant_vector_store(settings))
    if settings.reranker_provider == "fastembed":
        reranker = FastEmbedRerank(
            model=settings.fastembed_reranker_model,
            top_n=settings.rerank_top_n,
        )
    else:
        settings.require_jina_api_key()
        reranker = JinaRerank(
            api_key=settings.jina_api_key,
            model=settings.reranker_model,
            top_n=settings.rerank_top_n,
        )
    engine=index.as_query_engine(
        llm=create_llm(settings),
        similarity_top_k=settings.retrieval_top_k,
        sparse_top_k=settings.retrieval_top_k,
        vector_store_query_mode="hybrid",
        alpha=settings.hybrid_alpha,
        node_postprocessors=[reranker],
    )
    response=await engine.aquery("Answer only from the retrieved BitTeck context. If absent, say unavailable. Question: "+question)
    sources=sorted({node.node.metadata.get("file_name","unknown") for node in response.source_nodes})
    result=RagResult(answer=str(response),sources=sources)
    await cache.set(question,version,result.model_dump(exclude={"cached"}))
    log.info("retrieved chunks=%d",len(response.source_nodes)); return result
