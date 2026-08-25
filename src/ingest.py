"""Documents -> chunks -> dense and BM25 sparse vectors -> Qdrant pipeline."""
import hashlib
import json
from pathlib import Path
from llama_index.core import Document, Settings as LlamaSettings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.config import Settings, get_settings
VERSION_FILE = "index_version.json"
INDEX_SETUP_HELP = (
    "Index is missing. Create it first with "
    "`docker compose --profile cli run --rm app ingest` (Docker) or "
    "`python -m src.main ingest` (local Python)."
)

def document_paths(settings: Settings) -> list[Path]:
    return sorted(settings.data_dir.glob("*.txt")) + sorted(settings.ocr_dir.glob("*.txt"))

def load_documents(settings: Settings | None = None) -> list[Document]:
    settings = settings or get_settings(); documents=[]
    for path in document_paths(settings):
        text = path.read_text(encoding="utf-8").strip()
        if text: documents.append(Document(text=text, metadata={"file_name": path.name, "path": str(path.relative_to(settings.data_dir.parent))}))
    return documents

def split_documents(documents: list[Document], chunk_size: int, overlap: int):
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap).get_nodes_from_documents(documents)

def fingerprint(settings: Settings) -> str:
    digest=hashlib.sha256()
    digest.update(
        f"{settings.chunk_size}:{settings.chunk_overlap}:{settings.embedding_model}:"
        f"{settings.reranker_model}:{settings.sparse_model}:{settings.hybrid_alpha}:"
        f"{settings.qdrant_collection}:{settings.retrieval_top_k}:{settings.rerank_top_n}".encode()
    )
    for path in document_paths(settings): digest.update(path.name.encode()+path.read_bytes())
    return digest.hexdigest()[:16]

def read_index_version(index_dir: Path) -> str:
    path=index_dir / VERSION_FILE
    if not path.exists(): raise FileNotFoundError(INDEX_SETUP_HELP)
    return json.loads(path.read_text())["version"]

def configure_embedding(settings: Settings) -> None:
    settings.require_jina_api_key()
    LlamaSettings.embed_model = JinaEmbedding(
        model=settings.embedding_model,
        api_key=settings.jina_api_key,
    )

def qdrant_client(settings: Settings) -> QdrantClient:
    """Create the shared HTTP client without embedding credentials in code."""
    return QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout)

def qdrant_collection_exists(client: QdrantClient, settings: Settings) -> bool:
    """Check Qdrant separately so connection failures identify the failing service."""
    try:
        return client.collection_exists(settings.qdrant_collection)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Qdrant at {settings.qdrant_url!r}. Start it with "
            "`docker compose up -d qdrant`, confirm port 6333 is reachable, "
            "and then rerun ingestion. "
            f"Qdrant reported: {type(exc).__name__}: {exc}"
        ) from exc

def qdrant_vector_store(settings: Settings, client: QdrantClient | None = None) -> QdrantVectorStore:
    """Configure dense + BM25 sparse vectors and server-side hybrid fusion."""
    return QdrantVectorStore(
        client=client or qdrant_client(settings),
        collection_name=settings.qdrant_collection,
        enable_hybrid=True,
        fastembed_sparse_model=settings.sparse_model,
    )

def build_index(reindex: bool = False, settings: Settings | None = None) -> str:
    settings=settings or get_settings(); configure_embedding(settings)
    docs=load_documents(settings)
    if not docs: raise RuntimeError("No documents found; run `python -m src.main generate-data`.")
    client=qdrant_client(settings)
    # Every ingestion is a snapshot replacement, so removed documents cannot
    # survive in Qdrant. ``reindex`` remains an explicit workshop command.
    if qdrant_collection_exists(client, settings):
        client.delete_collection(settings.qdrant_collection)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    nodes=split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    vector_store=qdrant_vector_store(settings, client)
    storage_context=StorageContext.from_defaults(vector_store=vector_store)
    try:
        VectorStoreIndex(nodes, storage_context=storage_context)
    except Exception as exc:
        raise RuntimeError(
            "Index creation failed while generating Jina embeddings or downloading/"
            "running the BM25 sparse model, or while uploading vectors to Qdrant. "
            "Confirm JINA_API_KEY is valid, outbound HTTPS to api.jina.ai is "
            "available, and Qdrant remains reachable. The first BM25 run may need "
            "additional time to download its model. Set LOG_LEVEL=DEBUG for "
            f"the underlying traceback. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    version=fingerprint(settings)
    (settings.index_dir / VERSION_FILE).write_text(json.dumps({"version":version,"documents":len(docs),"chunks":len(nodes)}, indent=2))
    return version
