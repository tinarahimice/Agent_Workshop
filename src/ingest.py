"""LlamaIndex document -> chunks -> embeddings -> persistent index pipeline."""
import hashlib
import json
import shutil
from pathlib import Path
from llama_index.core import Document, Settings as LlamaSettings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.jinaai import JinaEmbedding
from src.config import Settings, get_settings
VERSION_FILE = "index_version.json"

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
        f"{settings.reranker_model}:{settings.retrieval_top_k}:{settings.rerank_top_n}".encode()
    )
    for path in document_paths(settings): digest.update(path.name.encode()+path.read_bytes())
    return digest.hexdigest()[:16]

def read_index_version(index_dir: Path) -> str:
    path=index_dir / VERSION_FILE
    if not path.exists(): raise FileNotFoundError("Index is missing. Run `python -m src.main ingest` first.")
    return json.loads(path.read_text())["version"]

def configure_embedding(settings: Settings) -> None:
    settings.require_jina_api_key()
    LlamaSettings.embed_model = JinaEmbedding(
        model=settings.embedding_model,
        api_key=settings.jina_api_key,
    )

def build_index(reindex: bool = False, settings: Settings | None = None) -> str:
    settings=settings or get_settings(); configure_embedding(settings)
    docs=load_documents(settings)
    if not docs: raise RuntimeError("No documents found; run `python -m src.main generate-data`.")
    if reindex and settings.index_dir.exists(): shutil.rmtree(settings.index_dir)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    nodes=split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    index=VectorStoreIndex(nodes); index.storage_context.persist(persist_dir=str(settings.index_dir))
    version=fingerprint(settings)
    (settings.index_dir / VERSION_FILE).write_text(json.dumps({"version":version,"documents":len(docs),"chunks":len(nodes)}, indent=2))
    return version
