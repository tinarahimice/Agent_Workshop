from src.config import Settings
from src.ingest import fingerprint,load_documents
from src.rag import rag_cache_version

def test_loads_text_and_ocr_documents(tmp_path):
    data=tmp_path/"data"; ocr=tmp_path/"ocr"; index=tmp_path/"index"; data.mkdir(); ocr.mkdir()
    (data/"products.txt").write_text("Product: Test"); (ocr/"scan.txt").write_text("OCR fact")
    settings=Settings(data_dir=data,ocr_dir=ocr,index_dir=index)
    docs=load_documents(settings)
    assert len(docs)==2 and {d.metadata["file_name"] for d in docs}=={"products.txt","scan.txt"}
    before=fingerprint(settings); (ocr/"scan.txt").write_text("changed")
    assert fingerprint(settings)!=before


def test_rerank_count_cannot_exceed_retrieval_count():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="RERANK_TOP_N"):
        Settings(retrieval_top_k=2, rerank_top_n=3)


def test_reranker_change_invalidates_rag_cache():
    first = Settings(reranker_model="reranker-a")
    second = Settings(reranker_model="reranker-b")
    assert rag_cache_version("index-v1", first) != rag_cache_version("index-v1", second)


def test_llm_provider_change_invalidates_rag_cache():
    hosted = Settings(llm_provider="openai")
    local = Settings(llm_provider="ollama")
    assert rag_cache_version("index-v1", hosted) != rag_cache_version("index-v1", local)


def test_invalid_llm_provider_is_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="LLM_PROVIDER"):
        Settings(llm_provider="unsupported")
