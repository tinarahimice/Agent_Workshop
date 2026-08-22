"""Typed, centralized workshop configuration."""
from functools import lru_cache
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    openai_api_key: str = ""
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_request_timeout: float = Field(120.0, gt=0)
    jina_api_key: str = ""
    embedding_model: str = "jina-embeddings-v3"
    reranker_model: str = "jina-reranker-v2-base-multilingual"
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(600, ge=1)
    chunk_size: int = Field(512, ge=64)
    chunk_overlap: int = Field(50, ge=0)
    retrieval_top_k: int = Field(8, ge=1)
    rerank_top_n: int = Field(3, ge=1)
    job_max_retries: int = Field(3, ge=0)
    log_level: str = "INFO"
    data_dir: Path = ROOT / "data"
    ocr_dir: Path = ROOT / "storage/ocr"
    index_dir: Path = ROOT / "storage/index"
    prompt_path: Path = ROOT / "prompts/system_prompt.txt"

    @model_validator(mode="after")
    def valid_chunks(self):
        if self.llm_provider not in {"openai", "ollama"}:
            raise ValueError("LLM_PROVIDER must be either 'openai' or 'ollama'")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.rerank_top_n > self.retrieval_top_k:
            raise ValueError("RERANK_TOP_N cannot exceed RETRIEVAL_TOP_K")
        return self

    def require_api_key(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing; copy .env.example to .env and add it.")

    def require_jina_api_key(self) -> None:
        if not self.jina_api_key:
            raise RuntimeError("JINA_API_KEY is missing; create a Jina AI key and add it to .env.")

@lru_cache
def get_settings() -> Settings:
    return Settings()
