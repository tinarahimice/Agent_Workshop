"""Choose the hosted OpenAI-compatible LLM or local Ollama fallback."""
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from src.config import Settings


def create_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "ollama":
        return Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=settings.ollama_request_timeout,
        )

    settings.require_api_key()
    kwargs = {"model": settings.llm_model, "api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["api_base"] = settings.openai_base_url
    return OpenAI(**kwargs)
