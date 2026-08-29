import pytest
from pydantic import ValidationError

from src.config import Settings


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("openai", "openai"),
        (" OpenAI ", "openai"),
        ("openrouter", "openai"),
        ("gapgpt", "openai"),
        (" GAPGPT ", "openai"),
        ("OPENAI-COMPATIBLE", "openai"),
        ("openai_compatible", "openai"),
        ("ollama", "ollama"),
        ("", "openai"),
    ],
)
def test_llm_provider_is_normalized(configured: str, expected: str) -> None:
    settings = Settings(_env_file=None, llm_provider=configured)

    assert settings.llm_provider == expected


def test_unknown_llm_provider_has_actionable_error() -> None:
    with pytest.raises(ValidationError, match="Unsupported LLM_PROVIDER='unknown'"):
        Settings(_env_file=None, llm_provider="unknown")


@pytest.mark.parametrize("field", ["embedding_provider", "reranker_provider"])
def test_retrieval_providers_accept_ollama(field: str) -> None:
    settings = Settings(_env_file=None, **{field: " OLLAMA "})

    assert getattr(settings, field) == "ollama"


@pytest.mark.parametrize("field", ["embedding_provider", "reranker_provider"])
def test_unknown_retrieval_provider_has_actionable_error(field: str) -> None:
    with pytest.raises(ValidationError, match="use 'jina' or 'ollama'"):
        Settings(_env_file=None, **{field: "unknown"})


def test_fully_local_retrieval_does_not_require_jina_key() -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="ollama",
        reranker_provider="ollama",
        jina_api_key="",
    )

    settings.require_jina_api_key()


def test_defaults_are_fully_local_ollama() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "ollama"
    assert settings.embedding_provider == "ollama"
    assert settings.reranker_provider == "ollama"


def test_streamlit_api_models_are_explicit_opt_in() -> None:
    settings = Settings(_env_file=None)

    local = settings.for_streamlit(False)
    cloud = settings.for_streamlit(True)

    assert (local.llm_provider, local.embedding_provider, local.reranker_provider) == (
        "ollama", "ollama", "ollama"
    )
    assert (cloud.llm_provider, cloud.embedding_provider, cloud.reranker_provider) == (
        "openai", "jina", "jina"
    )


def test_gapgpt_environment_uses_openai_compatible_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.gapgpt.app/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4")
    monkeypatch.setenv("LLM_PROVIDER", "gapgpt")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == "test-key"
    assert settings.openai_base_url == "https://api.gapgpt.app/v1"
    assert settings.llm_model == "gpt-4"
