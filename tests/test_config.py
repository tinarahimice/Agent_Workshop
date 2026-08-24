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
