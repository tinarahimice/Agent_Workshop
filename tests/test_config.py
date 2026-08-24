import pytest
from pydantic import ValidationError

from src.config import Settings


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("openai", "openai"),
        (" OpenAI ", "openai"),
        ("openrouter", "openai"),
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
