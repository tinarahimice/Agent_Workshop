from pathlib import Path

import pytest

from src.config import Settings
from src.ingest import (
    async_qdrant_client,
    embedding_setup_help,
    qdrant_client,
    qdrant_collection_exists,
    qdrant_vector_store,
    read_index_version,
)


def test_embedding_setup_help_explains_host_ollama_recovery() -> None:
    settings = Settings(
        ollama_base_url="http://localhost:11434",
        ollama_embedding_model="embeddinggemma",
    )

    message = embedding_setup_help(settings)

    assert "ollama pull embeddinggemma" in message
    assert "http://localhost:11434" in message


def test_embedding_setup_help_explains_compose_ollama_recovery() -> None:
    settings = Settings(ollama_base_url="http://ollama:11434")

    message = embedding_setup_help(settings)

    assert "docker compose --profile ollama up -d ollama" in message
    assert "docker compose --profile ollama exec ollama ollama pull" in message


def test_missing_index_explains_docker_and_local_setup(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        read_index_version(tmp_path)

    message = str(error.value)
    assert "docker compose --profile cli run --rm app ingest" in message
    assert "python -m src.main ingest" in message


def test_qdrant_timeout_has_actionable_error() -> None:
    class TimedOutClient:
        def collection_exists(self, _collection: str) -> bool:
            raise TimeoutError("timed out")

    settings = Settings(qdrant_url="http://localhost:6333")
    with pytest.raises(RuntimeError) as error:
        qdrant_collection_exists(TimedOutClient(), settings)  # type: ignore[arg-type]

    message = str(error.value)
    assert "Cannot reach Qdrant" in message
    assert "docker compose up -d qdrant" in message
    assert "TimeoutError: timed out" in message


@pytest.mark.parametrize(
    "url",
    ["http://localhost:6333", "http://127.0.0.1:6333", "http://[::1]:6333"],
)
def test_qdrant_client_bypasses_proxy_for_loopback(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    captured: dict[str, object] = {}

    def fake_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("src.ingest.QdrantClient", fake_client)

    qdrant_client(Settings(qdrant_url=url))

    assert captured["trust_env"] is False


def test_qdrant_client_keeps_proxy_support_for_remote_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("src.ingest.QdrantClient", fake_client)

    qdrant_client(Settings(qdrant_url="https://qdrant.example.com"))

    assert captured["trust_env"] is True


def test_async_qdrant_client_bypasses_proxy_for_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("src.ingest.AsyncQdrantClient", fake_client)

    async_qdrant_client(Settings(qdrant_url="http://localhost:6333"))

    assert captured["trust_env"] is False


def test_vector_store_receives_sync_and_async_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_client = object()
    async_client = object()
    captured: dict[str, object] = {}

    def fake_vector_store(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("src.ingest.async_qdrant_client", lambda _settings: async_client)
    monkeypatch.setattr("src.ingest.QdrantVectorStore", fake_vector_store)

    qdrant_vector_store(Settings(), client=sync_client)  # type: ignore[arg-type]

    assert captured["client"] is sync_client
    assert captured["aclient"] is async_client
