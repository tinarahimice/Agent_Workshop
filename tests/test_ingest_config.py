from pathlib import Path

import pytest

from src.config import Settings
from src.ingest import qdrant_collection_exists, read_index_version


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
