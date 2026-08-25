from pathlib import Path

import pytest

from src.ingest import read_index_version


def test_missing_index_explains_docker_and_local_setup(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        read_index_version(tmp_path)

    message = str(error.value)
    assert "docker compose --profile cli run --rm app ingest" in message
    assert "python -m src.main ingest" in message
