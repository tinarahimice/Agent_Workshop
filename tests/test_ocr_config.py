from pathlib import Path

import pytest

from src.ocr import resolve_tesseract_command


def test_resolve_tesseract_command_uses_path(monkeypatch) -> None:
    monkeypatch.setattr("src.ocr.shutil.which", lambda command: "/usr/bin/tesseract")

    assert resolve_tesseract_command("tesseract") == "/usr/bin/tesseract"


def test_resolve_tesseract_command_accepts_configured_path(monkeypatch) -> None:
    configured = str(Path("/opt/tesseract/bin/tesseract"))
    monkeypatch.setattr("src.ocr.shutil.which", lambda command: command)

    assert resolve_tesseract_command(configured) == configured


def test_missing_tesseract_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setattr("src.ocr.shutil.which", lambda command: None)

    with pytest.raises(RuntimeError, match="TESSERACT_CMD"):
        resolve_tesseract_command("tesseract")
