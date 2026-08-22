"""Replaceable OCR boundary; Tesseract runs off the event loop."""
import asyncio
import logging
import re
from pathlib import Path
from typing import Protocol
import pytesseract
from PIL import Image
from src.config import Settings, get_settings
log = logging.getLogger("OCR")
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

class OCRService(Protocol):
    async def extract_text(self, file_path: Path) -> str: ...

class TesseractOCRService:
    async def extract_text(self, file_path: Path) -> str:
        return await asyncio.to_thread(self._extract, file_path)
    @staticmethod
    def _extract(file_path: Path) -> str:
        with Image.open(file_path) as image:
            return pytesseract.image_to_string(image)

def normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip())

def validate_scanned_path(path: Path, settings: Settings) -> Path:
    resolved, root = path.resolve(), (settings.data_dir / "scanned").resolve()
    if resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document format: {resolved.suffix}; use {sorted(SUPPORTED_SUFFIXES)}")
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"OCR path must be an existing file below {root}")
    return resolved

async def process_image(path: Path, service: OCRService | None = None, settings: Settings | None = None) -> Path:
    settings = settings or get_settings(); path = validate_scanned_path(path, settings)
    log.info("OCR STARTED file=%s", path.name)
    text = normalize_text(await (service or TesseractOCRService()).extract_text(path))
    if not text: raise RuntimeError(f"OCR returned empty text for {path.name}; check image quality/Tesseract.")
    settings.ocr_dir.mkdir(parents=True, exist_ok=True)
    output = settings.ocr_dir / f"{path.stem}.txt"; output.write_text(text + "\n", encoding="utf-8")
    log.info("OCR COMPLETED output=%s", output)
    return output

async def process_all(settings: Settings | None = None) -> list[Path]:
    settings = settings or get_settings(); outputs=[]
    for path in sorted((settings.data_dir / "scanned").glob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES: continue
        try: outputs.append(await process_image(path, settings=settings))
        except Exception as exc: log.error("OCR FAILED file=%s error=%s", path.name, exc)
    return outputs
