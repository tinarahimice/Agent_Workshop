"""Replaceable OCR boundary; Tesseract runs off the event loop."""
import asyncio
import logging
import shutil
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
    def __init__(self, command: str = "tesseract") -> None:
        self.command = resolve_tesseract_command(command)

    async def extract_text(self, file_path: Path) -> str:
        return await asyncio.to_thread(self._extract, file_path)

    def _extract(self, file_path: Path) -> str:
        pytesseract.pytesseract.tesseract_cmd = self.command
        with Image.open(file_path) as image:
            return pytesseract.image_to_string(image)

def resolve_tesseract_command(command: str) -> str:
    """Resolve Tesseract once and provide installation guidance if absent."""
    configured = command.strip() or "tesseract"
    resolved = shutil.which(configured)
    if resolved:
        return resolved
    raise RuntimeError(
        f"Tesseract executable {configured!r} was not found. Install it with "
        "`apt-get install tesseract-ocr` (Debian/Ubuntu) or "
        "`brew install tesseract` (macOS), set TESSERACT_CMD to its full path, "
        "or run OCR in the Docker image. After changing Docker dependencies, "
        "rebuild with `docker compose build --no-cache`."
    )

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
    text = normalize_text(
        await (service or TesseractOCRService(settings.tesseract_cmd)).extract_text(path)
    )
    if not text: raise RuntimeError(f"OCR returned empty text for {path.name}; check image quality/Tesseract.")
    settings.ocr_dir.mkdir(parents=True, exist_ok=True)
    output = settings.ocr_dir / f"{path.stem}.txt"; output.write_text(text + "\n", encoding="utf-8")
    log.info("OCR COMPLETED output=%s", output)
    return output

async def process_all(settings: Settings | None = None) -> list[Path]:
    settings = settings or get_settings(); outputs=[]
    # A missing executable is an environment problem shared by every image, not
    # a corrupt-document error. Fail once instead of logging the same failure
    # for every scan and then returning a misleading successful exit status.
    service = TesseractOCRService(settings.tesseract_cmd)
    for path in sorted((settings.data_dir / "scanned").glob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES: continue
        try: outputs.append(await process_image(path, service=service, settings=settings))
        except Exception as exc: log.error("OCR FAILED file=%s error=%s", path.name, exc)
    return outputs
