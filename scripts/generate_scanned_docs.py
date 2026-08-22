"""Generate deterministic, scanned-looking pages without downloads."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
ROOT = Path(__file__).resolve().parents[1]
PAGES = {
 "product_catalog.png": ["NOVATECH SUPPLEMENTAL CATALOG", "Product: NovaMonitor Ultra", "SKU: NMU-900", "Category: Monitor", "Price: $850", "Warranty: 36 months", "Stock: Available", "Description: 32-inch color-accurate display."],
 "warranty_policy.png": ["NOVATECH SPECIAL WARRANTY NOTICE", "NovaMonitor Ultra pixel guarantee:", "During the first 90 calendar days,", "one or more bright pixels qualifies the", "display for warranty replacement.", "Proof of purchase is required."],
}
def make_page(lines: list[str], target: Path) -> None:
    image = Image.new("L", (1400, 900), 242); draw = ImageDraw.Draw(image); font = ImageFont.load_default(size=32)
    for y, line in enumerate(lines, 110): draw.text((120, y*55-5900), line, fill=25, font=font)
    # slight blur/noise-like bands give a scan appearance while preserving OCR readability
    draw.line((50, 70, 1350, 68), fill=180, width=2); image = image.filter(ImageFilter.GaussianBlur(.25))
    image.save(target)
def main() -> None:
    out = ROOT / "data/scanned"; out.mkdir(parents=True, exist_ok=True)
    for name, lines in PAGES.items(): make_page(lines, out/name)
    print(f"Generated {len(PAGES)} scanned documents")
if __name__ == "__main__": main()
