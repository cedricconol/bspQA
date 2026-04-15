"""Convert each PDF in ingestion/data/raw to text via MarkItDown into ingestion/data/parsed."""

from __future__ import annotations

import logging
from pathlib import Path

from markitdown import MarkItDown

logger = logging.getLogger(__name__)


def _raw_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "raw"


def _parsed_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "parsed"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw = _raw_dir()
    if not raw.is_dir():
        logger.error("Raw directory not found: %s", raw)
        return 1

    pdfs = sorted(raw.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files in %s", raw)
        return 1

    out = _parsed_dir()
    out.mkdir(parents=True, exist_ok=True)

    converter = MarkItDown()
    errors = 0
    for pdf in pdfs:
        dest = out / f"{pdf.stem}.txt"
        try:
            logger.info("Parsing: %s", pdf.name)
            result = converter.convert(pdf)
            text = (result.text_content or "").strip()
            if not text:
                text = (result.markdown or "").strip()
            dest.write_text(text + ("\n" if text else ""), encoding="utf-8")
            logger.info("  -> %s", dest.name)
        except OSError as e:
            logger.error("IO error for %s: %s", pdf.name, e)
            errors += 1
        except Exception as e:
            logger.error("Failed to convert %s: %s", pdf.name, e)
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
