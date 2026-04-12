"""Convert each PDF in ingestion/data/raw to text via MarkItDown into ingestion/data/parsed."""

from __future__ import annotations

import sys
from pathlib import Path

from markitdown import MarkItDown


def _raw_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "raw"


def _parsed_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "parsed"


def main() -> int:
    raw = _raw_dir()
    if not raw.is_dir():
        print(f"Raw directory not found: {raw}", file=sys.stderr)
        return 1

    pdfs = sorted(raw.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files in {raw}", file=sys.stderr)
        return 1

    out = _parsed_dir()
    out.mkdir(parents=True, exist_ok=True)

    converter = MarkItDown()
    errors = 0
    for pdf in pdfs:
        dest = out / f"{pdf.stem}.txt"
        try:
            print(f"Parsing: {pdf.name}")
            result = converter.convert(pdf)
            text = (result.text_content or "").strip()
            if not text:
                text = (result.markdown or "").strip()
            dest.write_text(text + ("\n" if text else ""), encoding="utf-8")
            print(f"  -> {dest.name}")
        except OSError as e:
            print(f"IO error for {pdf.name}: {e}", file=sys.stderr)
            errors += 1
        except Exception as e:
            print(f"Failed to convert {pdf.name}: {e}", file=sys.stderr)
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
