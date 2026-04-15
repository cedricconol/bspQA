"""Download PDFs listed in manifest.json into ingestion/data/raw."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024
TIMEOUT_S = 120
USER_AGENT = "bspQA-ingestion/1.0 (manifest fetch)"


def _manifest_path() -> Path:
    return Path(__file__).resolve().parent / "manifest.json"


def _output_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "raw"


def _filename_for_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name or "download.pdf"
    return unquote(name)


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    mpath = _manifest_path()
    if not mpath.is_file():
        logger.error("Manifest not found: %s", mpath)
        return 1

    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)

    with mpath.open(encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    if not documents:
        logger.error("No documents in manifest.")
        return 1

    errors = 0
    for item in documents:
        url = item.get("url")
        if not url:
            logger.warning("Skipping entry without url: %r", item)
            errors += 1
            continue

        dest = out / _filename_for_url(url)
        if dest.exists():
            logger.info("Skip (exists): %s", dest.name)
            continue

        try:
            logger.info("Downloading: %s", url)
            _download(url, dest)
            logger.info("  -> %s", dest)
        except urllib.error.HTTPError as e:
            logger.error("HTTP %s for %s: %s", e.code, url, e.reason)
            errors += 1
        except urllib.error.URLError as e:
            reason = e.reason if isinstance(e.reason, str) else repr(e.reason)
            logger.error("URL error for %s: %s", url, reason)
            errors += 1
        except OSError as e:
            logger.error("IO error for %s: %s", url, e)
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
