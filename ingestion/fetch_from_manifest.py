"""Download PDFs listed in manifest.json into ingestion/data/raw."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

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
    mpath = _manifest_path()
    if not mpath.is_file():
        print(f"Manifest not found: {mpath}", file=sys.stderr)
        return 1

    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)

    with mpath.open(encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    if not documents:
        print("No documents in manifest.", file=sys.stderr)
        return 1

    errors = 0
    for item in documents:
        url = item.get("url")
        if not url:
            print(f"Skipping entry without url: {item!r}", file=sys.stderr)
            errors += 1
            continue

        dest = out / _filename_for_url(url)
        if dest.exists():
            print(f"Skip (exists): {dest.name}")
            continue

        try:
            print(f"Downloading: {url}")
            _download(url, dest)
            print(f"  -> {dest}")
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} for {url}: {e.reason}", file=sys.stderr)
            errors += 1
        except urllib.error.URLError as e:
            reason = e.reason if isinstance(e.reason, str) else repr(e.reason)
            print(f"URL error for {url}: {reason}", file=sys.stderr)
            errors += 1
        except OSError as e:
            print(f"IO error for {url}: {e}", file=sys.stderr)
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
