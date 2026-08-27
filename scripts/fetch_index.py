"""Download the index built on Colab and unpack it into place.

Pairs with build_index_gpu.py, which uploads `index.zip` to the corpus dataset
when a token is present. Keeping the transfer in a script rather than in a
browser download means the step is reproducible and the index always lands in
the directory the rest of the code expects.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PROCESSED = Path("data/processed")
INDEX_ROOT = PROCESSED / "index"
ARCHIVE = PROCESSED / "index.zip"

HUB_URL = (
    "https://huggingface.co/datasets/aristeaaa/asistenti-fiskal-korpus/"
    "resolve/main/index.zip"
)


def download() -> Path:
    import urllib.request

    PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"po shkarkohet indeksi nga Hub ...")
    urllib.request.urlretrieve(HUB_URL, ARCHIVE)
    print(f"  {ARCHIVE}  ({ARCHIVE.stat().st_size / 1e6:.0f} MB)")
    return ARCHIVE


def unpack(archive: Path) -> None:
    # Replace rather than merge: a half-old index whose vectors no longer match
    # its chunks.jsonl would still load and would still return results.
    if INDEX_ROOT.exists():
        shutil.rmtree(INDEX_ROOT)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(PROCESSED)
    print(f"  u shpaketua në {INDEX_ROOT}")


def verify() -> None:
    problems = []
    for strategy in ("article", "fixed"):
        base = INDEX_ROOT / strategy
        for name in ("dense.faiss", "chunks.jsonl"):
            if not (base / name).exists():
                problems.append(f"mungon {strategy}/{name}")
    stamp = INDEX_ROOT / "model.txt"
    if not stamp.exists():
        problems.append("mungon model.txt")

    if problems:
        raise SystemExit("indeksi është i paplotë:\n  " + "\n  ".join(problems))

    print(f"\nmodeli: {stamp.read_text(encoding='utf-8').strip()}")
    for strategy in ("article", "fixed"):
        chunks = INDEX_ROOT / strategy / "chunks.jsonl"
        n = sum(1 for line in chunks.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"  {strategy:<8} {n:>7,} copëza")
    print("\nTani: python -m src.index.build --bm25-only")


def main() -> None:
    archive = ARCHIVE if ARCHIVE.exists() else download()
    if ARCHIVE.exists():
        print(f"po përdoret {ARCHIVE}")
    unpack(archive)
    verify()


if __name__ == "__main__":
    main()
