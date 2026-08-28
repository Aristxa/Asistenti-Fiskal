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

BASE = "https://huggingface.co/datasets/aristeaaa/asistenti-fiskal-korpus/resolve/main"

# Each arm is uploaded separately as soon as it is built, so a disconnect during
# the second cannot destroy the first. Both are fetched and unpacked into the
# same index directory.
ARMS = ("article", "fixed")


def download() -> list[Path]:
    import urllib.request

    PROCESSED.mkdir(parents=True, exist_ok=True)
    paths = []
    for arm in ARMS:
        target = PROCESSED / f"index-{arm}.zip"
        if target.exists():
            print(f"  {target.name} ekziston, po përdoret")
        else:
            print(f"po shkarkohet index-{arm}.zip ...")
            urllib.request.urlretrieve(f"{BASE}/index-{arm}.zip", target)
            print(f"  {target.name}  ({target.stat().st_size / 1e6:.0f} MB)")
        paths.append(target)
    return paths


def unpack(archives: list[Path]) -> None:
    # Replace rather than merge: a half-old index whose vectors no longer match
    # its chunks.jsonl would still load and would still return results.
    if INDEX_ROOT.exists():
        shutil.rmtree(INDEX_ROOT)
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(PROCESSED)
    print(f"  u shpaketuan {len(archives)} arkiva në {INDEX_ROOT}")


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
    unpack(download())
    # The GPU script does not write model.txt per arm, so stamp it here. Retrieval
    # reads this to encode queries with the model the corpus was encoded with.
    stamp = INDEX_ROOT / "model.txt"
    if not stamp.exists():
        stamp.write_text("BAAI/bge-m3", encoding="utf-8")
        print("  u shënua modeli: BAAI/bge-m3")
    verify()


if __name__ == "__main__":
    main()
