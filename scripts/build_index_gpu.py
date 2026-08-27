"""Build the dense index on a Colab GPU, then bring the result back.

WHY THIS EXISTS
---------------
The build machine has four CPU cores and no GPU. A real run of the full corpus
there measured ~20 s per batch of 32 — roughly 2 h 50 for the article arm alone,
five to six hours for both, and only by using a weaker encoder. On a free Colab
T4 the same job with BAAI/bge-m3 — much better Albanian coverage — finishes in
minutes at full sequence length.

Indexing happens once. It is worth moving.

HOW TO RUN
----------
1. https://colab.research.google.com → New notebook
   Runtime → Change runtime type → **T4 GPU** → Save

2. Paste these two lines into a cell and run. The corpus is pulled from the Hub,
   so nothing needs uploading:

       !wget -q https://huggingface.co/datasets/aristeaaa/asistenti-fiskal-korpus/resolve/main/build_index_gpu.py
       exec(open("build_index_gpu.py").read())

   ~10 min including the model download.

4. Download `index.zip` from the Files pane (right-click → Download).
   Unzip it so you have `data/processed/index/article/…` and `…/fixed/…`.

5. Back on this machine:

       python -m src.index.build --bm25-only
       python scripts/prepare_space.py

   That builds the lexical index and the diacritic-restoration map in seconds,
   against the real Albanian tokeniser, and stages the Space.

WHAT THIS DOES NOT BUILD
------------------------
Only the dense index. BM25 and the restoration map depend on the Albanian
tokeniser in `src/index/albanian.py`; copying that into a standalone Colab script
would let the two versions drift, and a mismatch between the tokeniser that built
the index and the one that queries it fails silently rather than loudly. Building
them locally against the real module removes that risk entirely.
"""

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 64          # the T4 has the memory; larger batches are much faster
MAX_SEQ_LENGTH = 512

INDEX_ROOT = Path("index")

CANDIDATE_INPUTS = [
    Path("chunks.jsonl.gz"),
    Path("chunks.jsonl"),
    Path("data/processed/chunks.jsonl.gz"),
    Path("data/processed/chunks.jsonl"),
]


# Colab cannot be handed a local file by automation — its upload button opens a
# native OS dialog outside the page — so the corpus is published to the Hub and
# fetched here instead. It is public legislation, and a downloadable corpus is
# part of what makes the thesis reproducible.
CORPUS_URL = (
    "https://huggingface.co/datasets/aristeaaa/asistenti-fiskal-korpus/"
    "resolve/main/chunks.jsonl.gz"
)


def find_input() -> Path:
    for path in CANDIDATE_INPUTS:
        if path.exists():
            return path

    print("korpusi nuk u gjet lokalisht; po shkarkohet nga Hugging Face ...")
    import urllib.request

    target = Path("chunks.jsonl.gz")
    urllib.request.urlretrieve(CORPUS_URL, target)
    print(f"  u shkarkua {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def load_chunks(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def embed_text(chunk: dict) -> str:
    """Must match src/index/build.py — the index and the queries share this."""
    heading = chunk.get("heading") or ""
    label = chunk.get("cite") or chunk.get("label") or ""
    prefix = f"{label}. {heading}".strip(" .")
    return f"{prefix}\n{chunk['text']}" if prefix else chunk["text"]


def install() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "sentence-transformers", "faiss-cpu"],
        check=True,
    )


def main() -> None:
    install()

    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("KUJDES: nuk u gjet GPU. Runtime → Change runtime type → T4 GPU.")
        print("Pa GPU kjo zgjat orë, jo minuta.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    source = find_input()
    rows = load_chunks(source)
    print(f"u lexuan {len(rows):,} copëza nga {source}")

    print(f"po ngarkohet {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    for strategy in ("article", "fixed"):
        subset = [r for r in rows if r["strategy"] == strategy]
        if not subset:
            print(f"  {strategy}: pa copëza, u kapërcye")
            continue

        out = INDEX_ROOT / strategy
        out.mkdir(parents=True, exist_ok=True)

        texts = [embed_text(c) for c in subset]
        print(f"  {strategy}: po ngulitën {len(texts):,} copëza ...")
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(out / "dense.faiss"))

        with (out / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in subset:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        np.save(out / "dim.npy", np.array([vectors.shape[1]]))
        print(f"  {strategy}: u ndërtua, dimensioni {vectors.shape[1]}")

    # Read by src/rag/retrieve.py so queries are always encoded by the model the
    # corpus was encoded with. A dimension mismatch would raise; two different
    # models of the same dimension would not, and that is the failure to prevent.
    (INDEX_ROOT / "model.txt").write_text(MODEL_NAME, encoding="utf-8")

    shutil.make_archive("index", "zip", ".", str(INDEX_ROOT))
    size = Path("index.zip").stat().st_size / 1e6
    print(f"\nu shkrua index.zip ({size:.0f} MB)")
    print("Shkarkoje nga paneli Files, shpaketoje në data/processed/, pastaj:")
    print("  python -m src.index.build --bm25-only")
    print("  python scripts/prepare_space.py")


if __name__ == "__main__":
    main()
