"""Build the full index on a GPU, then bring the result back.

The build machine for this project has 4 CPU cores and no GPU, where encoding the
31k-chunk corpus takes roughly 3 hours with a small model and 10 with a good one.
On a free Colab T4 the same job with BAAI/bge-m3 — the best Albanian coverage of
the candidates — takes a few minutes. Indexing is a one-off, so it is worth moving.

HOW TO RUN
----------
1. Open https://colab.research.google.com, new notebook,
   Runtime -> Change runtime type -> T4 GPU.

2. Upload `data/processed/chunks.jsonl` and `data/processed/documents.jsonl`
   (left sidebar -> Files -> Upload). They are ~40 MB together.

3. Paste this whole file into a cell and run it. It installs dependencies,
   encodes both chunking arms on the GPU, and writes `index.zip`.

4. Download `index.zip`, unzip it into `data/processed/` so you have
   `data/processed/index/article/...` and `data/processed/index/fixed/...`.

5. Locally, run `python -m src.index.build --bm25-only`. This builds the lexical
   index and the diacritic-restoration map in seconds.

Only the dense index is built here. BM25 and the restoration map depend on the
Albanian tokeniser in `src/index/albanian.py`, and duplicating that into this
standalone script would let the two copies drift — a silent mismatch between the
tokeniser that built the index and the one that queries it. Building them locally
against the real module removes that risk entirely.

The Space never runs this. It loads the committed index and encodes one query per
request, which is fast on CPU.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 64          # GPU has the memory; larger batches are much faster
MAX_SEQ_LENGTH = 512

PROCESSED = Path("data/processed") if Path("data/processed").exists() else Path(".")
INDEX_ROOT = Path("index")


def embed_text(chunk: dict) -> str:
    heading = chunk.get("heading") or ""
    label = chunk.get("cite") or chunk.get("label") or ""
    prefix = f"{label}. {heading}".strip(" .")
    return f"{prefix}\n{chunk['text']}" if prefix else chunk["text"]


def install() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "sentence-transformers==3.3.1", "faiss-cpu==1.9.0"],
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
        print("WARNING: no GPU detected — this will be slow. "
              "Runtime -> Change runtime type -> T4 GPU")

    chunks_path = PROCESSED / "chunks.jsonl"
    rows = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"loaded {len(rows)} chunks from {chunks_path}")

    print(f"loading {MODEL_NAME} on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    for strategy in ("article", "fixed"):
        subset = [r for r in rows if r["strategy"] == strategy]
        if not subset:
            print(f"  {strategy}: no chunks, skipping")
            continue

        out = INDEX_ROOT / strategy
        out.mkdir(parents=True, exist_ok=True)

        texts = [embed_text(c) for c in subset]
        print(f"  {strategy}: encoding {len(texts)} chunks ...")
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
        print(f"  {strategy}: done, dim {vectors.shape[1]}")

    (INDEX_ROOT / "model.txt").write_text(MODEL_NAME, encoding="utf-8")
    shutil.make_archive("index", "zip", INDEX_ROOT.parent, INDEX_ROOT.name)
    print("\nwrote index.zip — download it and unzip into data/processed/")
    print(f"IMPORTANT: set EMBED_MODEL={MODEL_NAME} when running the app locally,")
    print("so queries are encoded with the same model the index was built with.")


if __name__ == "__main__":
    main()
