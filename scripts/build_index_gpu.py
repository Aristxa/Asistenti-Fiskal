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

   ~45 min on a T4 for the full corpus (measured, not estimated).

3. Getting the result off the machine. Colab VMs are ephemeral — when the runtime
   disconnects the filesystem is discarded, and a 151 MB index left sitting there
   is simply gone. That happened once already and cost 43 minutes of GPU time.

   Preferred: add a Hugging Face write token to Colab's secret manager once —
   key icon in the left sidebar → New secret → name it `HF_TOKEN` → paste the
   token → enable "Notebook access". The script then uploads the index itself
   and nothing depends on the tab staying open.

   Otherwise: download `index.zip` from the Files pane immediately, before the
   runtime idles.

4. Back on this machine:

       python scripts/fetch_index.py        # if it went to the Hub
       python -m src.index.build --bm25-only
       python scripts/prepare_space.py

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

HUB_REPO = "aristeaaa/asistenti-fiskal-korpus"

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


def push_to_hub() -> bool:
    """Upload the finished index to the Hub, if a token is available.

    Colab machines are ephemeral: when the runtime disconnects the filesystem is
    discarded. A 151 MB artifact left sitting there is lost the moment the tab
    idles, which is exactly what happened on the first run -- 43 minutes of GPU
    time thrown away. Pushing it as soon as it exists makes the work survive.

    The token is read from Colab's secret manager, never from the notebook text.
    Add it once: key icon in the left sidebar -> New secret -> name HF_TOKEN ->
    paste a write token -> enable "Notebook access".
    """
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("HF_TOKEN")
    except Exception:
        import os
        token = os.environ.get("HF_TOKEN")

    if not token:
        print("\n(pa HF_TOKEN — indeksi mbetet vetëm në këtë makinë)")
        return False

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
                   check=True)
    from huggingface_hub import HfApi

    print("po ngarkohet indeksi në Hub ...")
    HfApi().upload_file(
        path_or_fileobj="index.zip",
        path_in_repo="index.zip",
        repo_id=HUB_REPO,
        repo_type="dataset",
        token=token,
    )
    print(f"  u ngarkua te {HUB_REPO}")
    return True


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
