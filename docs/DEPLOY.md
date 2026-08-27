# Deploying to Hugging Face Spaces

The Space runs on the **free CPU tier**. That is possible only because nothing
heavy happens at request time: the corpus is indexed offline and committed, and
the Space encodes one short question per request.

## What ships

```
app.py                      Gradio entry point
src/                        retrieval, generation, Albanian text processing
data/processed/index/       the committed indexes (dense + BM25 + restoration map)
requirements.txt
README.md                   from deploy/hf-space/README.md (carries the Space card)
```

Raw PDFs do **not** ship. Neither does `data/processed/chunks.jsonl` — the index
directory already holds the chunk text it needs.

## Steps

1. **Build the index** (once). On a free Colab T4, run `scripts/build_index_gpu.py`,
   download `index.zip`, unzip into `data/processed/`. Then locally:

   ```bash
   python -m src.index.build --bm25-only
   ```

   which builds the lexical index and the diacritic-restoration map in seconds
   against the real Albanian tokeniser.

2. **Create the Space** — huggingface.co → New Space → Gradio SDK → CPU basic.

3. **Track the index with git-lfs.** The bge-m3 index is ~130 MB.

   ```bash
   git lfs install
   git lfs track "*.faiss" "*.pkl"
   git add .gitattributes
   ```

4. **Use the Space card as the README.** Hugging Face reads the YAML frontmatter
   in `README.md` to configure the Space:

   ```bash
   cp deploy/hf-space/README.md README.md   # in the Space repo only
   ```

   Keep the project README on GitHub; the Space needs the frontmatter version.

5. **Set the secret.** Space → Settings → Variables and secrets → New secret:
   `ANTHROPIC_API_KEY`. Never commit it.

   Also set `EMBED_MODEL` to whatever `data/processed/index/model.txt` contains,
   or leave it — retrieval reads `model.txt` and uses that model regardless, which
   is what stops a silent encoder mismatch.

6. **Push.**

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/asistenti-fiskal
   git push space main
   ```

## Without an API key

The Space degrades gracefully: retrieval still runs and the matched articles are
shown, with a note that no answer was generated. Useful for demonstrating the
retrieval half without spending anything.

## Cost control

`src/web/ratelimit.py` caps usage at 10 questions per visitor per 10 minutes and
200 per hour globally, and questions over 500 characters are rejected. State is
per-process and resets when the Space sleeps — enough to stop runaway loops and
casual abuse, which is what a public demo needs.

The citation contract lives in a cached system prompt, so the repeated part of
every request is billed at roughly a tenth of the input rate.
