# Asistenti Fiskal

> Albanian-language question answering over Albanian tax and business legislation,
> where every factual claim is bound to the legal article that supports it — or the
> system refuses to answer.

**Status:** pipeline complete end-to-end and validated on a subset. The benchmark
(the part that requires domain judgement) is not built yet — that is the next step,
and it is the thesis's main contribution.

---

## Why

Albanian tax obligations are governed by a fragmented corpus — primary laws (*ligje*),
implementing guidelines (*udhëzime*), VKM decisions and annual amendments, spread across
14 categories on `tatime.gov.al` plus multi-year archives. A general-purpose LLM answers
questions about it fluently and often wrongly, with no way for the reader to verify.

This system answers only what the law says, cites the controlling article, links the
official source, states the vintage of the law it cites, and refuses out-of-scope
questions.

It does **not** give personalised tax advice, compute anyone's liability, or substitute
for a licensed accountant.

## Design questions

The project is built to thesis standard, around three questions that are not
foregone conclusions:

- **RQ1** — Does hybrid retrieval (dense + BM25) beat either component alone on Albanian
  legal queries? Albanian is morphologically rich, which hurts lexical matching; multilingual
  encoders have thin Albanian coverage. Neither is obviously dominant.
- **RQ2** — Does structure-aware chunking (one chunk = one *nen*) beat fixed-window
  chunking at equal retrieved-token budget? The retrieval unit and the citation unit
  coincide for legal text, which is the property being tested.
- **RQ3** — Under an explicit citation contract, what are the unsupported-claim rate and
  the false-refusal rate? Both directions must be measured; a system that refuses
  everything is trivially faithful.

See [`docs/PLAN.md`](docs/PLAN.md) for the full methodology and evaluation design, and
[`docs/FINDINGS.md`](docs/FINDINGS.md) for the empirical findings log.

## Pipeline

```
tatime.gov.al  ──crawl──▶  data/raw/       PDFs + manifest.jsonl
                              │
                         extract (PyMuPDF)
                              │
                    detect structural regime
                     ┌────────┴────────┐
                  neni │ decimal    flat │
                     └────────┬────────┘
                              ▼
              data/processed/chunks.jsonl
              (article-aware AND fixed-window, both arms)
                              │
                  dense (FAISS) + BM25 ──▶ RRF fusion
                              │
              Claude under citation contract
                              │
                    answer + citations, or refusal
```

Indexing runs on a free Colab GPU (`scripts/build_index_gpu.py`) because the build
machine has no GPU — see `docs/FINDINGS.md` F6. The index is committed; the Space
encodes only the question, so a free CPU tier is enough.

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# crawl one category (polite: 1.5s delay, resumable, skips what it has)
.venv/Scripts/python -m src.ingest.crawl --category 71 --limit 10

# crawl everything
.venv/Scripts/python -m src.ingest.crawl

# extract + segment
.venv/Scripts/python -m src.ingest.extract
```

Windows note: set `PYTHONIOENCODING=utf-8` before running, or the console mangles
Albanian diacritics on output (the data itself is fine).

## Current corpus

**253 documents, 262 MB, 31,083 chunks** (16,917 article-aware, 14,166 fixed-window)
across 11 categories of Albanian tax legislation.
Structural regimes: 129 decimal, 78 `neni`, 46 flat. The VAT law segments into 165
articles with sequential labels.

Five of 258 crawled files are `.docx`/`.doc` rather than PDF and are not yet
extracted (~2% of the corpus).

## Layout

```
src/ingest/crawl.py     rate-limited resumable crawler, magic-byte type sniffing
src/ingest/extract.py   PyMuPDF extraction, regime detection, dual segmentation
src/index/build.py      dense + BM25 index construction
src/rag/retrieve.py     dense / BM25 / RRF hybrid, deduplicated by citation
src/rag/answer.py       citation contract, cached system prompt
src/eval/retrieval_eval.py   hit-rate measurement, no API needed
src/eval/make_candidates.py  benchmark candidates + overlap guard
scripts/build_index_gpu.py   run this on Colab to index the full corpus
app.py                  Gradio Space entry point
docs/PLAN.md            the claim, contributions, and how results are reported
docs/FINDINGS.md        dated findings log
eval/gold/              benchmark                        [next step]
```

## Data and licensing

Only public legislative texts are collected; no personal data. Documents remain the
property of the Republic of Albania. Raw PDFs are gitignored — run the crawler to
reproduce the corpus.
