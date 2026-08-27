# Asistenti Fiskal

> Albanian-language question answering over Albanian tax and business legislation,
> where every factual claim is bound to the legal article that supports it — or the
> system refuses to answer.

**Status:** early build. Corpus acquisition and structure-aware segmentation are
working end-to-end; retrieval, generation and evaluation are not yet built.

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
                  dense (FAISS) + BM25 ──▶ RRF fusion     [not built]
                              │
              Claude under citation contract              [not built]
```

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

## Current corpus sample

15 documents → 1,583 chunks (508 article-aware, 1,075 fixed-window).
Regimes: 9 decimal, 3 flat, 3 `neni`. The VAT law segments into 165 articles.

## Layout

```
src/ingest/crawl.py     rate-limited resumable crawler, magic-byte type sniffing
src/ingest/extract.py   PyMuPDF extraction, regime detection, dual segmentation
src/index/              dense + BM25 + fusion            [not built]
src/rag/                retrieval, citation contract     [not built]
src/eval/               gold set, metrics, significance  [not built]
src/web/                interface                        [not built]
docs/PLAN.md            methodology and evaluation design
docs/FINDINGS.md        dated findings log
eval/gold/              gold-standard QA set             [not built]
```

## Data and licensing

Only public legislative texts are collected; no personal data. Documents remain the
property of the Republic of Albania. Raw PDFs are gitignored — run the crawler to
reproduce the corpus.
