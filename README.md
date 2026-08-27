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

## The claim

One sentence, and the whole project defends it:

> **In Albanian legal question answering, correctness is decided by retrieval, not by
> the language model — and segmenting documents on their legal structure rather than on
> fixed character windows is what makes retrieval find the right article.**

Three contributions support it, chosen because each is a thing that visibly *exists*
rather than a conclusion that has to be argued for:

1. **A benchmark** — the first public Albanian tax-law QA set, questions paired with the
   controlling article. Albanian is low-resource and has essentially no legal QA resources.
2. **A measured finding** — whether structure-aware chunking beats fixed-window chunking
   at finding that article.
3. **A working public system** — deployed, citing its sources, usable during a defense.

### How results are reported

No p-values, no significance tests. In their place a rule fixed before any result was
seen: **a difference under 10 percentage points is reported as "no clear difference,"**
and every number is printed as `n/N` beside its percentage so the sample size is always
visible. That is honest about what a benchmark of this size can distinguish, and it can
be defended out loud in one sentence.

See [`docs/PLAN.md`](docs/PLAN.md) for the full design, and
[`docs/FINDINGS.md`](docs/FINDINGS.md) for the dated findings log.

## The thesis

The written thesis lives in [`docs/teza/`](docs/teza/), in Albanian, following the
official Universiteti i Tiranës / FEUT format. It is kept as one file per chapter,
mirroring the system's components, because the system keeps growing and a document
written from memory at the end would be wrong.

Every chapter declares a status, and `python scripts/build_teza.py` assembles the
document and reports how complete it actually is. Chapters that depend on unperformed
measurements stay empty by rule — the evaluation chapter contains its measurement plan
and blank tables, not estimated results.

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
machine has no GPU — a measured run of the full corpus locally came to 5–6 hours,
against minutes on a T4 with a better encoder. See `docs/FINDINGS.md` F6. The index is committed; the Space
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
src/eval/review.py      candidate -> Excel workbook -> benchmark round trip
scripts/build_index_gpu.py   run this on Colab to index the full corpus
app.py                  Gradio Space entry point
docs/PLAN.md            the claim, contributions, and how results are reported
docs/FINDINGS.md        dated findings log
eval/gold/candidates.jsonl   120 selected articles, stratified by category
eval/gold/rishikim.csv       review workbook (open in Excel)
eval/gold/questions.jsonl    the benchmark                [awaiting review]
```

## Data and licensing

Only public legislative texts are collected; no personal data. Documents remain the
property of the Republic of Albania. Raw PDFs are gitignored — run the crawler to
reproduce the corpus.
