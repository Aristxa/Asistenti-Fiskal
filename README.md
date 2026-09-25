# Asistenti Fiskal

> Albanian-language question answering over Albanian tax and business legislation,
> where every factual claim is bound to the legal article that supports it — or the
> system refuses to answer.

**Try it:** [aristeaaa-assistent.hf.space](https://aristeaaa-assistent.hf.space) —
free, no account needed. The Space sleeps when idle, so the first request takes a
couple of minutes to wake it.

**Status:** complete and defended, September 2026. The benchmark was built, the
comparison was run under a rule fixed before any result was seen, and the headline
comparison came out null — which is reported as the result rather than reframed.

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


## Results

Sixty questions, each paired by hand with the article that answers it. Retrieval is
measured with no call to any language model, so the headline number is reproducible by
anyone for free.

| Retrieval | Chunking | Found the article | Found the act |
|---|---|---:|---:|
| **semantic** | **by article** | **28/60 = 47%** | 41/60 = 68% |
| semantic + keyword | by article | 23/60 = 38% | 62% |
| semantic + keyword | fixed window | 20/60 = 33% | 63% |
| semantic | fixed window | 16/60 = 27% | 63% |
| keyword | by article | 12/60 = 20% | 50% |
| keyword | fixed window | 10/60 = 17% | 47% |

Under the pre-registered rule — a difference counts only above 10 percentage points,
which on 60 questions is six questions:

- **article vs fixed window, on the hybrid arm: 5 points. Null.** This was the
  pre-registered comparison and it did not confirm the hypothesis.
- The same comparison on the semantic arm gives 20 points, but that arm was chosen
  *after* the results were seen, so it is reported as exploratory, not as proof.
- Hybrid over keyword-only: 18 points. Pre-registered, and a clear win.

One finding does not depend on the sample at all: a fixed window cites *"window 437"*,
whose boundaries coincide with no legal unit, so that strategy cannot produce a legal
citation even in principle.

On the generation side, across 81 measured answers **no citation marker refers to a
source that was not retrieved**, and 8 of the 9 refusals on answerable questions
happened after retrieval had already failed — the system declines when the context
genuinely does not hold the answer.


## The thesis

The written thesis is not in this repository. What is here is the system it describes,
its evaluation data, and the findings log — enough to reproduce every number above.

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
against minutes on a T4 with a better encoder. The index is committed; the Space
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
eval/gold/candidates.jsonl   120 selected articles, stratified by category
eval/gold/rishikim.csv       review workbook (open in Excel)
eval/gold/questions.jsonl    the benchmark: 82 questions, gold article assigned by hand
```

## Data and licensing

The code is MIT licensed — see [`LICENSE`](LICENSE).

Only public legislative texts are collected; no personal data. Documents remain the
property of the Republic of Albania. Raw PDFs and the built index are gitignored: run
the crawler to reproduce the corpus, or `python scripts/fetch_index.py` to pull the
prebuilt index from the Hub.
