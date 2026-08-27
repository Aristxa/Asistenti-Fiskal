"""Extract text from crawled PDFs and segment it on legal structure.

Two things are load-bearing here.

**Extractor choice.** PyMuPDF, not pypdf. On the primary VAT law pypdf emitted
22.4% glued tokens (`Nekuptimteketijneni`) because the source PDFs position
glyphs instead of writing space characters; PyMuPDF reconstructs word boundaries
from glyph positions and emits 0%. See docs/FINDINGS.md.

**Structural regime.** The corpus is not uniform. Laws (*ligje*), council
decisions (*VKM*) and some guidelines are article-structured (`Neni 12`), while
many ministerial guidelines (*udhezime*) use decimal hierarchy (`1.` / `1.1` /
`a)`). A single segmenter that assumes `Neni` silently returns one giant chunk
for half the corpus, so the regime is detected per document and dispatched.

Both an article-aware and a fixed-window segmentation are emitted for every
document, so RQ2 can compare them at equal token budget on identical text.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # pymupdf

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# `Neni` heading on its own line; tolerates the missing space PDFs sometimes emit
# (`Neni1`) and bis-style suffixes (`Neni 12/1`, `Neni 12a`).
NENI_HEADING = re.compile(r"(?m)^[ \t]*Neni[ \t]*(\d+(?:[/\-]\d+)?[a-zë]?)[ \t]*$")

# Decimal section heading: `1.` `1.1` `2.3.4` followed by title text.
DECIMAL_HEADING = re.compile(r"(?m)^[ \t]*(\d+(?:\.\d+){0,3})\.?[ \t]+(?=[A-ZËÇ])")

FIXED_WINDOW_CHARS = 1400
FIXED_WINDOW_OVERLAP = 200


@dataclass
class Chunk:
    chunk_id: str
    doc_id: int
    strategy: str          # "article" | "fixed"
    label: str             # "Neni 12" | "1.1" | "window 3"
    heading: str
    text: str
    chars: int
    order: int


def normalise(text: str) -> str:
    """Repair the artefacts that survive extraction, without touching content."""
    text = text.replace("\xa0", " ").replace("​", "")
    # `Neni12` -> `Neni 12`; the site's PDFs are inconsistent about this space.
    text = re.sub(r"\bNeni(\d)", r"Neni \1", text)
    # Soft-hyphen line breaks inside words.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Running headers/footers repeated on every page.
    text = re.sub(r"(?m)^\s*Adresa:.*$", "", text)
    text = re.sub(r"(?m)^\s*www\.tatime\.gov\.al\s*$", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> str:
    with fitz.open(path) as doc:
        return normalise("\n".join(page.get_text() for page in doc))


def detect_regime(text: str) -> str:
    neni = len(NENI_HEADING.findall(text))
    decimal = len(DECIMAL_HEADING.findall(text))
    if neni >= 3:
        return "neni"
    if decimal >= 5:
        return "decimal"
    return "flat"


def _slice_on(text: str, matches: list, label_of, heading_of) -> list[tuple[str, str, str]]:
    """Cut `text` at each match; return (label, heading, body) triples."""
    out = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        out.append((label_of(match), heading_of(lines), block))
    return out


def segment_articles(text: str, regime: str) -> list[tuple[str, str, str]]:
    if regime == "neni":
        matches = list(NENI_HEADING.finditer(text))
        return _slice_on(
            text, matches,
            label_of=lambda m: f"Neni {m.group(1)}",
            # line 0 is the `Neni N` heading itself; line 1 is the article rubric
            heading_of=lambda lines: lines[1] if len(lines) > 1 else "",
        )
    if regime == "decimal":
        matches = list(DECIMAL_HEADING.finditer(text))
        return _slice_on(
            text, matches,
            label_of=lambda m: m.group(1),
            heading_of=lambda lines: lines[0][:120],
        )
    return []


def segment_fixed(text: str) -> list[tuple[str, str, str]]:
    out, start, index = [], 0, 0
    while start < len(text):
        end = min(start + FIXED_WINDOW_CHARS, len(text))
        # Prefer to break on a paragraph boundary near the window edge.
        if end < len(text):
            nearest = text.rfind("\n", start + FIXED_WINDOW_CHARS // 2, end)
            if nearest != -1:
                end = nearest
        block = text[start:end].strip()
        if block:
            out.append((f"window {index}", "", block))
            index += 1
        if end >= len(text):
            break
        start = max(end - FIXED_WINDOW_OVERLAP, start + 1)
    return out


def build(min_chars: int = 120) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.jsonl"
    manifest = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                manifest[row["doc_id"]] = row

    documents, chunks = [], []
    regimes: dict[str, int] = {}

    for pdf in sorted(RAW.glob("*.pdf"), key=lambda p: int(p.stem)):
        doc_id = int(pdf.stem)
        meta = manifest.get(doc_id, {})
        try:
            text = read_pdf(pdf)
        except Exception as exc:
            print(f"  {doc_id}: extraction failed ({exc})")
            continue

        regime = detect_regime(text)
        regimes[regime] = regimes.get(regime, 0) + 1

        articles = segment_articles(text, regime)
        strategy_label = "article"
        if not articles:  # flat documents have no structure to exploit
            articles = []

        documents.append({
            "doc_id": doc_id,
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "url": meta.get("url", f"https://www.tatime.gov.al/shkarko.php?id={doc_id}"),
            "regime": regime,
            "chars": len(text),
            "n_articles": len(articles),
        })

        for order, (label, heading, body) in enumerate(articles):
            if len(body) < min_chars:
                continue
            chunks.append(asdict(Chunk(
                chunk_id=f"{doc_id}:article:{order}",
                doc_id=doc_id, strategy=strategy_label, label=label,
                heading=heading, text=body, chars=len(body), order=order,
            )))

        for order, (label, heading, body) in enumerate(segment_fixed(text)):
            if len(body) < min_chars:
                continue
            chunks.append(asdict(Chunk(
                chunk_id=f"{doc_id}:fixed:{order}",
                doc_id=doc_id, strategy="fixed", label=label,
                heading=heading, text=body, chars=len(body), order=order,
            )))

        print(f"  {doc_id:<6} {regime:<8} {len(text):>7} chars -> {len(articles):>4} articles  {meta.get('title','')[:46]}")

    _write(PROCESSED / "documents.jsonl", documents)
    _write(PROCESSED / "chunks.jsonl", chunks)

    art = sum(1 for c in chunks if c["strategy"] == "article")
    fix = sum(1 for c in chunks if c["strategy"] == "fixed")
    print(f"\ndocuments: {len(documents)}  regimes: {regimes}")
    print(f"chunks: {len(chunks)} (article-aware {art}, fixed-window {fix})")


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build()
