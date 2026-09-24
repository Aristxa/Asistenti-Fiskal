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
import zipfile
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

# A line of only digits and punctuation carries no title. Some source PDFs place a
# stray number between `Neni N` and the article rubric -- a column value in a table,
# a running count left by the layout -- which made the rubric read as its own number
# ("Neni 5 - 5" instead of "Neni 5 - Regjistrimi i personit te tatueshem"). Skipping
# such lines recovers the real rubric.
NUMERIC_LINE = re.compile(r"^[\d\W_]+$")

# How far past the `Neni N` line to look for the rubric. Deliberately short: an
# article that genuinely has no rubric must yield "" rather than its first body
# sentence, which would read as a title the legislator never wrote.
RUBRIC_SCAN_END = 4

# A line opening with a numbered paragraph (`1. `, `2) `) is the article's body,
# not its rubric. Many articles carry no rubric at all and run straight into their
# first paragraph; without this the citation read `Neni 37 - 2. Ministri i
# Financave nxjerr udhezim per zbatimin e neneve 11, 12 15, ...`. Length cannot be
# used to tell the two apart -- ratification rubrics run past 120 characters -- but
# this prefix is the layout convention the documents actually follow.
BODY_PARAGRAPH = re.compile(r"^\d+[.)]\s")

# Paragraph number alone on its own line, with the body starting on the next.
# This is how the national accounting standards are laid out, and it is invisible
# to DECIMAL_HEADING, which requires the number and the text on one line. Without
# it, SKK 5 — a 60k-character standard — produced no citable unit at all.
PARAGRAPH_NUMBER = re.compile(r"(?m)^[ \t]*(\d{1,3})[ \t]*$")

FIXED_WINDOW_CHARS = 1400
FIXED_WINDOW_OVERLAP = 200

# Long articles are split so that no article chunk dwarfs a fixed window. Without
# this the two arms of the chunking comparison carry very different amounts of text
# per retrieved chunk and the comparison is confounded (docs/FINDINGS.md F4).
# Parts keep the article label, so a citation stays article-level regardless of
# which part was retrieved.
MAX_ARTICLE_CHARS = FIXED_WINDOW_CHARS


@dataclass
class Chunk:
    chunk_id: str
    doc_id: int
    strategy: str          # "article" | "fixed"
    label: str             # "Neni 12" | "Neni 12 (2/3)" | "1.1" | "window 3"
    cite: str              # canonical citation unit, part suffix stripped: "Neni 12"
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
    # Running headers, footers and correspondence letterhead. The DPT bulletins are
    # compilations of scanned reply letters, so every few hundred characters of real
    # content is followed by an address block, a phone number, a protocol line and a
    # bare page number. Left in, they pad chunks with text that means nothing, get
    # embedded and indexed like content, and make articles unreadable to a human
    # reviewer -- which is how they were noticed.
    for pattern in (
        r"(?m)^\s*Adresa:.*$",
        r"(?m)^\s*www\.tatime\.gov\.al\.?\s*$",
        r"(?m)^\s*Web\s*site:.*$",
        r"(?m)^\s*Tel:.*$",
        r"(?m)^\s*Fax:.*$",
        r"(?m)^\s*DREJTORIA E P[ËE]RGJITHSHME E TATIMEVE\s*$",
        r"(?m)^\s*Nr\.?_{2,}\s*Prot\.?\s*$",
        r"(?m)^\s*L[ëe]nda:.*$",
        r"(?m)^\s*Tiran[ëe],?\s*m[ëe]?_+.*$",
        # A bare number on its own line is NOT stripped, tempting as it looks. That
        # is exactly how the national accounting standards number their paragraphs,
        # and removing it erased the `standard` regime entirely — 18 documents,
        # including SKK 5, silently reverted to unstructured. Stray page numbers
        # are the cheaper problem.
    ):
        text = re.sub(pattern, "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> str:
    with fitz.open(path) as doc:
        return normalise("\n".join(page.get_text() for page in doc))


def read_docx(path: Path) -> str:
    """Extract text from a Word file by reading its XML directly.

    A .docx is a zip holding word/document.xml. Paragraph ends become newlines
    before tags are stripped, so the structural segmentation downstream still has
    lines to work with.

    Not every zip the site serves is a Word file — one download is an archive of
    XSD schemas. Checking for word/document.xml distinguishes them, which
    magic-byte sniffing alone cannot.
    """
    if not zipfile.is_zipfile(path):
        raise ValueError("not a zip container")
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError("zip is not a Word document")
        xml = archive.read("word/document.xml").decode("utf-8", "ignore")

    xml = xml.replace("</w:p>", "\n").replace("<w:br/>", "\n").replace("<w:tab/>", " ")
    return normalise(re.sub(r"<[^>]+>", "", xml))


def read_document(path: Path) -> str:
    """Dispatch on file type. Raises if nothing readable can be produced."""
    if path.suffix == ".pdf":
        return read_pdf(path)
    if path.suffix in (".docx", ".doc"):
        # The `.doc` label is a magic-byte guess and is sometimes wrong: at least
        # one file sniffed as OLE is really a Word XML package, so the zip path is
        # tried regardless of the extension.
        return read_docx(path)
    raise ValueError(f"unsupported type {path.suffix}")


def detect_regime(text: str) -> str:
    """Which structural convention this document follows.

    Order matters. `neni` and `decimal` are checked first because a document using
    them may also contain stray standalone numbers (page numbers, table cells);
    `standard` is the fallback for the accounting standards, whose paragraph
    numbers sit alone on a line.
    """
    if len(NENI_HEADING.findall(text)) >= 3:
        return "neni"
    if len(DECIMAL_HEADING.findall(text)) >= 5:
        return "decimal"
    if len(PARAGRAPH_NUMBER.findall(text)) >= 8:
        return "standard"
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


def _article_rubric(lines: list[str]) -> str:
    """The article's rubric, or "" when the article has none.

    Skips stray numeric lines, and stops at the first numbered paragraph: once the
    body has started there is no rubric to find, and returning a body sentence
    would put words in the legislator's mouth.
    """
    for line in lines[1:RUBRIC_SCAN_END]:
        if NUMERIC_LINE.fullmatch(line):
            continue
        if BODY_PARAGRAPH.match(line):
            return ""
        return line
    return ""


def segment_articles(text: str, regime: str) -> list[tuple[str, str, str]]:
    if regime == "neni":
        matches = list(NENI_HEADING.finditer(text))
        return _slice_on(
            text, matches,
            label_of=lambda m: f"Neni {m.group(1)}",
            # line 0 is the `Neni N` heading itself; the rubric is the first line
            # after it that is not a stray number (see NUMERIC_LINE).
            heading_of=_article_rubric,
        )
    if regime == "decimal":
        matches = list(DECIMAL_HEADING.finditer(text))
        return _slice_on(
            text, matches,
            label_of=lambda m: m.group(1),
            heading_of=lambda lines: lines[0][:120],
        )
    if regime == "standard":
        matches = list(PARAGRAPH_NUMBER.finditer(text))
        return _slice_on(
            text, matches,
            label_of=lambda m: f"Paragrafi {m.group(1)}",
            # line 0 is the bare number; the rubric is the text that follows it
            heading_of=lambda lines: (lines[1][:120] if len(lines) > 1 else ""),
        )
    return []


def split_long_article(label: str, heading: str, body: str) -> list[tuple[str, str, str]]:
    """Break an over-long article into parts that all keep the article label."""
    if len(body) <= MAX_ARTICLE_CHARS:
        return [(label, heading, body)]

    parts, start = [], 0
    while start < len(body):
        end = min(start + MAX_ARTICLE_CHARS, len(body))
        if end < len(body):
            nearest = body.rfind("\n", start + MAX_ARTICLE_CHARS // 2, end)
            if nearest != -1:
                end = nearest
        block = body[start:end].strip()
        if block:
            parts.append(block)
        if end >= len(body):
            break
        start = end

    total = len(parts)
    # Repeat the heading on every part so a mid-article chunk still says what it is.
    return [
        (f"{label} ({i}/{total})", heading, f"{label}\n{heading}\n{block}" if i > 1 else block)
        for i, block in enumerate(parts, start=1)
    ]


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

    sources = sorted(
        (p for p in RAW.iterdir()
         if p.suffix in (".pdf", ".docx", ".doc") and p.stem.isdigit()),
        key=lambda p: int(p.stem),
    )

    skipped = []
    for source in sources:
        doc_id = int(source.stem)
        meta = manifest.get(doc_id, {})
        if source.stat().st_size == 0:
            skipped.append((doc_id, "empty file"))
            continue
        try:
            text = read_document(source)
        except Exception as exc:
            skipped.append((doc_id, str(exc)[:60]))
            continue
        if len(text) < 200:
            skipped.append((doc_id, f"only {len(text)} chars of text"))
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
            # Which body issued this, so a citation can say so. Rows crawled before
            # multi-source support have no domain and default to tax legislation.
            "domain": meta.get("domain", "tatime"),
            "authority": meta.get("authority", "Drejtoria e Përgjithshme e Tatimeve"),
            "archival": bool(meta.get("archival", False)),
            "url": meta.get("url", f"https://www.tatime.gov.al/shkarko.php?id={doc_id}"),
            "regime": regime,
            "chars": len(text),
            "n_articles": len(articles),
        })

        order = 0
        for label, heading, body in articles:
            for part_label, part_heading, part_body in split_long_article(label, heading, body):
                if len(part_body) < min_chars:
                    continue
                chunks.append(asdict(Chunk(
                    chunk_id=f"{doc_id}:article:{order}",
                    doc_id=doc_id, strategy=strategy_label, label=part_label,
                    cite=label, heading=part_heading, text=part_body,
                    chars=len(part_body), order=order,
                )))
                order += 1

        for order, (label, heading, body) in enumerate(segment_fixed(text)):
            if len(body) < min_chars:
                continue
            chunks.append(asdict(Chunk(
                chunk_id=f"{doc_id}:fixed:{order}",
                doc_id=doc_id, strategy="fixed", label=label,
                cite=label, heading=heading, text=body, chars=len(body), order=order,
            )))

        print(f"  {doc_id:<6} {source.suffix[1:]:<5} {regime:<8} {len(text):>7} chars -> "
              f"{len(articles):>4} articles  {meta.get('title','')[:40]}")

    _write(PROCESSED / "documents.jsonl", documents)
    _write(PROCESSED / "chunks.jsonl", chunks)

    if skipped:
        # Printed, not swallowed: a document that yields no text still looks like a
        # successful run otherwise, and 31 scanned PDFs disappeared from the corpus
        # this way before this report existed (docs/FINDINGS.md F9).
        by_reason: dict[str, list[int]] = {}
        for doc_id, reason in skipped:
            key = "no extractable text" if "chars of text" in reason else reason
            by_reason.setdefault(key, []).append(doc_id)
        print(f"\nskipped {len(skipped)} of {len(sources)} sources:")
        for reason, ids in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            sample = ", ".join(str(i) for i in ids[:6])
            more = f" (+{len(ids) - 6} të tjera)" if len(ids) > 6 else ""
            print(f"  {len(ids):>3}  {reason:<28} {sample}{more}")

    print(f"\ndocuments: {len(documents)}  regimes: {regimes}")
    print(f"chunks: {len(chunks)}")
    # Chunk size per arm is reported every run: if the two arms drift apart again,
    # the chunking comparison is confounded and the numbers say so immediately.
    for arm in ("article", "fixed"):
        sizes = sorted(c["chars"] for c in chunks if c["strategy"] == arm)
        if not sizes:
            continue
        median = sizes[len(sizes) // 2]
        print(f"  {arm:<8} n={len(sizes):<6} median={median:<6} max={sizes[-1]:<6} "
              f"mean={sum(sizes) // len(sizes)}")


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build()
