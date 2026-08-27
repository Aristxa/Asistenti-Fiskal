"""Crawl the legislation section of tatime.gov.al.

Documents are listed on category pages as `shkarko.php?id=N` links and served
as PDFs from that endpoint with no Content-Disposition, so the file type is
determined from magic bytes rather than trusted from the response.

The site publishes no robots.txt, so the crawler self-imposes a conservative
delay, identifies itself, and never re-fetches a document it already holds.
Runs are resumable: state lives in the manifest, not in memory.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser

BASE = "https://www.tatime.gov.al"
RAW = Path("data/raw")
MANIFEST = RAW / "manifest.jsonl"

USER_AGENT = (
    "AsistentiFiskal/0.1 (academic research crawler; "
    "public legislation only; contact: github.com/ledjolleshaj)"
)
DELAY_SECONDS = 1.5
TIMEOUT = httpx.Timeout(60.0, connect=30.0)

# Legislation categories, from the taxonomy at /c/6/legjislacioni
CATEGORIES: dict[int, str] = {
    69: "procedurat-tatimore",
    70: "tatimi-mbi-te-ardhurat",
    71: "tatimi-mbi-vleren-e-shtuar",
    72: "taksat-kombetare",
    73: "kontributet-e-sigurimeve",
    74: "lojrat-e-fatit",
    75: "taksat-vendore",
    125: "marreveshje-nderkombetare",
    184: "akte-te-dpt",
    219: "marreveshje-kombetare",
    452: "fiskalizimi",
    455: "shkembimi-automatik-i-informacionit",
    497: "arshiva-e-akteve-ligjore",
    591: "sinjalizimi",
}

MAGIC = {
    b"%PDF": "pdf",
    b"PK\x03\x04": "docx",
    b"\xd0\xcf\x11\xe0": "doc",
}


@dataclass
class Record:
    doc_id: int
    title: str
    category_id: int
    category: str
    url: str
    filetype: str
    sha256: str
    bytes: int
    fetched_at: str


def sniff(blob: bytes) -> str:
    for magic, kind in MAGIC.items():
        if blob.startswith(magic):
            return kind
    return "unknown"


def clean(text: str) -> str:
    """Collapse whitespace and strip the &nbsp; padding the site appends."""
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip(" .;")


def load_manifest() -> dict[int, Record]:
    if not MANIFEST.exists():
        return {}
    seen = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            seen[row["doc_id"]] = row
    return seen


def list_documents(client: httpx.Client, cat_id: int, slug: str) -> list[tuple[int, str]]:
    """Return (doc_id, title) for every document linked from a category page."""
    resp = client.get(f"{BASE}/c/6/{cat_id}/{slug}")
    resp.raise_for_status()

    found: dict[int, str] = {}
    for node in HTMLParser(resp.text).css("a"):
        href = node.attributes.get("href") or ""
        match = re.search(r"shkarko\.php\?id=(\d+)", href)
        if not match:
            continue
        doc_id = int(match.group(1))
        title = clean(node.text())
        # A document can be linked more than once; keep the most descriptive text.
        if len(title) > len(found.get(doc_id, "")):
            found[doc_id] = title
    return sorted(found.items())


def fetch_document(client: httpx.Client, doc_id: int) -> bytes:
    for attempt in range(3):
        try:
            resp = client.get(f"{BASE}/shkarko.php", params={"id": doc_id})
            resp.raise_for_status()
            return resp.content
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            if attempt == 2:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    retry in {wait}s ({exc.__class__.__name__})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def crawl(categories: dict[int, str] | None = None, limit_per_category: int | None = None) -> None:
    categories = categories or CATEGORIES
    RAW.mkdir(parents=True, exist_ok=True)

    seen = load_manifest()
    print(f"manifest holds {len(seen)} documents")

    headers = {"User-Agent": USER_AGENT, "Accept-Language": "sq,en;q=0.5"}
    new_count = 0

    with httpx.Client(headers=headers, timeout=TIMEOUT, follow_redirects=True, verify=False) as client:
        with MANIFEST.open("a", encoding="utf-8") as manifest:
            for cat_id, slug in categories.items():
                try:
                    documents = list_documents(client, cat_id, slug)
                except httpx.HTTPError as exc:
                    print(f"[{cat_id}] {slug}: category page failed ({exc}), skipping")
                    continue

                if limit_per_category:
                    documents = documents[:limit_per_category]
                print(f"[{cat_id}] {slug}: {len(documents)} documents listed")
                time.sleep(DELAY_SECONDS)

                for doc_id, title in documents:
                    if doc_id in seen:
                        continue
                    try:
                        blob = fetch_document(client, doc_id)
                    except Exception as exc:
                        print(f"    id={doc_id} FAILED: {exc}")
                        continue

                    kind = sniff(blob)
                    digest = hashlib.sha256(blob).hexdigest()
                    (RAW / f"{doc_id}.{kind}").write_bytes(blob)

                    record = Record(
                        doc_id=doc_id,
                        title=title,
                        category_id=cat_id,
                        category=slug,
                        url=f"{BASE}/shkarko.php?id={doc_id}",
                        filetype=kind,
                        sha256=digest,
                        bytes=len(blob),
                        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    )
                    manifest.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                    manifest.flush()
                    seen[doc_id] = asdict(record)
                    new_count += 1
                    print(f"    id={doc_id} {kind} {len(blob)//1024:>5} KB  {title[:70]}")
                    time.sleep(DELAY_SECONDS)

    print(f"\ndone: {new_count} new, {len(seen)} total")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl tatime.gov.al legislation")
    parser.add_argument("--category", type=int, help="crawl a single category id")
    parser.add_argument("--limit", type=int, help="max documents per category")
    args = parser.parse_args()

    selected = {args.category: CATEGORIES[args.category]} if args.category else None
    crawl(selected, args.limit)
