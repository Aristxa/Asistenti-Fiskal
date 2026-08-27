"""Crawl the non-tatime sources declared in src/ingest/sources.py.

These sites publish documents as ordinary PDF links on ordinary pages, so one
generic lister covers all of them. tatime.gov.al keeps its own module because its
listing is a numeric taxonomy with subcategories.

Records are appended to the same manifest, with a `source` and `domain` on every
row, so downstream code sees one corpus while still being able to say which
authority issued a given rule.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

from src.ingest.crawl import DELAY_SECONDS, MANIFEST, RAW, TIMEOUT, USER_AGENT, clean, sniff
from src.ingest.sources import EXTERNAL, Source, is_document_link

# Document ids for external sources are derived from the URL rather than supplied
# by the site, so they cannot collide with tatime.gov.al's numeric ids.
ID_BASE = 900_000


@dataclass
class ExtraRecord:
    doc_id: int
    title: str
    source: str
    domain: str
    authority: str
    category: str
    parent_category_id: None
    archival: bool
    url: str
    filetype: str
    sha256: str
    bytes: int
    fetched_at: str


def stable_id(url: str) -> int:
    """Deterministic id from the URL, so re-runs do not duplicate documents."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return ID_BASE + int(digest[:8], 16) % 90_000


def load_seen() -> set[int]:
    if not MANIFEST.exists():
        return set()
    seen = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seen.add(json.loads(line)["doc_id"])
    return seen


def list_links(client: httpx.Client, source: Source, page: str) -> list[tuple[str, str]]:
    """Return (absolute_url, title) for document links on one page.

    A page may itself be a document — the Labour Code is published as a bare PDF
    URL — in which case it is returned as the single link.
    """
    url = urljoin(source.base, page)
    if is_document_link(url, source.link_pattern):
        return [(url, Path(urlparse(url).path).stem.replace("-", " "))]

    resp = client.get(url)
    resp.raise_for_status()

    found: dict[str, str] = {}
    for node in HTMLParser(resp.text).css("a"):
        href = node.attributes.get("href") or ""
        if not is_document_link(href, source.link_pattern):
            continue
        absolute = urljoin(url, href)
        title = clean(node.text()) or Path(urlparse(absolute).path).stem.replace("-", " ")
        if len(title) > len(found.get(absolute, "")):
            found[absolute] = title
    return sorted(found.items())


def fetch(client: httpx.Client, url: str) -> bytes:
    for attempt in range(3):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            if attempt == 2:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    retry in {wait}s ({exc.__class__.__name__})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def crawl(only: str | None = None) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    seen = load_seen()
    print(f"manifest holds {len(seen)} documents")

    headers = {"User-Agent": USER_AGENT, "Accept-Language": "sq,en;q=0.5"}
    new_count = 0

    with httpx.Client(headers=headers, timeout=TIMEOUT, follow_redirects=True,
                      verify=False) as client:
        with MANIFEST.open("a", encoding="utf-8") as manifest:
            for source in EXTERNAL:
                if only and source.key != only:
                    continue
                print(f"\n[{source.key}] {source.name} — {source.authority}")

                links: list[tuple[str, str]] = []
                for page in source.pages:
                    try:
                        page_links = list_links(client, source, page)
                    except httpx.HTTPError as exc:
                        print(f"  {page}: failed ({exc})")
                        continue
                    print(f"  {page}: {len(page_links)} documents")
                    links.extend(page_links)
                    time.sleep(DELAY_SECONDS)

                for url, title in links:
                    doc_id = stable_id(url)
                    if doc_id in seen:
                        continue
                    try:
                        blob = fetch(client, url)
                    except Exception as exc:
                        print(f"    FAILED {url[:70]}: {exc}")
                        continue

                    kind = sniff(blob)
                    (RAW / f"{doc_id}.{kind}").write_bytes(blob)

                    record = ExtraRecord(
                        doc_id=doc_id,
                        title=title[:300],
                        source=source.key,
                        domain=source.domain,
                        authority=source.authority,
                        category=source.key,
                        parent_category_id=None,
                        archival=False,
                        url=url,
                        filetype=kind,
                        sha256=hashlib.sha256(blob).hexdigest(),
                        bytes=len(blob),
                        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    )
                    manifest.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                    manifest.flush()
                    seen.add(doc_id)
                    new_count += 1
                    print(f"    {doc_id} {kind} {len(blob)//1024:>5} KB  {title[:60]}")
                    time.sleep(DELAY_SECONDS)

    print(f"\ndone: {new_count} new, {len(seen)} total")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl the non-tatime sources")
    parser.add_argument("--source", help="only this source key (kkk, pune)")
    args = parser.parse_args()
    crawl(args.source)
