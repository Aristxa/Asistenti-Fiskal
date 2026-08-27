"""Where the corpus comes from.

The system is built for an Albanian accountant, and an accountant's obligations do
not live in one place. Tax law answers what to declare; the Labour Code answers
how to compute a payroll; the accounting standards answer how to present the
result. A tax-only corpus can only ever answer a third of the questions its user
actually has, and — worse — it would answer the other two thirds *confidently and
wrongly*, since nothing in the retrieved context would signal that the question is
outside what the corpus covers.

Each source declares its own listing strategy, because government sites do not
agree on how to publish documents. What they share is the download-and-record
path, so adding a source means describing where its links are, not writing another
crawler.

Every document carries a `domain`. It is not decoration: an answer about annual
leave must be traceable to the Labour Code rather than to a VAT guideline, and the
domain is what lets a citation say which authority issued the rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str                     # short id, used in the manifest
    name: str                    # human name, Albanian
    domain: str                  # tatime | kontabilitet | pune | biznes
    base: str
    authority: str               # which body issues these documents
    # "categories": tatime.gov.al's /c/6/{id}/{slug} taxonomy
    # "pages": a list of pages whose PDF links are the documents
    strategy: str
    pages: tuple[str, ...] = ()
    link_pattern: str = r"\.pdf(\?|$)"
    notes: str = ""


# tatime.gov.al keeps its own module (src/ingest/crawl.py) because its listing
# is a numeric taxonomy with subcategories, not a page of links.
TATIME = Source(
    key="tatime",
    name="Legjislacioni tatimor",
    domain="tatime",
    base="https://www.tatime.gov.al",
    authority="Drejtoria e Përgjithshme e Tatimeve",
    strategy="categories",
    notes="Shih src/ingest/crawl.py — taksonomi me nënkategori.",
)

KKK = Source(
    key="kkk",
    name="Standardet Kombëtare të Kontabilitetit",
    domain="kontabilitet",
    base="https://kkk.gov.al",
    authority="Këshilli Kombëtar i Kontabilitetit",
    strategy="pages",
    pages=(
        "/standardet-kombetare-te-kontabilitetit-date-efektive-1-janar-2025/",
        "/standardet-nderkombetare-te-kontabilitetit-dhe-raportimit-financiar/",
    ),
    notes=(
        "SKK 1-15 plus shtojcat me formatet e pasqyrave financiare "
        "(bilanci, pasqyra e performancës, fluksi monetar). robots.txt lejon "
        "këto shtigje; ndalon vetëm /wp-admin/."
    ),
)

INSPEKTORIATI = Source(
    key="pune",
    name="Kodi i Punës dhe legjislacioni i punës",
    domain="pune",
    base="https://inspektoriatipunes.gov.al",
    authority="Inspektorati Shtetëror i Punës dhe i Shërbimeve Shoqërore",
    strategy="pages",
    pages=(
        "/wp-content/uploads/2024/08/Kodi-i-punes-perditesuar-2024.pdf",
        "/legjislacioni/",
    ),
    notes="Kodi i Punës i përditësuar me ndryshimet e 2024 (ligji 91/2024).",
)

SOURCES: tuple[Source, ...] = (TATIME, KKK, INSPEKTORIATI)

EXTERNAL = tuple(s for s in SOURCES if s.strategy == "pages")

BY_KEY = {s.key: s for s in SOURCES}


def domain_of(source_key: str) -> str:
    source = BY_KEY.get(source_key)
    return source.domain if source else "tatime"


DOMAIN_LABELS = {
    "tatime": "Legjislacion tatimor",
    "kontabilitet": "Kontabilitet dhe raportim financiar",
    "pune": "Marrëdhëniet e punës",
    "biznes": "Regjistrimi dhe veprimtaria e biznesit",
}


def is_document_link(href: str, pattern: str) -> bool:
    return bool(href) and bool(re.search(pattern, href, re.I))
