"""Albanian-specific text processing for lexical retrieval.

Two properties of Albanian break naive BM25 on this corpus, and neither is
addressed by any general-purpose retrieval recipe.

**Diacritics are optional in practice, mandatory in the text.** Legal documents
are written with `ë` and `ç`; real users type `per`, `eshte`, `te` because the
Albanian layout is not what most keyboards default to. Measured on this corpus,
dropping diacritics from a question changed the BM25 top-5 completely (0/5
overlap on one query) — the same question, asked the way people actually type it,
retrieved nothing in common. Folding both sides makes matching diacritic-blind.

**Albanian is morphologically rich.** Nouns inflect for case, number and
definiteness, so the single concept *tatim* (tax) appears as `tatim`, `tatimi`,
`tatimit`, `tatimin`, `tatime`, `tatimet`, `tatimeve`. Exact-token BM25 treats
all seven as unrelated. There is no reliable open Albanian stemmer, so this
module implements a deliberately conservative suffix stripper.

Both transformations apply to the BM25 arm only. The dense arm keeps the original
text: multilingual encoders are trained on correctly written language, and folding
their input would degrade the embeddings rather than help them. The hybrid arm
therefore gets a diacritic-robust lexical signal and a linguistically intact
semantic one.
"""

from __future__ import annotations

import re

# ç and ë are the only non-ASCII letters in Albanian orthography.
FOLD = str.maketrans({"ë": "e", "Ë": "E", "ç": "c", "Ç": "C"})

# Case-insensitive on purpose. `tokenise` and `learn_restoration` lowercase their
# input first, but `restore` runs over the raw query, where a sentence-initial
# `Per` must still be matched and rewritten to `Për`.
TOKEN = re.compile(r"[a-zA-ZçÇëË0-9]+")

# High-frequency function words. Kept deliberately short: legal text is dense with
# meaning-bearing terms, and over-aggressive stopping removes real query signal
# (`mbi` and `nga` genuinely disambiguate in "tatimi mbi ..." constructions, so
# they stay).
STOPWORDS = {
    "i", "e", "te", "se", "ne", "me", "per", "nga", "dhe", "ose", "qe", "si",
    "a", "as", "by", "ku", "kur", "cdo", "ky", "kjo", "keto", "keta", "ai", "ajo",
    "eshte", "jane", "ishte", "ishin", "jete", "behet", "behen", "ka", "kane",
    "duhet", "mund", "do", "nuk", "s", "u", "ta", "te tilla", "tjeter", "tjera",
    "kete", "ketij", "kesaj", "tij", "saj", "tyre", "yne", "sipas", "lidhur",
    "vetem", "gjithe", "gjitha", "cila", "cilat", "cili", "cilit", "cilen",
}

# Longest-first: `tatimeve` must lose `-eve`, not `-e`.
# Nominal inflection (case / number / definiteness) plus the commonest verbal and
# participial endings.
#
# `te`, `se`, `je`, `ne` are deliberately absent. They look like suffixes but in
# practice they eat stem-final letters: the indefinite plural of `ligj` is `ligje`
# (stem + `e`), not `lig` + `je`. Including them split `ligje`/`ligji`,
# `subjekte`/`subjekt` and `pagese`/`pagesa` into separate tokens. The plural `-e`
# alone handles these correctly, and Albanian's `të`/`së` particles are removed as
# stopwords before stemming ever runs.
SUFFIXES = (
    "ndeve", "ndeje",
    "eve", "ave", "ive", "uar", "uara", "ues", "uese",
    "sve", "shme", "shem",
    "et", "at", "it", "in", "un", "ur", "es", "en",
    "ve", "sh", "ja",
    "e", "a", "i", "u",
)

# Bare `t` and `ut` are also absent, for the same reason. A large share of this
# corpus's vocabulary ends in `-t` in its base form — `subjekt`, `akt`, `objekt`,
# `dokument`, `produkt`, `kontribut`, `afat` — and stripping it split every one of
# them from its own inflected forms. The `-et`/`-at`/`-it` endings still cover the
# definite plural and genitive cases that `-t`/`-ut` were meant to catch; the cost
# is that `mikut`-style short genitives no longer reduce, which is rare here.

# Below this length, stripping destroys the word rather than normalising it.
# 3 rather than 4 so that short but load-bearing legal terms conflate: `neni`,
# `nene` -> `nen`. At 4 they stayed separate, which split citations of the single
# most common word in the corpus across three tokens.
MIN_STEM = 3


def fold(text: str) -> str:
    """ë -> e, ç -> c. Applied to corpus and query alike."""
    return text.translate(FOLD)


def stem(word: str) -> str:
    """Strip at most one inflectional suffix, conservatively.

    One suffix only: iterating would turn `tatimeve` into `tat`, collapsing it
    with unrelated words. A single pass conflates the inflected forms of a lemma
    without merging distinct lemmas, which is the tradeoff that matters for recall.
    """
    if len(word) <= MIN_STEM:
        return word
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            return word[: -len(suffix)]
    return word


def tokenise(text: str, *, use_stopwords: bool = True, use_stemming: bool = True) -> list[str]:
    """Full lexical pipeline: fold -> tokenise -> stop -> stem.

    The flags exist so the evaluation can isolate the contribution of each step
    rather than asserting that the whole pipeline helps.
    """
    tokens = TOKEN.findall(fold(text).lower())
    if use_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return [t for t in tokens if t]


def normalise_query(text: str) -> str:
    """Light touch for the dense arm: whitespace only, orthography preserved."""
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Diacritic restoration
# --------------------------------------------------------------------------
# Folding fixes the lexical arm, but the dense arm must keep correct orthography:
# multilingual encoders are trained on properly written text, so `per` and `për`
# do not embed identically. The fix is to repair the query before encoding rather
# than to damage the corpus.
#
# The mapping is learned from the corpus itself — no Albanian dictionary needed.
# Every token is folded to its ASCII skeleton, and the skeleton maps to whichever
# accented form the legislation actually uses most. `per` -> `për`, `eshte` ->
# `është`, `fature` -> `faturë`. Where the unaccented spelling is itself a real
# and common word, it is left alone.

import json as _json
from collections import Counter as _Counter
from pathlib import Path as _Path

# A skeleton is only rewritten if the accented form outnumbers the bare form by
# this factor. Prevents "corrections" of words that are genuinely spelled without
# diacritics (`akt`, `person`, `total`).
RESTORE_RATIO = 3.0


def learn_restoration(texts, min_count: int = 3) -> dict[str, str]:
    """Learn skeleton -> preferred accented form from corpus text."""
    by_skeleton: dict[str, _Counter] = {}
    for text in texts:
        for token in TOKEN.findall(text.lower()):
            skeleton = fold(token)
            by_skeleton.setdefault(skeleton, _Counter())[token] += 1

    mapping: dict[str, str] = {}
    for skeleton, forms in by_skeleton.items():
        if skeleton == "" or sum(forms.values()) < min_count:
            continue
        best, best_count = forms.most_common(1)[0]
        if best == skeleton:
            continue                      # corpus prefers the bare spelling
        bare_count = forms.get(skeleton, 0)
        if bare_count and best_count < bare_count * RESTORE_RATIO:
            continue                      # too close to call; leave it alone
        mapping[skeleton] = best
    return mapping


def save_restoration(mapping: dict[str, str], path: _Path) -> None:
    path.write_text(_json.dumps(mapping, ensure_ascii=False), encoding="utf-8")


def load_restoration(path: _Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return _json.loads(path.read_text(encoding="utf-8"))


def restore(text: str, mapping: dict[str, str]) -> str:
    """Rewrite an unaccented query into the orthography the corpus uses.

    Tokens the user already accented are left untouched, so a correctly typed
    question is never altered.
    """
    if not mapping:
        return text

    def swap(match: re.Match) -> str:
        token = match.group(0)
        if fold(token) != token:          # already carries ë or ç
            return token
        replacement = mapping.get(token.lower())
        if not replacement:
            return token
        return replacement.upper() if token.isupper() else (
            replacement.capitalize() if token[0].isupper() else replacement
        )

    return TOKEN.sub(swap, text)
