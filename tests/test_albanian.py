"""Tests for Albanian text processing.

These lock in behaviour that was arrived at by measurement, not by intuition —
every case below corresponds to a regression that actually occurred while tuning
the stemmer (docs/FINDINGS.md F8). Without them a future "simplification" of the
suffix list would silently split the corpus's most common words again.
"""

from __future__ import annotations

import pytest

from src.index.albanian import (
    fold,
    learn_restoration,
    restore,
    stem,
    tokenise,
)

# Each group is one lemma; every form must reduce to the same stem.
INFLECTION_GROUPS = {
    "tatim": ["tatim", "tatimi", "tatimit", "tatimin", "tatime", "tatimet", "tatimeve"],
    "deklarim": ["deklarim", "deklarimi", "deklarimit", "deklarimin", "deklarime"],
    "qarkullim": ["qarkullim", "qarkullimi", "qarkullimit", "qarkullimin"],
    "ligj": ["ligj", "ligji", "ligjit", "ligje", "ligjet", "ligjeve"],
    "nen": ["nen", "neni", "nenit", "nene", "nenet"],
    "fature": ["fature", "fatura", "faturat", "faturave", "faturen"],
    "person": ["person", "personi", "personit", "persona", "personat", "personave"],
    "sherbim": ["sherbim", "sherbimi", "sherbimit", "sherbime", "sherbimet"],
    "subjekt": ["subjekt", "subjekti", "subjektit", "subjekte", "subjektet"],
    "pagese": ["pagese", "pagesa", "pagesat", "pagesave", "pagesen"],
    "afat": ["afat", "afati", "afatit", "afate", "afatet"],
    "kontribut": ["kontribut", "kontributi", "kontributit", "kontribute"],
    "dokument": ["dokument", "dokumenti", "dokumentit", "dokumente", "dokumentet"],
    "norme": ["norme", "norma", "normat", "normave", "normen"],
    "vlere": ["vlere", "vlera", "vleres", "vleren", "vlerat"],
}

# Distinct concepts that must not collapse into one another.
DISTINCT = ["tatim", "takse", "ligj", "lloj", "vend", "viti", "akt", "afat",
            "person", "pagese", "nen", "biznes"]


@pytest.mark.parametrize("lemma,forms", INFLECTION_GROUPS.items())
def test_inflections_conflate(lemma, forms):
    stems = {stem(fold(form)) for form in forms}
    assert len(stems) == 1, f"{lemma} split into {sorted(stems)}"


def test_distinct_lemmas_do_not_collide():
    stems = [stem(fold(word)) for word in DISTINCT]
    assert len(set(stems)) == len(stems), f"collision among {stems}"


def test_min_stem_keeps_neni_together():
    """Regression: at MIN_STEM=4 these were three separate tokens."""
    assert stem("neni") == stem("nene") == stem("nen") == "nen"


@pytest.mark.parametrize("word,expected", [
    ("ligje", "ligj"),        # plural -e, not -je
    ("subjekte", "subjekt"),  # plural -e, not -te
    ("pagese", "pages"),      # -e, not -se
])
def test_two_letter_suffixes_do_not_eat_stems(word, expected):
    """Regression: `te`/`se`/`je`/`ne` in the suffix list truncated real stems."""
    assert stem(fold(word)) == expected


@pytest.mark.parametrize("word", ["subjekt", "akt", "objekt", "dokument",
                                  "produkt", "kontribut", "afat"])
def test_base_words_ending_in_t_survive(word):
    """Regression: stripping bare `-t` split these from their own inflections."""
    assert stem(fold(word)) == word


def test_folding_makes_tokenisation_diacritic_blind():
    accented = "Sa është kufiri i qarkullimit vjetor për TVSH?"
    plain = "Sa eshte kufiri i qarkullimit vjetor per TVSH?"
    assert tokenise(accented) == tokenise(plain)


def test_stopwords_removed_but_content_kept():
    tokens = tokenise("Cilat janë afatet për deklarimin e tatimit?")
    assert "afat" in tokens and "deklarim" in tokens and "tatim" in tokens
    assert "per" not in tokens and "jane" not in tokens


def test_tokenise_flags_isolate_each_step():
    text = "tatimeve për deklarimin"
    assert "per" in tokenise(text, use_stopwords=False, use_stemming=False)
    assert "tatimeve" in tokenise(text, use_stopwords=True, use_stemming=False)
    assert "tatim" in tokenise(text, use_stopwords=True, use_stemming=True)


class TestRestoration:
    corpus = [
        "për tatimin mbi vlerën e shtuar është e detyrueshme",
        "për çdo faturë të lëshuar për vlerën e furnizimit",
        "personi i tatueshëm është subjekt për këtë akt",
        "për vlerën e faturës çdo person duhet të deklarojë",
    ]

    def mapping(self):
        return learn_restoration(self.corpus, min_count=1)

    def test_learns_accented_forms(self):
        m = self.mapping()
        assert m.get("per") == "për"
        assert m.get("cdo") == "çdo"
        assert m.get("eshte") == "është"

    def test_leaves_genuinely_unaccented_words_alone(self):
        m = self.mapping()
        for word in ("person", "akt", "subjekt", "tatimin"):
            assert word not in m, f"{word} should not be rewritten"

    def test_restore_rewrites_plain_query(self):
        m = self.mapping()
        assert restore("per cdo fature", m) == "për çdo faturë"

    def test_restore_leaves_already_accented_text_untouched(self):
        m = self.mapping()
        text = "për çdo faturë"
        assert restore(text, m) == text

    def test_restore_preserves_capitalisation(self):
        m = self.mapping()
        assert restore("Per", m) == "Për"

    def test_empty_mapping_is_identity(self):
        assert restore("per cdo fature", {}) == "per cdo fature"
