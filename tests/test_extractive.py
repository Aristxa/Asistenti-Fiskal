"""Tests for the extractive answering mode.

The property worth protecting here is not answer quality but faithfulness: every
sentence shown must appear verbatim in the retrieved article. That is what makes
this mode unable to hallucinate, and it is the whole reason it exists alongside
the generated one.
"""

from __future__ import annotations

import pytest

from src.rag import extractive
from src.rag.retrieve import Hit

ARTICLE = (
    "Neni 92 Pushimet vjetore. Kohëzgjatja e pushimeve vjetore është jo më pak se "
    "22 ditë pune gjatë vitit të punës në vazhdim. Pushimet vjetore nuk përfshijnë "
    "ditët e festave zyrtare. Punëdhënësi cakton datat e pushimeve duke marrë "
    "parasysh kërkesat e punëmarrësit."
)


def make_hit(text: str = ARTICLE, rank: int = 1) -> Hit:
    return Hit(
        chunk={"doc_id": 1, "cite": "Neni 92", "heading": "Pushimet vjetore",
               "text": text, "chars": len(text)},
        score=0.9, rank=rank,
    )


def test_every_sentence_is_verbatim_from_the_article():
    """The guarantee the mode exists for: nothing is rewritten."""
    passages = extractive.select("Sa ditë pushimi vjetor më takojnë?", [make_hit()])
    assert passages
    for sentence in passages[0].sentences:
        assert sentence in " ".join(ARTICLE.split())


def test_it_finds_the_sentence_that_answers_the_question():
    passages = extractive.select("Sa ditë pushimi vjetor më takojnë?", [make_hit()])
    joined = " ".join(passages[0].sentences)
    assert "22 ditë pune" in joined


def test_sentences_keep_the_statute_order():
    """Legal provisions state conditions before consequences.

    Returning them in score order would change what the provision says, so
    selection ranks but presentation restores the original sequence.
    """
    text = (
        "Neni 1 Kushti. Nëse tatimpaguesi kalon kufirin e regjistrimit, ai duhet "
        "të njoftojë organin tatimor. Regjistrimi kryhet brenda 15 ditëve nga data "
        "e kalimit të kufirit."
    )
    passages = extractive.select("Kur duhet regjistrimi brenda 15 ditëve?",
                                 [make_hit(text)], per_article=2)
    chosen = passages[0].sentences
    positions = [" ".join(text.split()).index(s) for s in chosen]
    assert positions == sorted(positions)


def test_numbered_legal_references_do_not_split_sentences():
    """Regression guard: "nr. 9125, datë 29.7.2003" is not three sentences."""
    text = ("Neni 92 (Ndryshuar me ligjin nr. 9125, datë 29.7.2003). Kohëzgjatja e "
            "pushimeve vjetore është jo më pak se 22 ditë pune në vit.")
    sentences = extractive.split_sentences(text)
    assert any("22 ditë pune" in s for s in sentences)
    assert not any(s.strip() in {"9125,", "datë", "29.7.2003)."} for s in sentences)


def test_short_fragments_are_dropped():
    assert extractive.split_sentences("Neni 5. Po. Jo.") == []


def test_no_matching_sentence_says_so_rather_than_inventing_one():
    passages = extractive.select("Sa kushton një apartament në Tiranë?",
                                 [make_hit()], min_score=0.9)
    assert passages == []
    text = extractive.format_answer("Sa kushton një apartament në Tiranë?",
                                    [make_hit()])
    assert "Nuk gjeta" in text or "22 ditë" in text


def test_answer_carries_the_citation_and_the_document_link():
    text = extractive.format_answer("Sa ditë pushimi vjetor më takojnë?",
                                    [make_hit()])
    assert "Neni 92" in text
    assert "[S1]" in text
    assert "shkarko.php?id=1" in text


def test_concrete_facts_are_preferred_between_equal_matches():
    plain = extractive.score_sentence(
        "Pushimet vjetore rregullohen me kontratë kolektive.", {"pushim"})
    concrete = extractive.score_sentence(
        "Pushimet vjetore janë jo më pak se 22 ditë pune.", {"pushim"})
    assert concrete > plain
