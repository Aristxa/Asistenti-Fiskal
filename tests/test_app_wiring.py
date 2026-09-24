"""Tests for the Space's Gradio wiring.

The bugs these lock in were not logic errors — the functions were correct and the
API client passed. They were wiring errors that only manifested in a browser, and
each one presented as the same thing: a Pyet button that did nothing.
"""

from __future__ import annotations

import inspect

import app as app_module


def test_respond_inputs_match_the_declared_components():
    """The click must pass exactly as many components as respond() takes.

    Regression: session identity was a gr.State passed in `inputs`. Gradio 5.9.1's
    frontend counts a State as a user-supplied argument, concludes "Too many
    arguments provided for the endpoint", and refuses to submit — so the click
    produced no network request at all. gradio_client omits State inputs, so the
    API kept working and the failure survived every API-level test.
    """
    params = list(inspect.signature(app_module.respond).parameters.values())
    # gr.Request is injected by Gradio, never wired through `inputs`.
    wired = [p for p in params if p.name != "request"]
    assert len(wired) == 5, (
        f"respond() takes {len(wired)} wired arguments; the UI passes 5"
    )
    assert params[-1].name == "request"


def test_no_state_component_is_used_as_an_input():
    """A gr.State in `inputs` is the specific shape that breaks the frontend."""
    source = inspect.getsource(app_module.build_search)
    assert "gr.State(" not in source


def test_session_key_falls_back_when_there_is_no_request():
    assert app_module.session_key(None) == "local"


def test_session_key_prefers_the_session_hash():
    class FakeRequest:
        session_hash = "abc123"
        client = None

    assert app_module.session_key(FakeRequest()) == "abc123"


def test_session_key_uses_the_client_host_without_a_session_hash():
    class FakeRequest:
        session_hash = None
        client = type("C", (), {"host": "10.0.0.1"})()

    assert app_module.session_key(FakeRequest()) == "10.0.0.1"


def test_examples_do_not_use_the_broken_dataset_component():
    """gr.Examples renders a Dataset, which is broken in Gradio 5.9.1.

    Regression: the frontend sent the sample value where Dataset.preprocess
    expects its index, so every example click raised "list indices must be
    integers or slices, not str" server-side and silently did nothing.
    """
    source = inspect.getsource(app_module.build_search)
    assert "gr.Examples(" not in source


def test_every_example_is_still_offered():
    source = inspect.getsource(app_module.build_search)
    assert "EXAMPLES" in source
    assert len(app_module.EXAMPLES) == 5


class _Hit:
    def __init__(self, rank, doc_id):
        self.rank = rank
        self.chunk = {"doc_id": doc_id}


def test_citation_markers_become_links():
    """A reader decides whether to trust a sentence at the marker, so the source
    has to be reachable there rather than only in the list at the end."""
    out = app_module.linkify_citations("Afati është 15 ditë [S1].", [_Hit(1, 946263)])
    assert "[[S1]](https://www.tatime.gov.al/shkarko.php?id=946263)" in out


def test_every_marker_in_a_sentence_is_linked():
    out = app_module.linkify_citations("Sipas [S1] dhe [S3].",
                                       [_Hit(1, 111), _Hit(3, 222)])
    assert "id=111" in out and "id=222" in out


def test_unknown_marker_is_left_as_plain_text():
    """A citation contract is not honoured by inventing a destination."""
    out = app_module.linkify_citations("Pohim [S9].", [_Hit(1, 111)])
    assert out == "Pohim [S9]."


def test_text_without_markers_is_unchanged():
    assert app_module.linkify_citations("Pa citime.", [_Hit(1, 1)]) == "Pa citime."
