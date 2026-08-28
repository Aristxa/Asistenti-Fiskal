"""Asistenti Fiskal — Hugging Face Space entry point.

Gradio app. Loads the committed indexes and encodes one question per request —
nothing is embedded at request time, which is what keeps the hosted cost near
zero regardless of tier.

The retrieval controls are exposed on purpose. The thesis claim is that retrieval
decides correctness, so the demo lets anyone switch chunking strategy and
retrieval mode on the same question and watch the cited articles change. At a
defense that is the argument, made live, instead of a table on a slide.
"""

from __future__ import annotations

import os
import uuid

import gradio as gr

from src.rag.answer import ask
from src.rag.retrieve import retrieve
from src.web.ratelimit import limiter

# The Space runs on ZeroGPU hardware, which refuses to start unless at least one
# entry point is GPU-decorated. This application does not need a GPU — the corpus
# is embedded offline and only the question is encoded here — but the platform
# requires the decorator, and free CPU Gradio Spaces need a PRO subscription on
# this account. Since a GPU is attached anyway, the encoder picks it up
# (src/rag/retrieve.py chooses its device at load time) rather than idling it.
#
# `spaces` only exists on Hugging Face infrastructure, so locally this degrades
# to a no-op and the app runs unchanged.
try:
    import spaces

    gpu = spaces.GPU(duration=60)
except ImportError:  # running locally
    def gpu(fn):
        return fn

EXAMPLES = [
    "Kur duhet të regjistrohem si subjekt i TVSH-së?",
    "Cilat janë afatet për deklarimin e tatimit mbi të ardhurat?",
    "Si llogaritet kontributi i sigurimeve shoqërore për të vetëpunësuarit?",
    "Çfarë detyrimesh kam nëse mbyll biznesin?",
    "A duhet të lëshoj faturë fiskale për çdo shitje?",
]

DISCLAIMER = """
**Asistenti Fiskal** përgjigjet vetëm mbi bazën e legjislacionit tatimor shqiptar
të publikuar në [tatime.gov.al](https://www.tatime.gov.al), me citime te neni përkatës.

⚠️ Ky nuk është këshillë tatimore. Ligji tatimor ndryshon shpesh — verifikoni gjithmonë
në burimin zyrtar përpara se të veproni.
"""


def format_sources(hits) -> str:
    if not hits:
        return ""
    lines = ["\n---\n### Burimet"]
    for hit in hits:
        chunk = hit.chunk
        url = f"https://www.tatime.gov.al/shkarko.php?id={chunk['doc_id']}"
        # hit.rank, not a fresh counter. The answer cites [S5] by the position the
        # source held during retrieval, and the list is filtered to cited sources
        # only — renumbering here would make the prose and the list disagree.
        lines.append(f"**[S{hit.rank}]** {hit.citation}  \n[Dokumenti zyrtar]({url})")
    return "\n\n".join(lines)


@gpu
def respond(question: str, strategy: str, mode: str, k: int, session_id: str):
    question = (question or "").strip()
    if not question:
        return "Shkruaj një pyetje.", ""
    if len(question) > 500:
        return "Pyetja është shumë e gjatë. Shkurtoje nën 500 karaktere.", ""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        # Retrieval works without a key; only generation needs one. Showing the
        # retrieved articles keeps the Space useful (and demonstrable) regardless.
        hits = retrieve(question, strategy=strategy, mode=mode, k=k)
        return ("_Nuk ka çelës API — po shfaqen vetëm nenet e gjetura, pa përgjigje "
                "të gjeneruar._", format_sources(hits))

    allowed, message = limiter.check(session_id)
    if not allowed:
        return message, ""

    try:
        result = ask(question, strategy=strategy, mode=mode, k=k)
    except RuntimeError as exc:
        return f"Gabim: {exc}", ""

    return result.text, format_sources(result.sources)


def build() -> gr.Blocks:
    with gr.Blocks(title="Asistenti Fiskal", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧾 Asistenti Fiskal")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            question = gr.Textbox(
                label="Pyetja jote",
                placeholder="p.sh. Kur duhet të regjistrohem për TVSH?",
                lines=2,
                scale=4,
            )
            submit = gr.Button("Pyet", variant="primary", scale=1)

        with gr.Accordion("Cilësimet e kërkimit (për demonstrim)", open=False):
            gr.Markdown(
                "Këto kontrolle ekzistojnë për të treguar tezën: **cilësia e "
                "kërkimit përcakton saktësinë e përgjigjes.** Ndrysho strategjinë "
                "dhe shiko si ndryshojnë nenet e cituara për të njëjtën pyetje."
            )
            with gr.Row():
                strategy = gr.Radio(
                    ["article", "fixed"], value="article",
                    label="Ndarja e dokumentit",
                    info="article = një copë për çdo nen; fixed = copa me gjatësi fikse",
                )
                mode = gr.Radio(
                    ["hybrid", "dense", "bm25"], value="hybrid",
                    label="Mënyra e kërkimit",
                    info="hybrid = semantik + fjalëkyç",
                )
                k = gr.Slider(1, 10, value=5, step=1, label="Sa nene të merren")

        answer = gr.Markdown(label="Përgjigja")
        sources = gr.Markdown()

        gr.Examples(examples=EXAMPLES, inputs=question)

        # Per-browser-session id, so the rate limit is per visitor rather than global.
        session_id = gr.State(lambda: uuid.uuid4().hex)

        inputs = [question, strategy, mode, k, session_id]
        submit.click(respond, inputs=inputs, outputs=[answer, sources])
        question.submit(respond, inputs=inputs, outputs=[answer, sources])

    return demo


if __name__ == "__main__":
    build().launch()
