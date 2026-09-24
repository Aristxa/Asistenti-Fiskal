"""Asistenti Fiskal — Hugging Face Space entry point.

Gradio app. Loads the committed indexes and encodes one question per request —
nothing is embedded at request time, which is what keeps the hosted cost near
zero regardless of tier.

The retrieval controls are exposed on purpose but kept collapsed: the thesis claim
is that retrieval decides correctness, so switching chunking strategy on one
question and watching the cited articles change is the argument made live. They are
labelled for a reader, not for a demonstration.
"""

from __future__ import annotations

import os
import re

import gradio as gr

from src.calc import business, payroll
from src.rag import extractive
from src.rag.answer import ask
from src.rag.retrieve import retrieve
from src.web.ratelimit import limiter

# The Space runs on ZeroGPU hardware, which refuses to start unless at least one
# entry point is GPU-decorated. This application does not need a GPU: the corpus
# is embedded offline and only the question is encoded at request time, on CPU.
# Free CPU Gradio Spaces require a PRO subscription on this account, so ZeroGPU is
# what is available.
#
# The decorator therefore goes on a probe that is never called, satisfying the
# platform check without any request consuming GPU quota. `spaces` exists only on
# Hugging Face infrastructure; locally the import fails and nothing changes.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_probe():
        """Exists only so ZeroGPU finds a GPU-decorated function at startup.

        It is never called. Decorating the request handler instead was a real
        mistake: every question then requested a GPU allocation for work that runs
        entirely on CPU, and the free daily quota drained until requests failed
        with "exceeded your ZeroGPU quota". Gradio swallowed that error, so the
        symptom was a button that did nothing — no output, no console error, no
        network request that returned anything useful.
        """
        return None

except ImportError:  # running locally, or off Hugging Face
    pass

# The two answering modes, named once so the UI and the handler cannot drift.
GENERATED = "E përmbledhur"
EXTRACTIVE = "Fjalë për fjalë nga ligji"

EXAMPLES = [
    "Kur duhet të regjistrohem si subjekt i TVSH-së?",
    "Cilat janë afatet për deklarimin e tatimit mbi të ardhurat?",
    "Si llogaritet kontributi i sigurimeve shoqërore për të vetëpunësuarit?",
    "Çfarë detyrimesh kam nëse mbyll biznesin?",
    "A duhet të lëshoj faturë fiskale për çdo shitje?",
]

def format_sources(hits) -> str:
    """Render the cited articles as reference entries.

    These are what distinguishes the system from a chatbot, so they are set like
    citations in a law report rather than as a bulleted list: the marker and the
    article number lead, the heading follows, the provision itself comes next, and
    the official document sits beneath in smaller type.

    The provision is printed in full because a link alone does not amount to
    verification. `shkarko.php?id=N` returns the whole act with no page anchor, so
    a reader who only gets the link still has to open the PDF and search it for the
    article — the same work the citation was meant to remove. Printing the text
    makes the claim checkable on the page, and the link stays for anyone who wants
    the original document behind it.
    """
    if not hits:
        return ""
    lines = ["### Burimet"]
    for hit in hits:
        chunk = hit.chunk
        url = f"https://www.tatime.gov.al/shkarko.php?id={chunk['doc_id']}"
        cite = chunk.get("cite") or chunk.get("label") or ""
        heading = (chunk.get("heading") or "").strip()
        # hit.rank, not a fresh counter. The answer cites [S5] by the position the
        # source held during retrieval, and the list is filtered to cited sources
        # only — renumbering here would make the prose and the list disagree.
        title = f"**[S{hit.rank}] {cite}**"
        if heading and heading != cite:
            title += f" — {heading}"
        lines.append(f"{title}{article_text(chunk)}  \n"
                     f"[Shkarko dokumentin zyrtar]({url})")
    return "\n\n".join(lines)


def article_text(chunk) -> str:
    """The provision, collapsed to one line and safe to drop into the panel.

    Escaped rather than trusted: 24 articles in the corpus contain a literal `<`,
    which the renderer would read as the start of a tag and swallow the text after
    it. Whitespace is collapsed for the same reason in reverse — the extracted text
    carries the PDF's own line breaks, and left alone its numbered paragraphs would
    be parsed as markdown lists.
    """
    text = " ".join((chunk.get("text") or "").split())
    if not text:
        return ""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"  \n<span class='law'>{text}</span>"


CITATION = re.compile(r"\[S(\d+)\]")


def linkify_citations(text: str, hits) -> str:
    """Turn every [Sn] marker in the prose into a link to that article's document.

    The marker is where a reader decides whether to trust a sentence, so that is
    where the source should be reachable. Leaving the links only in the list at
    the end means checking a claim requires scrolling away from it.

    Markers that do not correspond to a retrieved source are left as plain text
    rather than pointed anywhere — a citation contract is not honoured by inventing
    a destination.
    """
    urls = {
        hit.rank: f"https://www.tatime.gov.al/shkarko.php?id={hit.chunk['doc_id']}"
        for hit in hits
    }

    def replace(match: re.Match) -> str:
        rank = int(match.group(1))
        url = urls.get(rank)
        return f"[[S{rank}]]({url})" if url else match.group(0)

    return CITATION.sub(replace, text or "")


def render_lines(title: str, result, *, total: str = "") -> str:
    """Shared rendering for every calculator: figure, working, article."""
    def lek(x: float) -> str:
        return f"{x:,.0f}".replace(",", " ") + " lekë"

    rows = ["| Zëri | Shuma | Baza ligjore |", "|---|---:|---|"]
    for line in result.lines:
        detail = f"<br><span class='det'>{line.detail}</span>" if line.detail else ""
        rows.append(f"| {line.label}{detail} | {lek(line.amount)} | "
                    f"[{line.source.cite}]({line.source.url}) |")

    out = [f"### {title}", "", "\n".join(rows), ""]
    if total:
        out.append(total)
    for w in getattr(result, "warnings", []):
        out.append(f"> ⚠️ {w}")
    out.append("")
    out.append("_Llogaritje orientuese. Çdo shifër rrjedh nga dispozita e cituar "
               "përbri saj — hape dokumentin dhe verifikoje vetë._")
    return "\n".join(out)


def compute_vat(amount, direction):
    try:
        amount = float(amount or 0)
    except (TypeError, ValueError):
        return "Shkruaj një vlerë të vlefshme."
    if amount <= 0:
        return "Shkruaj vlerën e furnizimit."
    includes = direction.startswith("Vlera përfshin")
    result = business.vat(amount, amount_includes_vat=includes)
    return render_lines("Tatimi mbi vlerën e shtuar", result)


def compute_business(profit, turnover):
    try:
        profit = float(profit or 0)
        turnover = float(turnover) if turnover else None
    except (TypeError, ValueError):
        return "Shkruaj vlera të vlefshme."
    if profit <= 0:
        return "Shkruaj fitimin vjetor të tatueshëm."
    result = business.business_income_tax(profit, turnover=turnover)
    return render_lines("Tatimi mbi të ardhurat nga biznesi", result)


def format_payroll(result) -> str:
    """Render a payroll breakdown with the provision behind every figure.

    Same contract as the answer layer: a number the reader cannot trace is worth
    less than no number at all, so each line names its article and links the
    document it came from.
    """
    def lek(x: float) -> str:
        return f"{x:,.0f}".replace(",", " ") + " lekë"

    rows = ["| Zëri | Shuma | Baza ligjore |", "|---|---:|---|"]
    for line in result.lines:
        detail = f"<br><span class='det'>{line.detail}</span>" if line.detail else ""
        rows.append(
            f"| {line.label}{detail} | {lek(line.amount)} | "
            f"[{line.source.cite}]({line.source.url}) |"
        )

    out = [
        f"### Paga bruto: {lek(result.gross)}",
        "",
        "\n".join(rows),
        "",
        f"**Paga neto (në dorë): {lek(result.net)}**  ",
        f"Kosto totale për punëdhënësin: {lek(result.employer_total)}",
    ]
    if result.warnings:
        out.append("")
        for w in result.warnings:
            out.append(f"> ⚠️ {w}")
    out.append("")
    out.append(
        "_Kjo llogaritje është orientuese. Çdo shifër rrjedh nga dispozita e "
        "cituar përbri saj — hape dokumentin dhe verifikoje vetë përpara se ta "
        "përdorësh._"
    )
    return "\n".join(out)


def compute_payroll(gross, min_wage, max_wage):
    try:
        gross = float(gross or 0)
    except (TypeError, ValueError):
        return "Shkruaj një pagë bruto të vlefshme."
    if gross <= 0:
        return "Shkruaj pagën bruto mujore."
    if gross > 100_000_000:
        return "Shuma është jashtë çdo shkalle reale."
    result = payroll.calculate(
        gross,
        min_wage=float(min_wage) if min_wage else None,
        max_wage=float(max_wage) if max_wage else None,
    )
    return format_payroll(result)


def pending(question: str) -> tuple[str, str]:
    """What the page shows while the real answer is being produced.

    Returns immediately, so the wait stops looking like a dead button. The first
    question of a session is the slowest one — the prompt cache is cold and the
    encoder runs on two shared cores — and it is also the one a visitor judges the
    system by.

    The empty second value clears any previous source list, so the articles on
    screen always belong to the question being asked rather than the last one.
    """
    if not (question or "").strip():
        return "Shkruaj një pyetje.", ""
    return "_Po kërkoj në legjislacion …_", ""


def session_key(request: gr.Request | None) -> str:
    """Per-visitor identity for the rate limiter.

    Taken from the request rather than from a gr.State input. A State in the
    inputs list made the Gradio 5.9.1 frontend count it as a user-supplied
    argument, decide the call had "Too many arguments provided for the endpoint",
    and refuse to submit — so clicking Pyet sent no request at all. The API client
    was unaffected because it omits State inputs, which is exactly why the failure
    survived testing through gradio_client.
    """
    if request is None:
        return "local"
    session = getattr(request, "session_hash", None)
    if session:
        return str(session)
    client = getattr(request, "client", None)
    return str(getattr(client, "host", "anon"))


def respond(question: str, strategy: str, mode: str, k: int,
            answer_mode: str = GENERATED, request: gr.Request | None = None):
    """Answer a question, either by generating prose or by quoting the statute.

    The two modes trade fluency against guarantee. Generation reads better and
    needs a commercial API; extraction shows the legislator's own sentences and
    therefore cannot invent anything — the citation contract becomes a property of
    the method rather than an instruction the model is asked to obey.

    Extraction is also what makes the system self-sufficient: with no key, no
    credit, or no network to Anthropic, it still answers.
    """
    question = (question or "").strip()
    if not question:
        return "Shkruaj një pyetje.", ""
    if len(question) > 500:
        return "Pyetja është shumë e gjatë. Shkurtoje nën 500 karaktere.", ""

    hits = retrieve(question, strategy=strategy, mode=mode, k=k)

    if answer_mode == EXTRACTIVE:
        return (linkify_citations(extractive.format_answer(question, hits), hits),
                format_sources(hits))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "_Nuk ka çelës API, prandaj u përdor mënyra **nxjerrëse**._\n\n"
            + extractive.format_answer(question, hits)
        ), format_sources(hits)

    allowed, message = limiter.check(session_key(request))
    if not allowed:
        return message, ""

    try:
        result = ask(question, strategy=strategy, mode=mode, k=k)
    except RuntimeError as exc:
        # Falling back to extraction rather than to an error message. Retrieval
        # has already succeeded at this point, so the answer exists in the
        # statute — only the paraphrase is missing, and quoting is a complete
        # answer rather than an apology.
        return (
            f"⚠️ **Përmbledhja nuk u prodhua** ({exc})\n\n"
            f"_Më poshtë janë nenet përkatëse, fjalë për fjalë._\n\n"
            + linkify_citations(extractive.format_answer(question, hits), hits)
        ), format_sources(hits)

    return (linkify_citations(result.text, result.sources),
            format_sources(result.sources))



# Visual design.
#
# The register is legal, so the page borrows from print law reporting rather than
# from consumer software: a serif for anything that names the work, a humanist
# sans for reading, one deep accent colour used sparingly, and rules instead of
# boxes wherever a rule will do.
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_background_fill="white",
    block_background_fill_dark="*neutral_900",
    # Both variants stated. Gradio falls back to its own slate for the dark theme
    # if only the light fill is given, and the accent vanishes exactly where the
    # page is most likely to be read.
    button_primary_background_fill="#1f3a5f",
    button_primary_background_fill_hover="#2c4f7c",
    button_primary_background_fill_dark="#2f5c92",
    button_primary_background_fill_hover_dark="#3d72b0",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_primary_border_color="#1f3a5f",
    button_primary_border_color_dark="#2f5c92",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

.gradio-container { max-width: 860px !important; margin: 0 auto !important;
                    padding-top: 28px !important; }

#hdr { margin-bottom: 26px; }
#hdr h1 { font-family: 'Source Serif 4', Georgia, serif; font-size: 2.15rem;
          font-weight: 600; letter-spacing: -0.015em; line-height: 1.1;
          margin: 0 0 10px 0; }
#hdr .rule { height: 3px; width: 52px; background: #1f3a5f; margin: 0 0 14px 0; }
.dark #hdr .rule { background: #7da2cc; }
#hdr .sub { font-size: 1.02rem; line-height: 1.55; opacity: 0.82; margin: 0;
            max-width: 68ch; }
#hdr .src { font-size: 0.8rem; letter-spacing: 0.02em; opacity: 0.55;
            margin-top: 12px; }

#note { font-size: 0.84rem; line-height: 1.6; opacity: 0.75;
        background: rgba(31,58,95,0.045); border-left: 3px solid #1f3a5f;
        padding: 12px 16px; margin: 0 0 24px 0; border-radius: 0 4px 4px 0; }
.dark #note { background: rgba(125,162,204,0.08); border-left-color: #7da2cc; }

#ask textarea { font-size: 1.05rem !important; line-height: 1.5 !important; }
#go { font-size: 1rem !important; font-weight: 600 !important; }

#out:not(:has(p)):not(:has(h2)):not(:has(h3)) { display: none; }
#out { margin-top: 26px; font-size: 1rem; line-height: 1.68; }
#out h2 { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.3rem;
          font-weight: 600; margin: 0 0 .6em 0; padding-bottom: .3em;
          border-bottom: 1px solid var(--border-color-primary); }
#out h3 { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.08rem;
          font-weight: 600; margin: 1.5em 0 .5em 0; }
#out blockquote { border-left: 2px solid #1f3a5f; margin: .9em 0;
                  padding: .2em 0 .2em 16px; opacity: .92; }
.dark #out blockquote { border-left-color: #7da2cc; }

/* Citation markers inside the prose: reachable, but never loud enough to break
   the sentence they sit in. */
#out a[href*="shkarko.php"] { text-decoration: none; font-weight: 600;
    font-size: .82em; vertical-align: .2em; color: #1f3a5f;
    background: rgba(31,58,95,0.08); border-radius: 3px; padding: 1px 4px; }
#out a[href*="shkarko.php"]:hover { background: rgba(31,58,95,0.18); }
.dark #out a[href*="shkarko.php"] { color: #9dc0e8;
    background: rgba(125,162,204,0.15); }

#src:not(:has(p)) { display: none; }
#src { margin-top: 8px; font-size: .92rem; }
#src h3 { font-family: 'Source Serif 4', Georgia, serif; font-size: .82rem;
          text-transform: uppercase; letter-spacing: .1em; font-weight: 600;
          opacity: .6; margin: 0 0 .8em 0; }
#src hr { display: none; }
#src p { border-left: 2px solid var(--border-color-primary);
         padding: 2px 0 2px 14px; margin: 0 0 .85em 0; line-height: 1.5; }
#src p strong { color: #1f3a5f; font-weight: 600; }
.dark #src p strong { color: #7da2cc; }
/* The provision itself. Set smaller and quieter than the citation above it: it is
   there to be checked against the answer, not to compete with it for attention. */
#src .law { display: block; margin-top: 4px; font-size: .86rem;
            line-height: 1.55; opacity: .78; }
#src a { display: inline-block; margin-top: 5px; font-size: .86rem;
         font-weight: 600; text-decoration: none; color: #1f3a5f;
         border: 1px solid var(--border-color-primary); border-radius: 4px;
         padding: 3px 10px; }
#src a:hover { border-color: #1f3a5f; background: rgba(31,58,95,0.06); }
.dark #src a { color: #9dc0e8; }
.dark #src a:hover { border-color: #7da2cc; background: rgba(125,162,204,0.10); }

#suggest { font-size: .76rem; text-transform: uppercase; letter-spacing: .11em;
           opacity: .5; margin: 26px 0 10px 0; font-weight: 600; }
#suggest-row button { font-weight: 400 !important; font-size: .87rem !important;
                      background: transparent !important;
                      border: 1px solid var(--border-color-primary) !important;
                      color: var(--body-text-color) !important; opacity: .85; }
#suggest-row button:hover { border-color: #1f3a5f !important; opacity: 1; }

#calc table { width: 100%; border-collapse: collapse; margin: 14px 0; }
#calc th, #calc td { border-bottom: 1px solid var(--border-color-primary);
                     padding: 8px 6px; text-align: left; font-size: .93rem; }
#calc td:nth-child(2) { text-align: right; white-space: nowrap; }
#calc .det { font-size: .78rem; opacity: .55; }
#calc blockquote { border-left: 3px solid #b45309; background: rgba(180,83,9,.06);
                   padding: 8px 12px; margin: 10px 0; font-size: .86rem; }

#foot { font-size: .78rem; opacity: .5; margin-top: 34px; line-height: 1.6;
        border-top: 1px solid var(--border-color-primary); padding-top: 14px; }
#foot a { color: inherit; text-decoration: underline; }

/* Phones: one column, and controls big enough to hit with a thumb. */
@media (max-width: 640px) {
  .gradio-container { padding: 16px 12px 0 12px !important; }
  #hdr h1 { font-size: 1.6rem; }
  #hdr .sub { font-size: .95rem; }
  #go { width: 100%; min-height: 46px; }
  #suggest-row button { width: 100%; }
}
"""

HEADER = """<div id="hdr">
<h1>Asistenti Fiskal</h1>
<div class="rule"></div>
<p class="sub">Pyetje-përgjigje mbi legjislacionin tatimor, marrëdhëniet e punës dhe
standardet kombëtare të kontabilitetit. Çdo pohim lidhet me nenin që e mbështet.</p>
<p class="src">313 dokumente zyrtare &nbsp;·&nbsp; Drejtoria e Përgjithshme e Tatimeve
&nbsp;·&nbsp; Këshilli Kombëtar i Kontabilitetit
&nbsp;·&nbsp; Inspektorati Shtetëror i Punës</p>
</div>"""

NOTE = """<div id="note">
Ky nuk është këshillë tatimore. Sistemi tregon çfarë thotë ligji dhe ku, por nuk
llogarit detyrime konkrete. Legjislacioni ndryshon — verifikoni në burimin zyrtar
përpara se të veproni.
</div>"""

FOOTER = """<div id="foot">
Mikrotezë · Universiteti i Tiranës, Fakulteti i Ekonomisë<br>
Korpusi: <a href="https://www.tatime.gov.al">tatime.gov.al</a> ·
Këshilli Kombëtar i Kontabilitetit · Inspektorati Shtetëror i Punës
</div>"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Asistenti Fiskal", theme=THEME, css=CSS) as demo:
        gr.HTML(HEADER)
        gr.HTML(NOTE)

        with gr.Tab("Pyet ligjin"):
            build_search()

        with gr.Tab("Llogaritësit"):
            build_calculators()

        gr.HTML(FOOTER)

    return demo


def build_calculators() -> None:
    """Three calculators, one rule: every figure names the provision behind it.

    Deliberately narrow. Each computes only what the corpus can support and cites
    it; where a parameter lives in an act the corpus does not contain, or where a
    step is a legal classification rather than arithmetic, it says so instead of
    guessing.
    """
    gr.Markdown(
        "**Çdo shifër shoqërohet me nenin nga i cili rrjedh.** Aty ku një "
        "parametër caktohet me akt që nuk gjendet në korpus, ose ku hapi është "
        "klasifikim ligjor e jo aritmetikë, mjeti e deklaron në vend që ta "
        "hamendësojë."
    )

    with gr.Tab("Paga"):
        build_payroll_calculator()

    with gr.Tab("TVSH"):
        gr.Markdown("Ndarja e një furnizimi në vlerë të tatueshme dhe TVSH.")
        with gr.Row():
            vat_amount = gr.Number(label="Vlera (lekë)", value=100_000)
            vat_dir = gr.Radio(
                ["Vlera është pa TVSH", "Vlera përfshin TVSH-në"],
                value="Vlera është pa TVSH", label="Si është dhënë vlera",
            )
        vat_btn = gr.Button("Llogarit", variant="primary")
        vat_out = gr.Markdown(elem_id="calc")
        vat_btn.click(compute_vat, inputs=[vat_amount, vat_dir], outputs=vat_out)
        vat_amount.submit(compute_vat, inputs=[vat_amount, vat_dir], outputs=vat_out)

    with gr.Tab("Biznesi"):
        gr.Markdown(
            "Tatimi vjetor mbi të ardhurat nga biznesi për tregtarë individualë "
            "dhe të vetëpunësuar. **Qarkullimi** jepet veçmas nga fitimi, sepse "
            "norma 0% varet nga qarkullimi, jo nga fitimi."
        )
        with gr.Row():
            profit = gr.Number(label="Fitimi vjetor i tatueshëm (lekë)",
                               value=3_000_000)
            turnover = gr.Number(label="Qarkullimi vjetor (lekë)", value=8_000_000)
        biz_btn = gr.Button("Llogarit", variant="primary")
        biz_out = gr.Markdown(elem_id="calc")
        biz_btn.click(compute_business, inputs=[profit, turnover], outputs=biz_out)
        profit.submit(compute_business, inputs=[profit, turnover], outputs=biz_out)


def build_payroll_calculator() -> None:
    """Gross-to-net for one month, with the article behind every figure.

    Deliberately not a general tax calculator. It computes what the corpus can
    support and cites it; where a parameter lives in a decision the corpus does
    not contain — the minimum and maximum contribution wage — it asks rather than
    assumes.
    """
    gr.Markdown(
        "Llogarit kontributet, tatimin dhe pagën neto për një muaj. "
        "**Çdo shifër shoqërohet me nenin nga i cili rrjedh.**\n\n"
        "Kufiri minimal dhe maksimal i pagës për kontribute caktohen me VKM të "
        "veçantë që nuk gjendet në korpus — lëri bosh ose plotësoji vetë."
    )
    with gr.Row():
        gross = gr.Number(label="Paga bruto mujore (lekë)", value=85_000)
        min_wage = gr.Number(label="Paga minimale (opsionale)", value=None)
        max_wage = gr.Number(label="Paga maksimale (opsionale)", value=None)
    calc_btn = gr.Button("Llogarit", variant="primary")
    payroll_out = gr.Markdown(elem_id="calc")

    controls = [gross, min_wage, max_wage]
    calc_btn.click(compute_payroll, inputs=controls, outputs=payroll_out)
    gross.submit(compute_payroll, inputs=controls, outputs=payroll_out)


def build_search() -> None:
        with gr.Row(equal_height=True):
            question = gr.Textbox(
                placeholder="Shkruani pyetjen tuaj — p.sh. Kur duhet të regjistrohem për TVSH?",
                lines=2, scale=5, elem_id="ask", show_label=False,
            )
            submit = gr.Button("Kërko", variant="primary", scale=1, elem_id="go")

        answer = gr.Markdown(elem_id="out")
        sources = gr.Markdown(elem_id="src")

        gr.HTML('<p id="suggest">Pyetje të sugjeruara</p>')
        with gr.Row(elem_id="suggest-row"):
            for example in EXAMPLES:
                gr.Button(example, size="sm", variant="secondary").click(
                    lambda text=example: text, outputs=question
                )

        # The parameter panel sits at the foot of the page, after the answer and
        # the suggestions. A taxpayer has no reason to choose between semantic and
        # lexical retrieval and should meet the question box first; a commission
        # looking for the Chapter III comparison will find the panel when it wants
        # it. Defaults are the configuration the measurement selected.
        with gr.Accordion("Parametrat e kërkimit", open=False):
            answer_mode = gr.Radio(
                [GENERATED, EXTRACTIVE], value=GENERATED,
                label="Forma e përgjigjes",
            )
            with gr.Row():
                strategy = gr.Radio(
                    [("Sipas nenit", "article"), ("Dritare fikse", "fixed")],
                    value="article", label="Ndarja e dokumentit",
                )
                mode = gr.Radio(
                    # dense by default, and that is a measured choice: on the
                    # benchmark it finds the controlling article 47% of the time
                    # against hybrid's 38%.
                    [("Semantik", "dense"), ("Semantik + fjalëkyç", "hybrid"),
                     ("Fjalëkyç", "bm25")],
                    value="dense", label="Mënyra e kërkimit",
                )
            k = gr.Slider(1, 10, value=8, step=1, label="Numri i neneve të marra")

        # The per-visitor identity comes from gr.Request inside respond(), not from
        # a gr.State input — see session_key().
        inputs = [question, strategy, mode, k, answer_mode]
        # An instant first step, then the slow one. Both entry points chain the
        # same pair.
        #
        # Bound straight to respond(), the click produced nothing visible for the
        # length of a generation: the answer component starts empty, an empty
        # Markdown renders no box, and Gradio's pending animation therefore has
        # nothing to attach itself to. The button looked broken — which is exactly
        # the symptom F11 wasted a week on, and a demonstration is the worst place
        # to meet it a second time.
        for trigger in (submit.click, question.submit):
            trigger(pending, inputs=[question], outputs=[answer, sources]).then(
                respond, inputs=inputs, outputs=[answer, sources])


def warm_up() -> None:
    """Load the encoder and both indexes before the app accepts a request.

    Left lazy, the first click pays for a 2.3 GB model download plus index load.
    On ZeroGPU that click runs inside a bounded GPU allocation, so it expires
    before returning and the button appears simply not to work — no error, no
    output. Paying the cost at startup makes boot slower and every query fast.
    """
    from src.rag.retrieve import get_encoder, get_index

    print("po ngarkohet modeli dhe indekset ...")
    get_encoder()
    for strategy in ("article", "fixed"):
        index = get_index(strategy)
        print(f"  {strategy}: {len(index.chunks):,} copëza")
    print("gati")


if __name__ == "__main__":
    warm_up()
    # ssr_mode=False on purpose. Gradio 5 turns server-side rendering on by
    # default on Spaces and flags it experimental in its own startup line. With it
    # on, the page rendered correctly but the Pyet button bound no handler: a click
    # produced no network request, no console error and no output — the failure
    # looked like a dead button rather than a framework mode.
    build().launch(ssr_mode=False)
