"""Review screen for extracted obligations.

The question review needed the author to write something for every row. This one
does not: the extractor has already produced an answer and the only judgement
required is whether that answer is right. So the whole interaction is two buttons,
and 60 rows is a short sitting rather than an afternoon.

What is being measured is the precision of a rule-based extractor — a real number
for Chapter III, which currently describes extracting 2,077 obligations without
saying how often they are correct.

    .venv/Scripts/python scripts/review_obligations_app.py
    (or double-click Detyrimet.bat)
"""

from __future__ import annotations

import csv
from pathlib import Path

import gradio as gr

WORKBOOK = Path("eval/gold/detyrimet_rishikim.csv")

COL_ID, COL_VERDICT, COL_NOTE = 0, 1, 2
COL_DOMAIN, COL_CITE, COL_SUBJECT = 3, 4, 5
COL_DEADLINE, COL_RATE, COL_PENALTY, COL_TEXT = 6, 7, 8, 9

# A handful of articles run to tens of thousands of characters. Showing one whole
# freezes the browser and the reviewer cannot find the relevant passage anyway, so
# very long articles are shown from the start with an explicit marker. Nothing is
# silently cut: the marker says the text continues, which is the difference
# between an informed judgement and a blind one.
DISPLAY_LIMIT = 12000


def load() -> tuple[list[list[str]], int]:
    if not WORKBOOK.exists():
        raise SystemExit(
            f"nuk ka {WORKBOOK} — ekzekuto "
            f"`python -m src.extract.review_obligations export`"
        )
    with WORKBOOK.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    header = next(i for i, r in enumerate(rows) if r and r[0].strip() == "obligation_id")
    return rows, header


def save(rows: list[list[str]]) -> None:
    with WORKBOOK.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle, delimiter=";").writerows(rows)


class Session:
    """Workbook held in memory; every judgement is flushed to disk immediately."""

    def __init__(self) -> None:
        self.rows, self.header = load()
        self.data = self.rows[self.header + 1:]
        self.data = [r for r in self.data if r and r[COL_ID].strip()]
        self.index = self._first_unjudged()

    def _first_unjudged(self) -> int:
        for i, row in enumerate(self.data):
            if not (row[COL_VERDICT] or "").strip():
                return i
        return 0

    @property
    def total(self) -> int:
        return len(self.data)

    @property
    def done(self) -> int:
        return sum(1 for r in self.data if (r[COL_VERDICT] or "").strip())

    @property
    def correct(self) -> int:
        return sum(1 for r in self.data
                   if (r[COL_VERDICT] or "").strip().lower() == "po")

    def current(self) -> list[str]:
        return self.data[self.index]

    def move(self, delta: int) -> None:
        self.index = max(0, min(self.total - 1, self.index + delta))

    def record(self, verdict: str, note: str) -> None:
        row = self.current()
        row[COL_VERDICT] = verdict
        row[COL_NOTE] = (note or "").strip()
        save(self.rows)


session = Session()


def field(label: str, value: str) -> str:
    value = (value or "").strip()
    return f"- **{label}:** {value}" if value else f"- **{label}:** _(asnjë)_"


def render() -> tuple:
    row = session.current()
    judged = session.done
    rate = f" · saktësia deri tani **{100 * session.correct / judged:.0f}%**" if judged else ""
    progress = (f"### Detyrimi {session.index + 1} nga {session.total} · "
                f"**{judged} të vlerësuara**{rate}")

    text = row[COL_TEXT] or ""
    if len(text) > DISPLAY_LIMIT:
        text = (text[:DISPLAY_LIMIT]
                + f"\n\n---\n\n_[Neni vazhdon edhe {len(row[COL_TEXT]) - DISPLAY_LIMIT:,} "
                  f"karaktere të tjera — i gjatë jashtëzakonisht.]_")

    article = f"**Neni {row[COL_CITE]}** · {row[COL_DOMAIN]}\n\n---\n\n{text}"
    extracted = "\n".join([
        "#### Çfarë nxori sistemi",
        field("Subjekti", row[COL_SUBJECT]),
        field("Afati", row[COL_DEADLINE]),
        field("Norma", row[COL_RATE]),
        field("Sanksioni", row[COL_PENALTY]),
        "",
        "_A e përshkruan ky rresht vërtet një detyrim, dhe a janë të sakta "
        "atributet e mësipërme sipas nenit majtas?_",
    ])
    verdict = (row[COL_VERDICT] or "").strip().lower()
    status = {"po": "✅ Saktë", "jo": "❌ Gabim"}.get(verdict, "— e pavlerësuar")
    return progress, article, extracted, row[COL_NOTE], status


def judge(verdict: str, note: str) -> tuple:
    session.record(verdict, note)
    session.move(1)
    return render()


def mark_correct(note: str) -> tuple:
    return judge("po", note)


def mark_wrong(note: str) -> tuple:
    return judge("jo", note)


def skip(note: str) -> tuple:
    session.record("", note)
    session.move(1)
    return render()


def go_back(note: str) -> tuple:
    session.move(-1)
    return render()


def build() -> gr.Blocks:
    with gr.Blocks(title="Rishikimi i detyrimeve", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Rishikimi i detyrimeve të nxjerra")
        gr.Markdown(
            "Lexo nenin majtas dhe gjyko nëse atributet djathtas janë të sakta. "
            "Çdo vendim ruhet menjëherë — mund ta mbyllësh dritaren në çdo moment "
            "dhe të vazhdosh më vonë nga i njëjti vend."
        )
        # The rule is on screen, not in someone's memory. Sixty judgements made
        # against a drifting standard produce a number that means nothing, and the
        # drift is invisible afterwards.
        gr.Markdown(
            "> **Rregulli i gjykimit — saktësia, jo plotësia.**\n"
            "> \n"
            "> - ✅ **Saktë** — rreshti përshkruan vërtet një detyrim **dhe** çdo fushë "
            "*e plotësuar* është e vërtetë sipas nenit.\n"
            "> - ❌ **Gabim** — nuk është detyrim i vërtetë, **ose** një fushë e "
            "plotësuar thotë diçka që neni nuk e thotë.\n"
            "> \n"
            "> Një fushë **bosh nuk është gabim** — është mungesë. Nëse neni ka një "
            "afat ose subjekt që sistemi nuk e kapi, kjo mbetet ✅, por **shkruaje te "
            "shënimet** (p.sh. *“subjekti mungon; norma 3% e humbur”*). Kështu "
            "mungesat maten veçmas nga gabimet."
        )

        progress = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                article = gr.Markdown()
            with gr.Column(scale=2):
                extracted = gr.Markdown()
                status = gr.Markdown()
                note = gr.Textbox(label="Shënime (opsionale)", lines=2,
                                  placeholder="p.sh. afati i saktë është 30 ditë, jo 15")
                with gr.Row():
                    ok_btn = gr.Button("✅ Saktë", variant="primary")
                    bad_btn = gr.Button("❌ Gabim", variant="stop")
                with gr.Row():
                    back_btn = gr.Button("← Mbrapa")
                    skip_btn = gr.Button("Kaloje")

        outputs = [progress, article, extracted, note, status]
        ok_btn.click(mark_correct, inputs=[note], outputs=outputs)
        bad_btn.click(mark_wrong, inputs=[note], outputs=outputs)
        skip_btn.click(skip, inputs=[note], outputs=outputs)
        back_btn.click(go_back, inputs=[note], outputs=outputs)
        demo.load(render, outputs=outputs)

    return demo


if __name__ == "__main__":
    build().launch(ssr_mode=False, inbrowser=True)
