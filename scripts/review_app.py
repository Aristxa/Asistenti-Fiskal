"""A one-question-at-a-time review screen for the benchmark.

Reviewing 120 articles in a spreadsheet means scrolling 900-character cells in a
column, losing your place, and re-reading the same text three times. The judgement
being asked for is small — *what would someone actually ask about this?* — and the
format was making it feel large.

This shows one article, with a box for the question and one button. Progress is
saved after every row, so it can be closed and reopened at any point. It writes
the same `rishikim.csv` the spreadsheet flow uses, so the two are interchangeable
and nothing downstream changes.

    python scripts/review_app.py

Then open the printed address. Runs entirely locally; nothing is sent anywhere.
"""

from __future__ import annotations

import csv
from pathlib import Path

import gradio as gr

WORKBOOK = Path("eval/gold/rishikim.csv")
COLUMNS_START = "id"

# Column positions in the workbook, matching src/eval/review.py
COL_ID, COL_CATEGORY, COL_QUESTION = 0, 1, 2
COL_GOLD, COL_DOC, COL_ANSWERABLE, COL_REVIEWED, COL_NOTES = 3, 4, 5, 6, 7
COL_HEADING, COL_TEXT = 8, 9


def load() -> tuple[list[list[str]], int]:
    if not WORKBOOK.exists():
        raise SystemExit(
            f"nuk ka {WORKBOOK} — ekzekuto `python -m src.eval.review export`"
        )
    with WORKBOOK.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    header = next(i for i, r in enumerate(rows) if r and r[0].strip() == COLUMNS_START)
    return rows, header


def save(rows: list[list[str]]) -> None:
    with WORKBOOK.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle, delimiter=";").writerows(rows)


class Session:
    """Holds the workbook in memory; every edit is flushed to disk immediately."""

    def __init__(self) -> None:
        self.rows, self.header = load()
        self.data = self.rows[self.header + 1:]
        self.index = self._first_unreviewed()

    def _first_unreviewed(self) -> int:
        for i, row in enumerate(self.data):
            if (row[COL_REVIEWED] or "").strip().lower() != "po":
                return i
        return 0

    @property
    def total(self) -> int:
        return len(self.data)

    @property
    def done(self) -> int:
        return sum(1 for r in self.data
                   if (r[COL_REVIEWED] or "").strip().lower() == "po"
                   and (r[COL_QUESTION] or "").strip())

    def current(self) -> list[str]:
        return self.data[self.index]

    def move(self, delta: int) -> None:
        self.index = max(0, min(self.total - 1, self.index + delta))

    def record(self, question: str, gold: str, reviewed: bool) -> None:
        row = self.current()
        row[COL_QUESTION] = (question or "").strip()
        row[COL_GOLD] = (gold or "").strip() or row[COL_GOLD]
        row[COL_REVIEWED] = "po" if reviewed else "jo"
        save(self.rows)


session = Session()


def render() -> tuple:
    row = session.current()
    progress = (f"### Neni {session.index + 1} nga {session.total}  ·  "
                f"**{session.done} të përfunduara**")
    context = (
        f"**Kategoria:** {row[COL_CATEGORY]}  ·  **Neni:** {row[COL_GOLD]}\n\n"
        f"**{row[COL_HEADING]}**\n\n---\n\n{row[COL_TEXT]}"
    )
    return (progress, context, row[COL_QUESTION], row[COL_GOLD],
            (row[COL_REVIEWED] or "").strip().lower() == "po")


def save_and_next(question: str, gold: str) -> tuple:
    session.record(question, gold, reviewed=bool((question or "").strip()))
    session.move(1)
    return render()


def skip(question: str, gold: str) -> tuple:
    """Mark as not usable and move on. Unreviewed rows are dropped on import."""
    session.record("", gold, reviewed=False)
    session.move(1)
    return render()


def go_back(question: str, gold: str) -> tuple:
    session.record(question, gold, reviewed=bool((question or "").strip()))
    session.move(-1)
    return render()


def build() -> gr.Blocks:
    with gr.Blocks(title="Rishikimi i bazës së testimit", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Rishikimi i bazës së testimit")
        gr.Markdown(
            "Lexo nenin, shkruaj pyetjen që do të bënte një tatimpagues i zakonshëm.\n\n"
            "**Rregulli i vetëm që ka rëndësi:** mos përdor fjalët e nenit. Nëse pyetja "
            "përsërit fjalorin e tekstit, kërkimi me fjalëkyçe e gjen pa mundim dhe "
            "testi nuk mat asgjë.\n\n"
            "Ruhet vetë pas çdo neni — mund ta mbyllësh dhe ta vazhdosh më vonë."
        )

        progress = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                context = gr.Markdown()
            with gr.Column(scale=2):
                question = gr.Textbox(label="Pyetja", lines=3,
                                      placeholder="p.sh. Kur duhet të regjistrohem për TVSH?")
                gold = gr.Textbox(label="Neni përgjegjës (korrigjoje nëse s'është ky)")
                reviewed = gr.Checkbox(label="E rishikuar", interactive=False)
                with gr.Row():
                    back_btn = gr.Button("← Mbrapa")
                    skip_btn = gr.Button("Kaloje")
                    next_btn = gr.Button("Ruaj dhe vazhdo →", variant="primary")

        outputs = [progress, context, question, gold, reviewed]
        next_btn.click(save_and_next, [question, gold], outputs)
        skip_btn.click(skip, [question, gold], outputs)
        back_btn.click(go_back, [question, gold], outputs)
        demo.load(render, None, outputs)

    return demo


if __name__ == "__main__":
    print(f"po hapet rishikimi: {session.done}/{session.total} të përfunduara")
    build().launch(inbrowser=True)
