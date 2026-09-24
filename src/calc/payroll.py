"""Payroll calculation for Albania, with every parameter traced to its article.

This module exists under one constraint: **no number is written here that does not
appear in the corpus**, and every one carries the document and article it came
from, plus the period it is valid for. A calculator that produces a figure the
reader cannot trace back to a provision is exactly the failure the rest of the
system was built to avoid — it would state a number with the same confidence
whether it was right or wrong.

Two consequences follow, and both are surfaced rather than hidden:

* **Parameters expire.** The monthly employment table is a transitional provision
  that ran to 31 December 2024. The calculator says so instead of quietly applying
  a lapsed rate.
* **Some parameters are not in the corpus at all.** Contribution rates are bounded
  by a minimum and maximum wage that the decision defers to a separate VKM. Those
  bounds are asked of the user, never guessed.

The arithmetic here is deliberately simple. The contribution of this module is
provenance, not cleverness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

DOC_URL = "https://www.tatime.gov.al/shkarko.php?id={}"


@dataclass(frozen=True)
class Source:
    """Where a parameter comes from, and until when it holds."""

    doc_id: int
    cite: str
    act: str
    valid_from: date | None = None
    valid_until: date | None = None

    @property
    def url(self) -> str:
        return DOC_URL.format(self.doc_id)

    @property
    def label(self) -> str:
        return f"{self.act}, {self.cite}"

    def expired_on(self, when: date) -> bool:
        return self.valid_until is not None and when > self.valid_until


# --------------------------------------------------------------------- sources

INCOME_TAX_ACT = "Ligj nr. 29/2023 «Për tatimin mbi të ardhurat»"
CONTRIB_ACT = "VKM nr. 77, datë 28.1.2015"

MONTHLY_TABLE_SRC = Source(
    doc_id=14530, cite="Neni 69, pika 2", act=INCOME_TAX_ACT,
    valid_from=date(2023, 6, 1), valid_until=date(2024, 12, 31),
)
ANNUAL_RATE_SRC = Source(doc_id=14530, cite="Neni 24", act=INCOME_TAX_ACT)
DEDUCTION_SRC = Source(doc_id=14530, cite="Neni 22", act=INCOME_TAX_ACT)
CONTRIB_SRC = Source(doc_id=14444, cite="Kreu III, pika 3", act=CONTRIB_ACT)


# ------------------------------------------------------------------ parameters

# Monthly employment tax, transposed exactly from the table in Neni 69, pika 2.
# Each row: (gross from, gross to or None, [(taxable from, taxable to, rate)]).
# The structure is unusual — the bracket applied depends on the *gross*, not only
# on the taxable amount — so it is kept as the statute states it rather than
# simplified into a single ladder.
MONTHLY_BANDS = [
    (0, 50_000, [(0, None, 0.0)]),
    (50_001, 60_000, [(0, 35_000, 0.0), (35_001, None, 0.13)]),
    (60_001, None, [(0, 30_000, 0.0), (30_001, 200_000, 0.13),
                    (200_001, None, 0.23)]),
]
# The top band is stated in the act as a fixed amount plus a marginal rate.
TOP_BAND_FIXED = 22_100

# Annual employment rates, which is what the act provides once the transitional
# monthly table lapses.
ANNUAL_BANDS = [(0, 2_040_000, 0.13), (2_040_001, None, 0.23)]

SOCIAL_EMPLOYEE = 0.095
SOCIAL_EMPLOYER = 0.150
HEALTH_EMPLOYEE = 0.017
HEALTH_EMPLOYER = 0.017


@dataclass
class Line:
    """One line of the result, with the provision it rests on."""

    label: str
    amount: float
    source: Source
    detail: str = ""


@dataclass
class Result:
    gross: float
    lines: list[Line] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def named(self, label: str) -> float:
        return next((l.amount for l in self.lines if l.label == label), 0.0)

    @property
    def net(self) -> float:
        return (self.gross
                - self.named("Sigurime shoqërore (punëmarrësi)")
                - self.named("Sigurime shëndetësore (punëmarrësi)")
                - self.named("Tatimi mbi të ardhurat nga paga"))

    @property
    def employer_total(self) -> float:
        return (self.gross
                + self.named("Sigurime shoqërore (punëdhënësi)")
                + self.named("Sigurime shëndetësore (punëdhënësi)"))


def group(value: float) -> str:
    """Albanian thousands grouping: a space, never a comma."""
    return f"{value:,.0f}".replace(",", " ")


def monthly_income_tax(gross: float) -> tuple[float, str]:
    """Tax on one month's employment income, per the table in Neni 69, pika 2.

    The statute writes the top step as "22 100 lekë + 23% të shumës mbi 200 000".
    That fixed 22 100 **is** the tax already accrued on the 30 001–200 000 band
    (170 000 × 13%), so it replaces the lower step rather than adding to it.
    Treating it as an extra term overstates the tax on high salaries by exactly
    22 100 — the arithmetic still looks plausible, which is what makes the mistake
    worth guarding with a test.
    """
    if gross <= 50_000:
        return 0.0, ""

    if gross <= 60_000:
        taxable = max(0.0, gross - 35_000)
        return taxable * 0.13, f"13% × {group(taxable)}"

    if gross <= 200_000:
        taxable = gross - 30_000
        return taxable * 0.13, f"13% × {group(taxable)}"

    over = gross - 200_000
    return (TOP_BAND_FIXED + over * 0.23,
            f"{group(TOP_BAND_FIXED)} + 23% × {group(over)}")


def calculate(gross_monthly: float, *, min_wage: float | None = None,
              max_wage: float | None = None, today: date | None = None) -> Result:
    """Break a gross monthly salary into contributions, tax and net pay."""
    today = today or date.today()
    result = Result(gross=gross_monthly)

    # Contributions are computed on a base bounded by the minimum and maximum
    # wage. Those two figures are set by a separate decision that the corpus does
    # not contain, so they are taken from the caller — never assumed.
    base = gross_monthly
    if min_wage is not None:
        base = max(base, min_wage)
    if max_wage is not None:
        base = min(base, max_wage)
    if min_wage is None or max_wage is None:
        result.warnings.append(
            "Kontributet llogariten mbi pagën bruto pa kufirin minimal/maksimal: "
            "ato caktohen me VKM të veçantë që nuk gjendet në korpusin e këtij "
            "sistemi. Shto vlerat vetë nëse dëshiron llogaritje të plotë."
        )

    # Group the thousands separately from the rate. Formatting the whole string
    # and then swapping commas for spaces also swaps the decimal comma, turning
    # "9,5%" into "9 5%".
    shown = group(base)
    result.lines += [
        Line("Sigurime shoqërore (punëmarrësi)", base * SOCIAL_EMPLOYEE,
             CONTRIB_SRC, f"9,5% × {shown}"),
        Line("Sigurime shëndetësore (punëmarrësi)", base * HEALTH_EMPLOYEE,
             CONTRIB_SRC, f"1,7% × {shown}"),
        Line("Sigurime shoqërore (punëdhënësi)", base * SOCIAL_EMPLOYER,
             CONTRIB_SRC, f"15,0% × {shown}"),
        Line("Sigurime shëndetësore (punëdhënësi)", base * HEALTH_EMPLOYER,
             CONTRIB_SRC, f"1,7% × {shown}"),
    ]

    tax, steps = monthly_income_tax(gross_monthly)
    result.lines.append(
        Line("Tatimi mbi të ardhurat nga paga", tax, MONTHLY_TABLE_SRC, steps))

    if MONTHLY_TABLE_SRC.expired_on(today):
        result.warnings.append(
            f"Tabela mujore e tatimit ({MONTHLY_TABLE_SRC.cite}) është dispozitë "
            f"kalimtare me afat deri më 31.12.2024. Sot është "
            f"{today.strftime('%d.%m.%Y')}, prandaj kjo shifër duhet verifikuar "
            f"kundrejt normave në fuqi — korpusi i këtij sistemi nuk përmban një "
            f"tabelë mujore më të re."
        )

    return result
