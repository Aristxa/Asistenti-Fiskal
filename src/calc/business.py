"""VAT and business income tax, under the same rule as the payroll module.

Every rate here was located in the corpus before a line of arithmetic was written,
and each carries the act, the article and — where the provision is transitional —
the date it stops applying. See `src.calc.payroll` for why that matters: a figure
looks identical whether it is right or wrong, so provenance is the only thing that
lets a reader check it.

Two provisions in here expire, and both are surfaced rather than applied silently:

* the 0% rate on personal business income up to 14 million lekë runs to
  31 December 2029;
* the reduced VAT rates are not implemented at all, because which supply qualifies
  is a legal classification this module has no business making.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.calc.payroll import Line, Source, group

VAT_ACT = "Ligj nr. 92/2014 «Për tatimin mbi vlerën e shtuar»"
INCOME_ACT = "Ligj nr. 29/2023 «Për tatimin mbi të ardhurat»"
GUIDANCE_ACT = "Udhëzim nr. 26, datë 8.9.2023"

VAT_STANDARD_SRC = Source(doc_id=14534, cite="Neni 48", act=VAT_ACT)
VAT_REDUCED_SRC = Source(doc_id=14534, cite="Neni 49", act=VAT_ACT)
BUSINESS_RATE_SRC = Source(doc_id=14530, cite="Neni 24, pika 2", act=INCOME_ACT)
SMALL_BUSINESS_SRC = Source(
    doc_id=14596, cite="pika 50", act=GUIDANCE_ACT,
    valid_until=date(2029, 12, 31),
)

VAT_STANDARD = 0.20

# Neni 24, pika 2: annual bands for individual traders and the self-employed.
BUSINESS_BANDS = [(0, 14_000_000, 0.15), (14_000_001, None, 0.23)]
SMALL_BUSINESS_CEILING = 14_000_000


@dataclass
class Result:
    lines: list[Line] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def named(self, label: str) -> float:
        return next((l.amount for l in self.lines if l.label == label), 0.0)


def vat(amount: float, *, amount_includes_vat: bool = False) -> Result:
    """Split an amount into net, VAT and gross at the standard rate.

    Both directions are offered because both are asked in practice: a seller
    starts from a net price, a buyer starts from what is on the receipt. The
    second is where people go wrong — VAT on a gross amount is 20/120 of it, not
    20%.
    """
    result = Result()
    if amount_includes_vat:
        net = amount / (1 + VAT_STANDARD)
        tax = amount - net
        detail = f"{group(amount)} × 20/120"
    else:
        net = amount
        tax = amount * VAT_STANDARD
        detail = f"20% × {group(amount)}"

    result.lines += [
        Line("Vlera e tatueshme (pa TVSH)", net, VAT_STANDARD_SRC),
        Line("TVSH", tax, VAT_STANDARD_SRC, detail),
        Line("Vlera me TVSH", net + tax, VAT_STANDARD_SRC),
    ]
    result.warnings.append(
        "Llogaritja përdor shkallën standarde 20%. Për furnizime me shkallë të "
        "reduktuar (Neni 49) klasifikimi është çështje ligjore dhe nuk kryhet "
        "automatikisht nga ky mjet."
    )
    return result


def business_income_tax(annual_profit: float, *, turnover: float | None = None,
                        today: date | None = None) -> Result:
    """Annual tax on business income for an individual trader or self-employed.

    The 0% transitional rate depends on **turnover**, not on profit, so the two
    are asked separately rather than conflated.
    """
    today = today or date.today()
    result = Result()

    small = turnover is not None and turnover <= SMALL_BUSINESS_CEILING
    expired = SMALL_BUSINESS_SRC.expired_on(today)

    if small and not expired:
        result.lines.append(
            Line("Tatimi mbi të ardhurat nga biznesi", 0.0, SMALL_BUSINESS_SRC,
                 f"0% — qarkullim {group(turnover)} ≤ 14 000 000"))
        result.warnings.append(
            "Norma 0% zbatohet deri më 31.12.2029 për të ardhurat personale nga "
            "biznesi kur qarkullimi vjetor është deri në 14 milionë lekë."
        )
        return result

    if small and expired:
        result.warnings.append(
            f"Norma 0% për qarkullim deri në 14 milionë lekë ka pasur afat deri më "
            f"31.12.2029; sot është {today.strftime('%d.%m.%Y')}, prandaj u zbatuan "
            f"normat e përgjithshme."
        )

    if annual_profit <= 14_000_000:
        tax = annual_profit * 0.15
        detail = f"15% × {group(annual_profit)}"
    else:
        tax = 14_000_000 * 0.15 + (annual_profit - 14_000_000) * 0.23
        detail = (f"15% × 14 000 000  +  23% × "
                  f"{group(annual_profit - 14_000_000)}")

    result.lines.append(
        Line("Tatimi mbi të ardhurat nga biznesi", tax, BUSINESS_RATE_SRC, detail))
    if turnover is None:
        result.warnings.append(
            "Qarkullimi vjetor nuk u dha, prandaj nuk u kontrollua nëse zbatohet "
            "norma 0% për biznesin e vogël."
        )
    return result
