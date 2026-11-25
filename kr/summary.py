"""Aggregate run results across seeds into markdown tables. Pure: records in, text out."""
from __future__ import annotations

from statistics import fmean, stdev
from typing import Callable, Sequence


def mean_sd(xs: Sequence[float]) -> str:
    """'0.73 ± 0.08 (n=3)'; sd is the sample standard deviation, '—' for one value."""
    if not xs:
        return "—"
    sd = f"{stdev(xs):.2f}" if len(xs) > 1 else "—"
    return f"{fmean(xs):.2f} ± {sd}"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def column(runs: Sequence[dict], get: Callable[[dict], float]) -> str:
    return mean_sd([get(r) for r in runs])
