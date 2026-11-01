"""Result primitives shared by every analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class Status(StrEnum):
    MEASURED = "measured"  # the data fully supports the result
    LIMITED = "limited"  # a result exists but with a stated limitation (bound, assumption)
    UNAVAILABLE = "unavailable"  # the data cannot support it


class Level(StrEnum):
    OK = "ok"
    INFO = "info"
    WARN = "warn"
    PROBLEM = "problem"


@dataclass(frozen=True)
class Value:
    """A number with its 1-sigma statistical uncertainty and how it should be read."""

    value: float
    error: float = math.nan
    unit: str = ""
    bound: str | None = None  # "upper" / "lower": the value is a limit, not an estimate

    @property
    def ok(self) -> bool:
        return math.isfinite(self.value)

    def scaled(self, factor: float, unit: str | None = None) -> Value:
        return Value(
            self.value * factor,
            self.error * abs(factor),
            self.unit if unit is None else unit,
            self.bound,
        )

    def format(self, digits: int | None = None) -> str:
        return format_value(self.value, self.error, self.unit, self.bound, digits)


def format_value(
    value: float,
    error: float = math.nan,
    unit: str = "",
    bound: str | None = None,
    digits: int | None = None,
) -> str:
    """``1.234 ± 0.012 e⁻``; the uncertainty sets the number of digits shown.

    The error gets two significant digits when it starts with 1 or 2, else one (Particle Data
    Group convention), and the value never shows more than five significant digits: beyond
    that the statistical error is far below any systematic one.
    """
    if not math.isfinite(value):
        return "—"
    prefix = {"upper": "≤ ", "lower": "≥ "}.get(bound or "", "")
    if digits is None:
        if math.isfinite(error) and error > 0:
            exponent = math.floor(math.log10(error))
            leading = error / 10**exponent
            decimals = max(0, (1 if leading < 3 else 0) - exponent)
            if value != 0:
                decimals = min(decimals, max(0, 4 - math.floor(math.log10(abs(value)))))
            shown = max(error, 10.0**-decimals)
            text = f"{value:.{decimals}f} ± {shown:.{decimals}f}"
            return f"{prefix}{text}{(' ' + unit) if unit else ''}"
        digits = 3
    text = _sig(value, digits)
    return f"{prefix}{text}{(' ' + unit) if unit else ''}"


def _sig(value: float, digits: int) -> str:
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    if magnitude >= 6 or magnitude <= -5:
        return f"{value:.{digits - 1}e}"
    decimals = max(0, digits - 1 - magnitude)
    return f"{value:,.{decimals}f}" if abs(value) >= 10_000 else f"{value:.{decimals}f}"


MISSING = Value(math.nan)  # "not measured"; Value is immutable, so one instance serves all


@dataclass(frozen=True)
class Note:
    level: Level
    text: str


@dataclass
class Section:
    """Common header of every analysis result: what was possible and what to capture next."""

    status: Status = Status.UNAVAILABLE
    method: str = ""
    notes: list[Note] = field(default_factory=list)
    unlock: list[str] = field(default_factory=list)  # captures that would improve this result

    def note(self, level: Level, text: str) -> None:
        self.notes.append(Note(level, text))


def weighted_mean(values: np.ndarray, errors: np.ndarray) -> tuple[float, float, float]:
    """Inverse-variance weighted mean, its error and the reduced chi-square.

    When the values scatter more than their errors allow (chi2 > 1) the error is scaled by
    sqrt(chi2): disagreement between independent sets is real uncertainty.
    """
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    good = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if not good.any():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return math.nan, math.nan, math.nan
        err = float(finite.std(ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else math.nan
        return float(finite.mean()), err, math.nan
    v, e = values[good], errors[good]
    w = 1 / e**2
    mean = float(np.sum(w * v) / np.sum(w))
    err = float(math.sqrt(1 / np.sum(w)))
    chi2 = float(np.sum(w * (v - mean) ** 2) / (v.size - 1)) if v.size > 1 else math.nan
    if math.isfinite(chi2) and chi2 > 1:
        err *= math.sqrt(chi2)
    return mean, err, chi2


def quadrature(*errors: float) -> float:
    finite = [e for e in errors if math.isfinite(e)]
    return math.sqrt(sum(e * e for e in finite)) if finite else math.nan
