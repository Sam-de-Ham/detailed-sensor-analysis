"""Display stretches for sensor maps (pure numpy, shared by the UI and the report).

A stretch maps data to 0..1 for display; the data itself is never changed, so readouts and
exports always show real values.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from dsa.stats import MAD_TO_SIGMA


class Stretch(StrEnum):
    AUTO = "Auto (STF)"  # PixInsight-style screen transfer: background lifted, highlights kept
    PERCENTILE = "0.5–99.5 %"
    SYMMETRIC = "± 5σ"  # centred on the median: for maps of deviations
    MINMAX = "Min–max"
    ASINH = "Asinh"


def _sample(data: np.ndarray, limit: int = 2_000_000) -> np.ndarray:
    flat = data.ravel()
    step = max(1, flat.size // limit)
    sample = flat[::step]
    return sample[np.isfinite(sample)]


def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    """Midtones transfer function: maps ``m`` to 0.5, keeps 0 and 1."""
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def levels(data: np.ndarray, stretch: Stretch) -> tuple[float, float]:
    """The data values shown as black and white for a linear stretch."""
    sample = _sample(data)
    if sample.size == 0:
        return 0.0, 1.0
    if stretch is Stretch.MINMAX:
        lo, hi = float(sample.min()), float(sample.max())
    elif stretch is Stretch.SYMMETRIC:
        med = float(np.median(sample))
        sigma = float(np.median(np.abs(sample - med)) * MAD_TO_SIGMA) or float(sample.std()) or 1.0
        lo, hi = med - 5 * sigma, med + 5 * sigma
    else:
        lo, hi = (float(v) for v in np.percentile(sample, [0.5, 99.5]))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def apply(data: np.ndarray, stretch: Stretch) -> tuple[np.ndarray, tuple[float, float]]:
    """``(image, levels)`` ready for display. Linear stretches return the data unchanged."""
    if stretch in (Stretch.PERCENTILE, Stretch.SYMMETRIC, Stretch.MINMAX):
        return data, levels(data, stretch)
    sample = _sample(data)
    lo, hi = (float(v) for v in np.percentile(sample, [0.01, 99.99])) if sample.size else (0, 1)
    hi = max(hi, lo + 1e-9)
    x = np.clip((data.astype(np.float32) - lo) / (hi - lo), 0, 1)
    if stretch is Stretch.ASINH:
        med = float(np.clip((np.median(sample) - lo) / (hi - lo), 1e-4, 1))
        beta = 0.1 / med
        return np.arcsinh(beta * x) / np.arcsinh(beta), (0.0, 1.0)
    # Auto (STF): black point 2.8 MAD below the median, background mapped to 25 % grey.
    s = np.clip((sample - lo) / (hi - lo), 0, 1)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * MAD_TO_SIGMA
    black = float(np.clip(med - 2.8 * mad, 0, 1))
    norm = np.clip((x - black) / max(1 - black, 1e-9), 0, 1)
    m = med - black
    midtone = float(_mtf(0.25, np.array(m / max(1 - black, 1e-9)))) if 0 < m < 1 else 0.5
    midtone = min(max(midtone, 1e-4), 0.9999)
    return _mtf(midtone, norm).astype(np.float32), (0.0, 1.0)
