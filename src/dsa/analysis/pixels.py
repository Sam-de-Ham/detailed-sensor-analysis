"""Defective pixels: hot and warm (dark current), dead and weak (flats), bad columns, RTS.

EMVA 1288 deliberately gives no fixed hot-pixel threshold; it shows the distribution. DSA does
both: the cumulative distribution of excess dark current, plus counts at robust-z thresholds
(warm: 5–50 σ, hot: ≥ 50 σ above the local level, so amp glow isn't counted as hot pixels).

Random telegraph signal (RTS) pixels jump between two or more discrete levels. They are found
in two steps: pixels whose temporal variance is far above what read noise (and their own dark
shot noise) explains, then a bimodality test on each candidate's time series — the best split
into a low and a high group must separate the groups by many times their internal scatter.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from dsa.analysis.common import Section, Status
from dsa.analysis.dark import DarkCurrent
from dsa.analysis.data import ModeData
from dsa.analysis.flat import MIN_LEVEL, FlatField
from dsa.analysis.gain import Gain
from dsa.analysis.noise import ReadNoise, noise_source
from dsa.catalog import format_exposure
from dsa.cfa import CfaLayout
from dsa.frames import FrameInfo, FrameType
from dsa.stats import robust_sigma

MAX_LISTED = 100_000
LOCAL = 7  # neighbourhood (plane pixels) defining a pixel's expected level
MAX_RTS_CANDIDATES = 20_000
# Two groups this many pooled sigmas apart. Candidates already have >= 4x the expected
# variance; a Gaussian pixel split at its median gives ~2.5-3.5, a telegraph pixel its
# step / read noise.
RTS_SEPARATION = 4.0

SeriesReader = Callable[[Sequence[FrameInfo], np.ndarray, np.ndarray], np.ndarray]


@dataclass
class PixelClass:
    key: str
    label: str
    description: str
    count: int
    fraction: float
    ys: np.ndarray  # full-resolution coordinates (at most MAX_LISTED, worst first)
    xs: np.ndarray
    values: np.ndarray  # the quantity that classified them (unit in ``unit``)
    unit: str = ""


@dataclass
class Rts:
    source: str
    n_frames: int
    candidates: int
    count: int
    fraction: float
    ys: np.ndarray  # positions of the RTS pixels, full resolution
    xs: np.ndarray
    amplitudes: np.ndarray  # step between levels (e⁻, or ADU without gain)
    unit: str
    traces: np.ndarray  # [examples, frames]
    trace_xy: list[tuple[int, int]]
    trace_split: list[float]  # threshold between the two levels per example


@dataclass
class Defects(Section):
    total_pixels: int = 0
    classes: list[PixelClass] = field(default_factory=list)
    cumulative: tuple[np.ndarray, np.ndarray] | None = None  # (excess, pixels above)
    cumulative_unit: str = ""
    sub_noise: float = math.nan  # one sub's noise in the cumulative plot's unit
    bad_columns: list[tuple[int, str, float]] = field(default_factory=list)  # x, source, z
    rts: Rts | None = None

    def get(self, key: str) -> PixelClass | None:
        return next((c for c in self.classes if c.key == key), None)


def _per_plane_excess(image: np.ndarray, layout: CfaLayout) -> tuple[np.ndarray, np.ndarray]:
    """Deviation from the 7×7 neighbourhood median (per colour plane) and its robust z-score.

    Defective pixels are isolated, while amp glow, vignetting and dust vary smoothly: a small
    median follows steep glow gradients that a large background would mistake for hot pixels.
    """
    excess = np.zeros(image.shape, dtype=np.float32)
    z = np.zeros(image.shape, dtype=np.float32)
    for name, plane in layout.planes(image):
        ex = plane - ndimage.median_filter(plane, size=LOCAL, mode="nearest")
        sigma = robust_sigma(ex[::2, ::2]) or 1e-12
        layout.plane(excess, name)[...] = ex
        layout.plane(z, name)[...] = ex / sigma
    return excess, z


def _pixel_class(
    key: str,
    label: str,
    description: str,
    mask: np.ndarray,
    order: np.ndarray,
    values: np.ndarray,
    unit: str,
) -> PixelClass:
    ys, xs = np.nonzero(mask)
    vals = values[ys, xs]
    keep = np.argsort(-np.abs(order[ys, xs]))[:MAX_LISTED]
    return PixelClass(
        key,
        label,
        description,
        int(ys.size),
        float(ys.size / mask.size),
        ys[keep].astype(np.int32),
        xs[keep].astype(np.int32),
        vals[keep],
        unit,
    )


def find_defects(
    data: ModeData,
    dark: DarkCurrent,
    flat: FlatField,
    rn: ReadNoise,
    gain: Gain,
    read_series: SeriesReader | None = None,
) -> Defects:
    result = Defects()
    layout = data.layout
    shape = (data.mode.height, data.mode.width)
    result.total_pixels = shape[0] * shape[1]
    sources = []

    if dark.map is not None:
        sources.append("darks")
        excess, z = _per_plane_excess(dark.map, layout)
        unit = dark.unit
        # In units of one sub's noise: what a pixel looks like in a single frame.
        sub = dark.sub_noise if math.isfinite(dark.sub_noise) and dark.sub_noise > 0 else math.nan
        level = excess / sub if math.isfinite(sub) else z / 5
        t = format_exposure(dark.map_exposure)
        significant = z >= 5  # also clearly above the master's own noise
        result.classes.append(
            _pixel_class(
                "hot",
                "Hot",
                f"Excess dark signal ≥ 5× the noise of one {t} dark: stands out in every sub.",
                significant & (level >= 5),
                level,
                excess,
                unit,
            )
        )
        result.classes.append(
            _pixel_class(
                "warm",
                "Warm",
                f"Excess dark signal 2–5× the noise of one {t} dark: shows in stacks, removed by "
                "matching darks.",
                significant & (level >= 2) & (level < 5),
                level,
                excess,
                unit,
            )
        )
        values = np.sort(excess[z >= 3].ravel())[::-1]
        if values.size:
            sigma = float(np.median(np.abs(excess)) * 1.4826) or 1e-12
            thresholds = np.geomspace(max(sigma, values[-1]), max(values[0], sigma * 2), 120)
            above = values.size - np.searchsorted(values[::-1], thresholds, side="left")
            result.cumulative = (thresholds, above)
            result.cumulative_unit = unit
            result.sub_noise = sub
        result.bad_columns += _bad_columns(excess, layout, "dark")

    bias = data.best(FrameType.BIAS)
    if bias is not None:
        sources.append("bias")
        excess, z = _per_plane_excess(bias[1].mean_map.astype(np.float32), layout)
        result.classes.append(
            _pixel_class(
                "bias",
                "Offset outliers",
                "Bias level ≥ 8σ from its neighbours (fixed offset "
                "defects; removed by bias/dark subtraction).",
                np.abs(z) >= 8,
                z,
                excess,
                "ADU",
            )
        )
        result.bad_columns += _bad_columns(excess, layout, "bias")

    # The flat with the best signal-to-noise: level × √frames, among bright-enough flats.
    candidates = [s for s in flat.sets if s.prnu_map is not None and s.level_fraction >= MIN_LEVEL]
    flat_set = max(candidates, key=lambda s: s.level * math.sqrt(s.n), default=None)
    if flat_set is not None:
        sources.append("flats")
        response = flat_set.prnu_map / 100 + 1
        sigma = max(robust_sigma(flat_set.prnu_map[::2, ::2]) / 100, 1e-4)
        low_cut = 1 - max(5 * sigma, 0.05)
        dead = response < 0.5
        weak = (response < low_cut) & ~dead
        strong = response > 1 + max(5 * sigma, 0.05)
        rel = (response - 1) * 100
        result.classes.append(
            _pixel_class(
                "dead",
                "Dead",
                f"Response below 50 % of their neighbours in {flat_set.label}.",
                dead,
                -rel,
                rel,
                "%",
            )
        )
        result.classes.append(
            _pixel_class(
                "weak",
                "Low response",
                f"Response {100 * (1 - low_cut):.0f}–50 % below their neighbours.",
                weak,
                -rel,
                rel,
                "%",
            )
        )
        result.classes.append(
            _pixel_class(
                "strong",
                "High response",
                "Response well above their neighbours.",
                strong,
                rel,
                rel,
                "%",
            )
        )
        result.bad_columns += _bad_columns(response - 1, layout, "flat")

    if read_series is not None:
        result.rts = find_rts(data, rn, gain, dark, read_series)
        rts = result.rts
        if rts is not None and rts.count:
            result.classes.append(
                PixelClass(
                    "rts",
                    "RTS",
                    "Random telegraph signal: the pixel jumps between discrete levels.",
                    rts.count,
                    rts.fraction,
                    rts.ys,
                    rts.xs,
                    rts.amplitudes,
                    rts.unit,
                )
            )

    if not sources:
        result.status = Status.UNAVAILABLE
        result.method = "Needs darks (hot pixels), bias (offset defects) or flats (dead pixels)."
        result.unlock.append("Darks at your sub length, and flats.")
        return result
    result.status = Status.MEASURED
    result.method = (
        "Deviations from the 7×7 neighbourhood median, per colour plane: darks "
        "(hot/warm), bias (offset outliers), flats (dead/weak/strong). RTS: bimodal "
        "per-pixel time series."
    )
    if dark.map is None:
        result.unlock.append("Darks at your sub length reveal hot and warm pixels.")
    if flat_set is None:
        result.unlock.append("Flats reveal dead and low-response pixels.")
    return result


def _bad_columns(
    excess: np.ndarray, layout: CfaLayout, source: str
) -> list[tuple[int, str, float]]:
    """Columns whose median departs from the others by > 8σ *and* by at least one pixel's noise.

    With thousands of pixels per column, the 8σ test alone flags offsets far too small to
    see; requiring a pixel's worth of deviation keeps only columns that stand out as lines.
    """
    found = []
    for name, plane in layout.planes(excess):
        cols = np.median(plane, axis=0)
        sigma = robust_sigma(cols) or 1e-12
        pixel_sigma = robust_sigma(plane[::4, ::4])
        deviation = cols - np.median(cols)
        z = deviation / sigma
        dx = layout.offsets[name][1]
        for c in np.flatnonzero((np.abs(z) > 8) & (np.abs(deviation) >= pixel_sigma)):
            found.append((int(c * layout.step + dx), source, float(z[c])))
    return sorted(found)


# ------------------------------------------------------------------------------- RTS --


def bimodality(
    series: np.ndarray, min_fraction: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best two-group split of each column of ``series`` ([frames, pixels]).

    Returns (separation in units of the pooled within-group sigma, step between the group means,
    threshold between the groups) per pixel.
    """
    x = np.sort(series, axis=0).astype(np.float64)
    n = x.shape[0]
    c1 = np.cumsum(x, axis=0)
    c2 = np.cumsum(x * x, axis=0)
    total1, total2 = c1[-1], c2[-1]
    lo_min = max(2, math.ceil(min_fraction * n))
    best_sep = np.zeros(x.shape[1])
    best_step = np.zeros(x.shape[1])
    best_thr = np.zeros(x.shape[1])
    best_score = np.full(x.shape[1], -np.inf)
    for k in range(lo_min, n - lo_min + 1):
        m1 = c1[k - 1] / k
        m2 = (total1 - c1[k - 1]) / (n - k)
        ss1 = c2[k - 1] - k * m1**2
        ss2 = (total2 - c2[k - 1]) - (n - k) * m2**2
        within = np.sqrt(np.maximum((ss1 + ss2) / (n - 2), 1e-12))
        score = (m2 - m1) ** 2 * k * (n - k)  # between-group variance (Otsu)
        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_sep = np.where(better, (m2 - m1) / within, best_sep)
        best_step = np.where(better, m2 - m1, best_step)
        best_thr = np.where(better, (x[k - 1] + x[k]) / 2, best_thr)
    return best_sep, best_step, best_thr


def find_rts(
    data: ModeData,
    rn: ReadNoise,
    gain: Gain,
    dark: DarkCurrent,
    read_series: SeriesReader,
    min_frames: int = 8,
) -> Rts | None:
    source = noise_source(data)
    if source is None or rn.sigma_map is None:
        return None
    fs, st = source
    if st.n_frames < min_frames:
        return None
    layout = data.layout
    var = rn.sigma_map.astype(np.float64) ** 2
    expected = np.zeros_like(var)
    for _i, (name, plane) in enumerate(layout.planes(var)):
        floor = float(np.median(plane)) or 1e-12
        exp_plane = np.full(plane.shape, floor)
        if fs.kind is FrameType.DARK and gain.gain.ok:
            # A hot pixel's own dark shot noise is not RTS: variance (ADU²) = signal / g.
            mean_plane = layout.plane(st.mean_map, name).astype(np.float64)
            signal = np.clip(mean_plane - np.median(mean_plane), 0, None)
            exp_plane += signal / gain.gain.value
        layout.plane(expected, name)[...] = exp_plane
    ratio = var / expected
    cand = np.flatnonzero(ratio.ravel() > 4)
    if cand.size > MAX_RTS_CANDIDATES:
        cand = cand[np.argsort(-ratio.ravel()[cand])[:MAX_RTS_CANDIDATES]]
    ys, xs = np.unravel_index(cand, var.shape)
    used = set(st.paths)
    frames = [f for f in fs.frames if str(f.path) in used]
    if cand.size == 0 or not frames:
        empty = np.zeros(0, np.int32)
        return Rts(
            fs.label,
            st.n_frames,
            0,
            0,
            0.0,
            empty,
            empty,
            np.zeros(0),
            "ADU",
            np.zeros((0, 0)),
            [],
            [],
        )
    series = read_series(frames, ys, xs)
    # Remove frame-to-frame level jumps of the whole frame.
    series = series - (st.frame_mean.mean(axis=1) - st.frame_mean.mean())[: series.shape[0], None]
    sep, step, thr = bimodality(series)
    is_rts = sep > RTS_SEPARATION
    g = gain.gain.value if gain.gain.ok else 1.0
    unit = "e⁻" if gain.gain.ok else "ADU"
    order = np.argsort(-step * is_rts)
    examples = [int(i) for i in order[:8] if is_rts[i]]
    return Rts(
        source=fs.label,
        n_frames=series.shape[0],
        candidates=int(cand.size),
        count=int(is_rts.sum()),
        fraction=float(is_rts.sum() / var.size),
        ys=ys[is_rts].astype(np.int32),
        xs=xs[is_rts].astype(np.int32),
        amplitudes=(step[is_rts] * g).astype(np.float32),
        unit=unit,
        traces=series[:, examples].T.astype(np.float32),
        trace_xy=[(int(xs[i]), int(ys[i])) for i in examples],
        trace_split=[float(thr[i]) for i in examples],
    )
