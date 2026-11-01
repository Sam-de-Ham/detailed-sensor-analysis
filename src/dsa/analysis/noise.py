"""Read noise and its structure: per-pixel, row and column components, temporal and fixed.

Read noise is temporal noise at zero exposure. It is measured from the *difference* of two
bias frames, which cancels every fixed pattern: σ_read = σ(B1 − B2)/√2 (Janesick; EMVA 1288).
Every disjoint pair of the set gives one estimate; their scatter is the uncertainty.

Pattern noise follows the EMVA 1288 decomposition into pixel, row and column variances, applied
twice: to frame-to-frame deviations of the row/column means (temporal banding, which no
calibration removes) and to the averaged frame (fixed pattern, which bias/dark subtraction
removes). Averages of N frames still contain σ²/N of temporal noise; that residue is subtracted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from dsa.analysis.common import MISSING, Level, Section, Status, Value, quadrature
from dsa.analysis.data import ModeData
from dsa.analysis.levels import DataFormat, quantisation_variance
from dsa.catalog import FrameSet, format_exposure
from dsa.frames import FrameType
from dsa.measure import SetStats
from dsa.stats import central_roi, clipped_stats, large_scale

DETREND_ROWS = 51  # median-filter length separating banding from slow gradients


@dataclass
class ReadNoise(Section):
    source: str = ""  # "bias" / "dark" / ""
    set_label: str = ""
    planes: list[str] = field(default_factory=list)
    adu: list[Value] = field(default_factory=list)  # per plane, file ADU
    combined_adu: Value = MISSING
    electrons: list[Value] = field(default_factory=list)
    combined_e: Value = MISSING
    quantisation_adu: float = 0.0  # rms quantisation noise removed
    per_pair: np.ndarray | None = None  # [pairs, planes] read noise per pair, ADU
    pair_times: np.ndarray | None = None
    median_pixel_adu: list[float] = field(default_factory=list)
    rms_pixel_adu: list[float] = field(default_factory=list)
    sigma_hist: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # edges, counts
    sigma_map: np.ndarray | None = None  # per-pixel temporal sigma, file ADU
    n_frames: int = 0
    exposure: float = 0.0

    def to_electrons(self, gain: Value) -> None:
        if not gain.ok:
            return
        self.electrons = [_times(v, gain) for v in self.adu]
        self.combined_e = _times(self.combined_adu, gain)


def _times(v: Value, gain: Value) -> Value:
    value = v.value * gain.value
    rel = quadrature(v.error / v.value if v.value else math.nan, gain.error / gain.value)
    return Value(value, abs(value) * rel, "e⁻", v.bound)


def noise_source(data: ModeData) -> tuple[FrameSet, SetStats] | None:
    """Bias subs, else the shortest dark subs: frames with pairs and least extra signal."""
    bias = [p for p in data.of(FrameType.BIAS) if p[1].n_pairs > 0]
    if bias:
        return max(bias, key=lambda p: p[1].n_frames)
    darks = [p for p in data.of(FrameType.DARK) if p[1].n_pairs > 0]
    if darks:
        return min(darks, key=lambda p: (p[0].exposure or 0, -p[1].n_frames))
    return None


def read_noise(data: ModeData, fmt: DataFormat) -> ReadNoise:
    result = ReadNoise(planes=data.planes)
    source = noise_source(data)
    if source is None:
        masters = data.of(FrameType.BIAS, subs=False) + data.of(FrameType.DARK, subs=False)
        result.status = Status.UNAVAILABLE
        result.method = "Needs at least two single bias (or short dark) frames."
        if masters:
            result.note(
                Level.INFO,
                "Only master frames are available. A master's pixel-to-pixel "
                "scatter is fixed-pattern noise plus read noise / √N, so read noise "
                "cannot be separated from it.",
            )
        result.unlock.append(
            "Capture 20 or more bias frames (shortest exposure, cap on) at this setting."
        )
        return result

    fs, st = source
    result.source = "bias" if fs.kind is FrameType.BIAS else "dark"
    result.set_label = fs.label
    result.n_frames = st.n_frames
    result.exposure = float(fs.exposure or 0.0)
    scale = fmt.container_scale
    q_values = []
    per_pair = np.full_like(st.pair_var, np.nan)
    for i, _ in enumerate(data.planes):
        v = st.pair_var[:, i]
        var = float(np.mean(v))
        if v.size > 1:
            err_var = float(np.std(v, ddof=1) / math.sqrt(v.size))
        else:
            err_var = var * math.sqrt(2 / max(st.pair_npix[0, i], 2))
        level = float(np.mean(st.pair_mean[:, i]))
        q = quantisation_variance(st, i, level, math.sqrt(var)) if fmt.lossy else scale**2 / 12
        q_values.append(q)
        if var < 0.24 * scale**2:
            # EMVA 1288: below 0.24 DN² the quantisation correction is unreliable.
            result.adu.append(Value(math.sqrt(var), math.nan, "ADU", "upper"))
            result.note(
                Level.WARN,
                f"{data.planes[i]}: noise is below 0.5 native ADU; "
                "quantisation dominates, so only an upper limit is given.",
            )
        else:
            rn = math.sqrt(var - q)
            result.adu.append(Value(rn, err_var / (2 * rn), "ADU"))
        per_pair[:, i] = np.sqrt(np.clip(v - q, 0, None))
    result.per_pair = per_pair
    result.pair_times = st.times[0 : 2 * st.n_pairs : 2]
    result.quantisation_adu = math.sqrt(float(np.mean(q_values)))

    variances = np.array([v.value**2 for v in result.adu])
    var_errs = np.array([2 * v.value * v.error for v in result.adu])
    combined = math.sqrt(float(variances.mean()))
    combined_err = float(np.sqrt(np.nansum(var_errs**2)) / len(var_errs) / (2 * combined))
    bound = "upper" if any(v.bound for v in result.adu) else None
    result.combined_adu = Value(combined, combined_err, "ADU", bound)

    _pixel_distribution(result, data, st, q_values)

    if result.source == "bias":
        result.status = Status.MEASURED
        result.method = (
            f"σ(B₁ − B₂)/√2 over {st.n_pairs} independent bias pairs, central 50 % "
            "of the frame, 5σ-clipped, quantisation noise removed."
        )
    else:
        result.status = Status.LIMITED
        t = format_exposure(fs.exposure or 0)
        result.method = (
            f"σ(D₁ − D₂)/√2 over {st.n_pairs} pairs of {t} darks: includes the "
            "shot noise of the dark signal, so it is an upper limit."
        )
        result.adu = [Value(v.value, v.error, v.unit, "upper") for v in result.adu]
        result.combined_adu = Value(combined, combined_err, "ADU", "upper")
        result.unlock.append(
            "Capture bias frames (shortest exposure) to measure read noise without dark current."
        )
    return result


def _pixel_distribution(result: ReadNoise, data: ModeData, st: SetStats, q: list[float]) -> None:
    """Per-pixel temporal noise: median, rms, histogram and map."""
    if st.var_map is None or st.n_frames < 3:
        return
    sigma_full = np.zeros(st.var_map.shape, dtype=np.float32)
    for i, (name, plane_var) in enumerate(st.layout.planes(st.var_map)):
        # Frame-to-frame jumps of the whole frame's level add to every pixel's variance.
        jitter = float(np.var(st.frame_mean[:, i], ddof=1))
        var = np.clip(plane_var - jitter - q[i], 0, None)
        sigma = np.sqrt(var)
        st.layout.plane(sigma_full, name)[...] = sigma
        roi = sigma[central_roi(sigma.shape)]
        result.median_pixel_adu.append(float(np.median(roi)))
        result.rms_pixel_adu.append(float(np.sqrt(np.mean(var[central_roi(var.shape)]))))
        top = max(float(np.percentile(roi, 99.99)) * 1.5, 1e-3)
        edges = np.linspace(0, top, 200)
        counts, _ = np.histogram(sigma, bins=edges)
        result.sigma_hist.append((edges, counts))
    result.sigma_map = sigma_full


# --------------------------------------------------------------------------- pattern --


@dataclass
class Peak:
    axis: str  # "row" (horizontal bands) or "column"
    kind: str  # "temporal" or "fixed"
    frequency: float  # cycles per pixel (plane pixels)
    period: float  # pixels (full-resolution)
    amplitude: float  # ADU, sinusoid amplitude


@dataclass
class PatternNoise(Section):
    planes: list[str] = field(default_factory=list)
    source_label: str = ""
    fixed_label: str = ""
    pixel_temporal: list[Value] = field(default_factory=list)  # white per-pixel part
    row_temporal: list[Value] = field(default_factory=list)
    col_temporal: list[Value] = field(default_factory=list)
    pixel_fixed: list[Value] = field(default_factory=list)  # DSNU of the offset, high-pass
    row_fixed: list[Value] = field(default_factory=list)
    col_fixed: list[Value] = field(default_factory=list)
    gradient_ptp: list[float] = field(default_factory=list)  # slow row/col gradients, ADU p-p
    row_profile: np.ndarray | None = None  # fixed row means (plane 0), ADU
    col_profile: np.ndarray | None = None
    row_deviation: np.ndarray | None = None  # temporal row-mean deviations [frames, rows]
    psd: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    spectrum2d: np.ndarray | None = None  # log10 power of the fixed pattern, centred
    peaks: list[Peak] = field(default_factory=list)
    banding_ratio: float = math.nan  # temporal row noise / pixel noise


def _temporal_profile_var(profiles: np.ndarray) -> tuple[float, np.ndarray]:
    """Variance of frame-to-frame deviations of a row/column-mean profile."""
    dev = profiles - profiles.mean(axis=0, keepdims=True)
    dev = dev - dev.mean(axis=1, keepdims=True)  # a global level jump is not banding
    n = profiles.shape[0]
    return float(np.mean(dev**2) * n / (n - 1)), dev


def _detrend(profile: np.ndarray) -> np.ndarray:
    size = min(DETREND_ROWS, max(3, profile.size // 4 * 2 + 1))
    return profile - ndimage.median_filter(profile, size=size, mode="nearest")


def _psd(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = signal - signal.mean(axis=-1, keepdims=True)
    power = np.abs(np.fft.rfft(signal, axis=-1)) ** 2 / signal.shape[-1]
    if power.ndim > 1:
        power = power.mean(axis=0)
    freq = np.fft.rfftfreq(signal.shape[-1])
    return freq[1:], power[1:]


def _find_peaks(
    freq: np.ndarray, power: np.ndarray, length: int, axis: str, kind: str, step: int
) -> list[Peak]:
    if power.size < 32:
        return []
    baseline = ndimage.median_filter(power, size=31, mode="nearest")
    ratio = power / np.maximum(baseline, 1e-30)
    peaks = []
    for i in np.flatnonzero((ratio > 25) & (freq > 4 / length)):
        if power[i] < power[max(0, i - 1) : i + 2].max():
            continue
        amplitude = 2 * math.sqrt(power[i] * length) / length  # a sinusoid has |X| = A·L/2
        peaks.append(Peak(axis, kind, float(freq[i]), float(step / freq[i]), amplitude))
    peaks.sort(key=lambda p: -p.amplitude)
    return peaks[:5]


def pattern_noise(data: ModeData, rn: ReadNoise) -> PatternNoise:
    result = PatternNoise(planes=data.planes)
    source = noise_source(data)
    step = data.layout.step
    if source is not None and source[1].n_frames >= 3:
        fs, st = source
        result.source_label = fs.label
        for i, _name in enumerate(data.planes):
            pix_var = rn.adu[i].value ** 2 if rn.adu else float(np.mean(st.pair_var[:, i]))
            rows, cols = st.row_means[i], st.col_means[i]
            row_var, row_dev = _temporal_profile_var(rows)
            col_var, _ = _temporal_profile_var(cols)
            n_cols, n_rows = cols.shape[1], rows.shape[1]
            row_t = max(row_var - pix_var / n_cols, 0.0)
            col_t = max(col_var - pix_var / n_rows, 0.0)
            err_r = row_var * math.sqrt(2 / rows.size)
            err_c = col_var * math.sqrt(2 / cols.size)
            result.row_temporal.append(_sqrt_value(row_t, err_r))
            result.col_temporal.append(_sqrt_value(col_t, err_c))
            result.pixel_temporal.append(_sqrt_value(max(pix_var - row_t - col_t, 0.0), math.nan))
            if i == 0:
                result.row_deviation = row_dev.astype(np.float32)
                result.psd["row temporal"] = _psd(_detrend_rows(row_dev))
                result.psd["column temporal"] = _psd(
                    _detrend_rows(cols - cols.mean(axis=0, keepdims=True))
                )
                result.peaks += _find_peaks(
                    *result.psd["row temporal"], n_rows, "row", "temporal", step
                )
                result.peaks += _find_peaks(
                    *result.psd["column temporal"], n_cols, "column", "temporal", step
                )
        pixel = np.mean([v.value for v in rn.adu]) if rn.adu else math.nan
        result.banding_ratio = float(np.mean([v.value for v in result.row_temporal]) / pixel)

    fixed = data.best(FrameType.BIAS) or noise_source(data)
    if fixed is not None:
        fs, st = fixed
        result.fixed_label = fs.label
        n_avg = st.n_combined if fs.is_master else st.n_frames
        _fixed_pattern(result, data, st, rn, n_avg)

    if not result.row_temporal and not result.row_fixed:
        result.status = Status.UNAVAILABLE
        result.method = "Needs bias or dark frames."
        result.unlock.append("Capture 20 or more bias frames.")
        return result
    result.status = Status.MEASURED if result.row_temporal else Status.LIMITED
    if not result.row_temporal:
        result.unlock.append("Temporal (random) banding needs at least three single bias frames.")
    result.method = (
        "EMVA 1288 row/column/pixel decomposition of the frame-to-frame deviations of "
        "row and column means (temporal) and of the averaged frame (fixed), with the "
        "temporal residue of the average removed."
    )
    if math.isfinite(result.banding_ratio) and result.banding_ratio > 0.15:
        result.note(
            Level.WARN,
            f"Random row noise is {result.banding_ratio:.0%} of the pixel "
            "read noise: horizontal banding can be visible in single frames. It averages "
            "out when stacking (dithering helps) but cannot be calibrated away.",
        )
    return result


def _detrend_rows(dev: np.ndarray) -> np.ndarray:
    return dev - dev.mean(axis=1, keepdims=True)


def _sqrt_value(var: float, var_err: float) -> Value:
    sigma = math.sqrt(max(var, 0.0))
    err = var_err / (2 * sigma) if sigma > 0 and math.isfinite(var_err) else math.nan
    return Value(sigma, err, "ADU")


def _fixed_pattern(
    result: PatternNoise, data: ModeData, st: SetStats, rn: ReadNoise, n_avg: int
) -> None:
    step = data.layout.step
    for i, (_name, plane) in enumerate(st.layout.planes(st.mean_map)):
        plane = plane.astype(np.float64)
        temporal = (rn.adu[i].value ** 2 / max(n_avg, 1)) if rn.adu else 0.0
        rows_raw = plane.mean(axis=1)
        cols_raw = plane.mean(axis=0)
        result.gradient_ptp.append(
            float(
                np.ptp(
                    ndimage.median_filter(
                        rows_raw, size=min(DETREND_ROWS, rows_raw.size), mode="nearest"
                    )
                )
            )
        )
        rows, cols = _detrend(rows_raw), _detrend(cols_raw)
        n_rows, n_cols = plane.shape
        hp = plane - large_scale(plane, 64)
        s2 = clipped_stats(hp[central_roi(hp.shape)]).std ** 2 - temporal
        s2_rav = float(np.var(rows)) - temporal / n_cols
        s2_cav = float(np.var(cols)) - temporal / n_rows
        s2_row, s2_col = max(s2_rav, 0.0), max(s2_cav, 0.0)
        for _ in range(5):  # EMVA eq. 42, solved iteratively
            s2_pix = max(s2 - s2_row - s2_col, 0.0)
            s2_row = max(s2_rav - s2_pix / n_cols, 0.0)
            s2_col = max(s2_cav - s2_pix / n_rows, 0.0)
        result.pixel_fixed.append(_sqrt_value(max(s2 - s2_row - s2_col, 0.0), math.nan))
        result.row_fixed.append(_sqrt_value(s2_row, math.nan))
        result.col_fixed.append(_sqrt_value(s2_col, math.nan))
        if i == 0:
            result.row_profile = rows_raw.astype(np.float32)
            result.col_profile = cols_raw.astype(np.float32)
            result.psd["row fixed"] = _psd(rows)
            result.psd["column fixed"] = _psd(cols)
            result.peaks += _find_peaks(*result.psd["row fixed"], n_rows, "row", "fixed", step)
            result.peaks += _find_peaks(
                *result.psd["column fixed"], n_cols, "column", "fixed", step
            )
            result.spectrum2d = spectrum_2d(hp)


def spectrum_2d(image: np.ndarray, size: int = 1024, out: int = 256) -> np.ndarray:
    """Centred log10 power spectrum of the central ``size``² crop, block-averaged to ``out``²."""
    h, w = image.shape
    n = min(size, h, w) // 2 * 2
    crop = image[(h - n) // 2 : (h + n) // 2, (w - n) // 2 : (w + n) // 2]
    crop = crop - np.median(crop)
    sigma = 5 * max(float(np.std(crop)), 1e-9)
    crop = np.clip(crop, -sigma, sigma)  # outliers would spread white power everywhere
    power = np.fft.fftshift(np.abs(np.fft.fft2(crop)) ** 2) / crop.size
    factor = max(1, n // out)
    m = n // factor * factor
    power = power[:m, :m].reshape(m // factor, factor, m // factor, factor).mean(axis=(1, 3))
    return np.log10(np.maximum(power, 1e-12)).astype(np.float32)
