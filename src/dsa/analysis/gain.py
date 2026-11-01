"""Conversion gain (e⁻/ADU), photon transfer, saturation, linearity and dynamic range.

Gain uses the photon transfer relation for a shot-noise-limited signal: the temporal variance of
S electrons is S, so in ADU  σ²(S) − σ²(0) = S_ADU / g. With flat pairs and an offset
reference (Janesick; EMVA 1288 eq. 18):

    g = (μ_F − μ_offset) / (σ²_F − σ²_B),  σ²_F = var(F₁ − F₂)/2,  σ²_B = var(B₁ − B₂)/2

Differences cancel PRNU and all fixed patterns. Each disjoint flat pair gives one estimate; sets
taken through different filters (all at one level, as NINA's flat wizard makes them) are
independent repeats, so their agreement is a real test.

A single flat level gives no curve, but vignetting does: splitting the frame into tiles spreads
the signal over the illumination falloff (typically 20-50 %). Variance vs mean of all tiles is a
short photon transfer curve whose slope must agree with the pair estimate. A true multi-level
series additionally gives saturation (the PTC turnover) and linearity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dsa.analysis.common import MISSING, Level, Section, Status, Value, quadrature, weighted_mean
from dsa.analysis.data import ModeData
from dsa.analysis.levels import DataFormat, Offset, find_offset, quantisation_variance
from dsa.analysis.noise import ReadNoise, noise_source
from dsa.frames import FrameType
from dsa.measure import SetStats
from dsa.stats import fit_line, fit_proportional, robust_sigma


@dataclass
class SetGain:
    label: str
    filter: str | None
    exposure: float
    n_pairs: int
    level_adu: list[float]  # signal above offset per plane
    level_fraction: float  # of the usable range (offset .. white)
    gain: list[Value]  # per plane, e⁻/ADU
    combined: Value
    pair_gain: np.ndarray  # [pairs, planes]
    saturated: float  # fraction of pixels at the white level
    stability: float  # frame-to-frame level rms / level
    correlation: float  # mean lag-1 correlation of difference images
    offset_source: str


@dataclass
class Linearity:
    filter: str | None
    exposure: np.ndarray
    signal: np.ndarray  # ADU above offset
    fit_slope: float
    fit_intercept: float
    deviation: np.ndarray  # % of fit
    error_pct: float  # EMVA linearity error: half the spread of deviations in 5-95 %


@dataclass
class Gain(Section):
    planes: list[str] = field(default_factory=list)
    container_scale: int = 1
    per_set: list[SetGain] = field(default_factory=list)
    gain: Value = MISSING  # e⁻ per file ADU
    gain_planes: list[Value] = field(default_factory=list)
    chi2: float = math.nan
    tile_gain: Value = MISSING
    tile_intercept: float = math.nan  # ADU² (should equal ~0 after read-noise subtraction)
    ptc_signal: np.ndarray | None = None  # tile points, ADU above offset
    ptc_variance: np.ndarray | None = None  # ADU², read noise removed
    ptc_set: np.ndarray | None = None  # set index of each tile point
    ptc_plane: np.ndarray | None = None
    set_points: np.ndarray | None = None  # [sets, 2] (signal, variance) central-ROI means
    read_variance: list[float] = field(default_factory=list)
    offset: Offset | None = None
    estimated_from: str = ""  # when the gain is scaled from another setting
    multi_level: bool = False
    saturation_signal: float = math.nan  # ADU above offset at the PTC turnover
    linearity: list[Linearity] = field(default_factory=list)

    @property
    def native(self) -> Value:
        """Gain in native ADC units (e⁻ per ADC step)."""
        return self.gain.scaled(self.container_scale)


def _offset_variance(data: ModeData, rn: ReadNoise, scale: int) -> tuple[list[float], str]:
    """σ²_B per plane: the temporal variance of the offset frames, quantisation included."""
    source = noise_source(data)
    if source is not None:
        return [float(np.mean(source[1].pair_var[:, i])) for i in range(len(data.planes))], source[
            0
        ].label
    return [scale**2 / 12] * len(data.planes), ""


def measure_gain(data: ModeData, fmt: DataFormat, rn: ReadNoise) -> Gain:
    result = Gain(planes=data.planes, container_scale=fmt.container_scale)
    flats = [(fs, st) for fs, st in data.of(FrameType.FLAT) if st.n_pairs > 0]
    if not flats:
        result.status = Status.UNAVAILABLE
        result.method = "Needs at least two flat frames at the same level."
        result.unlock.append(
            "Capture 10 or more flats at this setting (any filter, 30–60 % of full scale)."
        )
        return result

    read_var, read_label = _offset_variance(data, rn, fmt.container_scale)
    result.read_variance = read_var
    if not read_label:
        result.note(
            Level.INFO,
            "No bias or dark pairs: the read-noise term of the flat variance "
            "is taken as quantisation only. At flat levels shot noise dominates, so the "
            "effect on gain is far below 1 %.",
        )

    tile_s, tile_v, tile_set, tile_plane, set_points = [], [], [], [], []
    for k, (fs, st) in enumerate(flats):
        offset = find_offset(data, exposure=fs.exposure)
        result.offset = result.offset or offset
        if not offset.known:
            result.note(
                Level.WARN,
                f"{fs.label}: no offset reference (bias, dark or raw black "
                "level), so the signal level is unknown.",
            )
            continue
        sg = _set_gain(fs.label, fs, st, offset, read_var, fmt, data)
        result.per_set.append(sg)
        set_points.append(
            (
                float(np.mean(sg.level_adu)),
                float(np.mean(sg.level_adu) / sg.combined.value) if sg.combined.ok else math.nan,
            )
        )
        if st.tile_mean is not None:
            s, v, p = _tile_points(st, offset, read_var, fmt)
            tile_s.append(s)
            tile_v.append(v)
            tile_plane.append(p)
            tile_set.append(np.full(s.size, k))

    if not result.per_set:
        result.status = Status.UNAVAILABLE
        result.method = "Flats exist, but there is no offset reference to measure their signal."
        result.unlock.append("Capture bias frames (or dark flats) at this setting.")
        return result

    values = np.array([s.combined.value for s in result.per_set])
    errors = np.array([s.combined.error for s in result.per_set])
    mean, err, chi2 = weighted_mean(values, errors)
    result.gain = Value(mean, err, "e⁻/ADU")
    result.chi2 = chi2
    for i in range(len(data.planes)):
        v = np.array([s.gain[i].value for s in result.per_set])
        e = np.array([s.gain[i].error for s in result.per_set])
        m, er, _ = weighted_mean(v, e)
        result.gain_planes.append(Value(m, er, "e⁻/ADU"))
    result.set_points = np.array(set_points)

    in_range = _multi_level(result)
    if tile_s:
        sets_idx = np.concatenate(tile_set)
        use = np.isin(sets_idx, np.flatnonzero(in_range))
        _tile_fit(
            result,
            np.concatenate(tile_s)[use],
            np.concatenate(tile_v)[use],
            sets_idx[use],
            np.concatenate(tile_plane)[use],
        )
    _gain_notes(result, fmt, flats)
    return result


def _set_gain(
    label: str,
    fs,
    st: SetStats,
    offset: Offset,
    read_var: list[float],
    fmt: DataFormat,
    data: ModeData,
) -> SetGain:
    planes = len(data.planes)
    pair_gain = np.full((st.n_pairs, planes), np.nan)
    gains, levels = [], []
    for i in range(planes):
        signal = st.pair_mean[:, i] - offset.level[i]
        q_extra = 0.0
        if fmt.lossy:  # signal-dependent quantisation of lossy raw doesn't cancel against bias
            q_flat = quantisation_variance(
                st,
                i,
                float(np.mean(st.pair_mean[:, i])),
                math.sqrt(float(np.mean(st.pair_var[:, i]))),
            )
            q_extra = q_flat - fmt.container_scale**2 / 12
        variance = st.pair_var[:, i] - read_var[i] - q_extra
        g = signal / variance
        pair_gain[:, i] = g
        levels.append(float(np.mean(signal)))
        if g.size > 1:
            gains.append(
                Value(float(np.mean(g)), float(np.std(g, ddof=1) / math.sqrt(g.size)), "e⁻/ADU")
            )
        else:  # one pair: the error of a variance from N pixels
            rel = math.sqrt(2 / max(st.pair_npix[0, i], 2))
            rel = quadrature(rel, float(offset.error[i] / signal[0]))
            gains.append(Value(float(g[0]), abs(float(g[0])) * rel, "e⁻/ADU"))
    mean, err, _ = weighted_mean(
        np.array([g.value for g in gains]), np.array([g.error for g in gains])
    )
    usable = st.white_level - float(np.mean(offset.level))
    level_rms = np.std(st.frame_mean, axis=0, ddof=1) if st.n_frames > 1 else np.zeros(planes)
    return SetGain(
        label=label,
        filter=fs.filter,
        exposure=float(fs.exposure or 0.0),
        n_pairs=st.n_pairs,
        level_adu=levels,
        level_fraction=float(np.mean(levels)) / usable if usable > 0 else math.nan,
        gain=gains,
        combined=Value(mean, err, "e⁻/ADU"),
        pair_gain=pair_gain,
        saturated=float(np.mean(st.frac_high)),
        stability=float(np.mean(level_rms / np.maximum(np.array(levels), 1e-9))),
        correlation=float(np.nanmean(np.abs(np.concatenate([st.pair_rho_x, st.pair_rho_y])))),
        offset_source=offset.source,
    )


def _tile_points(st: SetStats, offset: Offset, read_var: list[float], fmt: DataFormat):
    s_all, v_all, p_all = [], [], []
    for i in range(st.tile_mean.shape[1]):
        s = st.tile_mean[:, i].mean(axis=0) - offset.level[i]
        v = st.tile_var[:, i].mean(axis=0) - read_var[i]
        good = np.isfinite(s) & np.isfinite(v) & (s > 0)
        s_all.append(s[good].ravel())
        v_all.append(v[good].ravel())
        p_all.append(np.full(int(good.sum()), i))
    return np.concatenate(s_all), np.concatenate(v_all), np.concatenate(p_all)


def _tile_fit(
    result: Gain, s: np.ndarray, v: np.ndarray, sets: np.ndarray, planes: np.ndarray
) -> None:
    """Robust straight-line fit of tile variance vs signal; rejects dust and defect tiles."""
    keep = np.ones(s.size, dtype=bool)
    for _ in range(5):
        k, _ = fit_proportional(s[keep], v[keep])
        resid = (v - k * s) / np.maximum(k * s, 1e-12)
        sigma = robust_sigma(resid[keep])
        new_keep = np.abs(resid - np.median(resid[keep])) < 4 * sigma
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    line = fit_line(s[keep], v[keep])
    k, k_err = fit_proportional(s[keep], v[keep])
    # The slope error from ~10⁴ tiles is tiny; tiles within a frame aren't fully independent
    # (shared offset error), so the per-set scatter of the pair method stays the main error.
    result.tile_gain = Value(1 / k, k_err / k**2, "e⁻/ADU")
    result.tile_intercept = line.intercept
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(keep)
    if idx.size > 20_000:
        idx = np.sort(rng.choice(idx, 20_000, replace=False))
    result.ptc_signal = s[idx].astype(np.float32)
    result.ptc_variance = v[idx].astype(np.float32)
    result.ptc_set = sets[idx].astype(np.int16)
    result.ptc_plane = planes[idx].astype(np.int8)


def _ptc_slope(signal: np.ndarray, variance: np.ndarray, rel_err: np.ndarray) -> Value:
    """Gain from the photon-transfer slope: weighted fit of variance = signal / g."""
    sigma = np.maximum(variance * rel_err, 1e-12)
    w = 1 / sigma**2
    k = float(np.sum(w * signal * variance) / np.sum(w * signal**2))
    k_err = math.sqrt(1 / float(np.sum(w * signal**2)))
    if signal.size > 1:
        chi2 = float(np.sum(w * (variance - k * signal) ** 2)) / (signal.size - 1)
        k_err *= math.sqrt(max(chi2, 1.0))
    return Value(1 / k, k_err / k**2, "e⁻/ADU")


def _multi_level(result: Gain) -> np.ndarray:
    """Saturation, gain over the linear range and linearity when flats span several levels.

    Returns which sets lie in the linear photon-transfer range (all of them for single-level
    data). With a series, the gain is the slope of variance vs signal up to 70 % of saturation
    (EMVA 1288 §6.6) instead of an average of per-set gains: above full well the variance
    collapses and per-set gains become meaningless.
    """
    n = len(result.per_set)
    in_range = np.ones(n, dtype=bool)
    levels = np.array([np.mean(s.level_adu) for s in result.per_set])
    if n < 3 or levels.max() / max(levels.min(), 1e-9) < 3:
        return in_range
    result.multi_level = True
    variances = np.array([np.mean(s.level_adu) / s.combined.value for s in result.per_set])
    order = np.argsort(levels)
    peak = int(order[np.nanargmax(variances[order])])
    if peak != order[-1]:  # the variance falls again after its peak: saturation turnover
        result.saturation_signal = float(levels[peak])
        in_range = levels <= 0.7 * result.saturation_signal
    in_range &= levels > 0
    if in_range.sum() >= 2:
        rel = np.array([s.combined.error / s.combined.value for s in result.per_set])
        result.gain = _ptc_slope(levels[in_range], variances[in_range], rel[in_range])
        result.gain_planes = []
        for i in range(len(result.planes)):
            lv = np.array([s.level_adu[i] for s in result.per_set])
            gv = np.array([s.gain[i].value for s in result.per_set])
            ge = np.array([s.gain[i].error / s.gain[i].value for s in result.per_set])
            sel = in_range & (gv > 0)
            result.gain_planes.append(_ptc_slope(lv[sel], lv[sel] / gv[sel], ge[sel]))
        result.chi2 = math.nan
    by_filter: dict[str | None, list[SetGain]] = {}
    for s in result.per_set:
        by_filter.setdefault(s.filter, []).append(s)
    sat = result.saturation_signal if math.isfinite(result.saturation_signal) else levels.max()
    for filt, sets in by_filter.items():
        if len(sets) < 3:
            continue
        t = np.array([s.exposure for s in sets])
        sig = np.array([np.mean(s.level_adu) for s in sets])
        order = np.argsort(t)
        t, sig = t[order], sig[order]
        use = (sig > 0.05 * sat) & (sig < 0.95 * sat)  # EMVA linearity range
        if use.sum() < 3:
            continue
        line = fit_line(t[use], sig[use])
        deviation = (sig - line(t)) / line(t) * 100
        dev_use = deviation[use]
        result.linearity.append(
            Linearity(
                filt,
                t,
                sig,
                line.slope,
                line.intercept,
                deviation,
                float((dev_use.max() - dev_use.min()) / 2),
            )
        )
    return in_range


def _gain_notes(result: Gain, fmt: DataFormat, flats) -> None:
    n_sets = len(result.per_set)
    total_pairs = sum(s.n_pairs for s in result.per_set)
    result.status = Status.MEASURED
    result.method = (
        f"Photon transfer from {total_pairs} flat pairs in {n_sets} set"
        f"{'s' if n_sets > 1 else ''}: (μ_F − μ_offset)/(σ²_F − σ²_B), per colour "
        "plane, central 50 % of the frame; cross-checked by the slope of variance vs "
        "signal over tiles."
    )
    if fmt.lossy:
        result.status = Status.LIMITED
        result.note(Level.WARN, fmt.lossy_note)
    if result.multi_level:
        result.method = (
            f"Photon transfer curve from {n_sets} signal levels: slope of temporal variance vs "
            "signal up to 70 % of saturation (EMVA 1288), per colour plane."
        )
    if n_sets > 1 and math.isfinite(result.chi2) and result.chi2 > 4:
        result.note(
            Level.WARN,
            f"The flat sets disagree more than their errors allow "
            f"(χ²/dof = {result.chi2:.1f}); the uncertainty includes that scatter.",
        )
    if result.tile_gain.ok and result.gain.ok:
        diff = result.tile_gain.value / result.gain.value - 1
        if abs(diff) > 0.03:
            result.note(
                Level.WARN,
                f"The tile photon-transfer slope differs from the pair "
                f"estimate by {diff:+.1%}: suggests non-linearity, an offset error or "
                "illumination changing between frames.",
            )
    for s in result.per_set:
        if s.saturated > 0.001:
            result.note(
                Level.PROBLEM,
                f"{s.label}: {s.saturated:.2%} of pixels are clipped; "
                "clipped pixels have no shot noise and bias the gain.",
            )
        if s.correlation > 0.05:
            result.note(
                Level.WARN,
                f"{s.label}: neighbouring pixels' noise is correlated "
                f"({s.correlation:.2f}). In-camera filtering or compression lowers "
                "variance, which makes the gain look too high.",
            )
        if not result.multi_level and s.level_fraction < 0.05:
            result.status = Status.LIMITED
            result.note(
                Level.WARN,
                f"{s.label}: the flat is only {s.level_fraction:.1%} of full scale. At so few "
                "electrons, small offset differences and non-linearity near black matter: treat "
                "this gain as uncertain at the few-% level (30–60 % flats are ideal).",
            )
        elif (
            not result.multi_level
            and math.isfinite(s.level_fraction)
            and not 0.15 <= s.level_fraction <= 0.85
        ):
            result.note(
                Level.INFO,
                f"{s.label}: flat level is {s.level_fraction:.0%} of full "
                "scale (30–60 % is ideal for gain).",
            )
    if not result.multi_level:
        result.unlock.append(
            "For a full photon transfer curve (saturation, linearity): flats at 8–20 exposure "
            "times from ~1 % to past saturation, with the same panel brightness and filter "
            "(two frames each)."
        )


def scale_gain(
    target: Gain, reference: Gain, ref_setting: float, setting: float, ref_label: str
) -> None:
    """Estimate ``target``'s gain from another setting, assuming gain ∝ 1 / analogue gain."""
    factor = ref_setting / setting
    target.gain = reference.gain.scaled(factor)
    target.gain_planes = [g.scaled(factor) for g in reference.gain_planes]
    target.estimated_from = ref_label
    target.status = Status.LIMITED
    target.method = (
        f"Not measured at this setting: scaled from {ref_label} assuming the gain "
        "is inversely proportional to the analogue gain (invalid across a "
        "conversion-gain switch)."
    )


# ----------------------------------------------------------------------- saturation --


@dataclass
class DynamicRange(Section):
    saturation_e: Value = MISSING
    saturation_adu: float = math.nan
    saturation_kind: str = ""  # "ADC clip" or "photon transfer"
    stops: Value = MISSING
    db: float = math.nan


def dynamic_range(gain: Gain, rn: ReadNoise, fmt: DataFormat, offset: Offset) -> DynamicRange:
    result = DynamicRange()
    if not gain.gain.ok or not offset.known:
        result.status = Status.UNAVAILABLE
        result.method = "Needs the gain (flats) and an offset reference."
        return result
    if math.isfinite(gain.saturation_signal):
        sat_adu, kind = gain.saturation_signal, "photon transfer"
    else:
        sat_adu, kind = fmt.white_level - float(np.mean(offset.level)), "ADC clip"
    result.saturation_adu = sat_adu
    result.saturation_kind = kind
    result.saturation_e = gain.gain.scaled(sat_adu, "e⁻")
    if rn.combined_e.ok:
        ratio = result.saturation_e.value / rn.combined_e.value
        rel = quadrature(
            gain.gain.error / gain.gain.value, rn.combined_e.error / rn.combined_e.value
        )
        bound = {"upper": "lower"}.get(rn.combined_e.bound or "")
        result.stops = Value(math.log2(ratio), rel / math.log(2), "stops", bound)
        result.db = 20 * math.log10(ratio)
        result.status = (
            Status.MEASURED if kind == "photon transfer" or not bound else Status.LIMITED
        )
    else:
        result.status = Status.LIMITED
        result.unlock.append("Dynamic range also needs read noise: capture bias frames.")
    if kind == "ADC clip":
        result.method = (
            "Saturation capacity = (white level − offset) × gain: the ADC clip, which "
            "can be below the pixel's physical full well. DR = log₂(saturation / "
            "read noise)."
        )
    else:
        result.method = (
            "Saturation at the photon-transfer turnover; DR = log₂(saturation / read noise)."
        )
    return result
