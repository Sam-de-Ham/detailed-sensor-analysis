"""Dark current: level, dependence on exposure and temperature, per-pixel map and amp glow.

Three estimators, used as the data allows and cross-checked against each other:

* **Slope** (EMVA 1288 §7.1, preferred): mean dark signal vs exposure at one temperature. The
  intercept absorbs any difference between a bias and a true zero-exposure dark.
* **Level**: (dark − offset) / t from one exposure, assuming bias = zero-exposure dark.
* **Noise**: dark current adds shot noise, σ²_dark − σ²_bias = D·t / g² (ADU²). Independent of
  the offset level, so it still works when a camera clamps the mean (EMVA requires it then).
  Without bias frames it gives an upper limit on dark current *and* read noise together.

Temperature: D = D_ref · 2^((T − T_ref)/T_d) and the Arrhenius form D ∝ T^1.5 · exp(−E_a / kT),
fitted when darks span several temperatures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dsa.analysis.common import MISSING, Level, Section, Status, Value, quadrature
from dsa.analysis.data import ModeData
from dsa.analysis.gain import Gain
from dsa.analysis.levels import DataFormat, Offset, plane_level
from dsa.analysis.noise import ReadNoise, noise_source
from dsa.catalog import FrameSet
from dsa.frames import FrameType
from dsa.measure import SetStats
from dsa.stats import bin_image, fit_line, large_scale

BOLTZMANN_EV = 8.617333262e-5


@dataclass
class DarkPoint:
    label: str
    exposure: float
    temperature: float
    is_master: bool
    n: int
    signal_adu: float  # mean over planes, above offset (NaN when no offset)
    signal_err: float
    current: Value  # e⁻/s/px (or ADU/s when the gain is unknown) from the level
    noise_current: Value  # e⁻/s/px from the variance increase (may be an upper bound)


@dataclass
class AmpGlow:
    map: np.ndarray  # large-scale excess dark current, binned, e⁻/s (or ADU/s)
    bin_factor: int
    peak: float  # maximum excess, same unit
    peak_xy: tuple[float, float]  # full-resolution pixel position
    location: str  # "right edge", "top-left corner", "none"
    edge_excess: dict[str, float]  # mean excess in a band along each edge
    area_fraction: float  # fraction of the sensor with significant glow
    relative: float  # peak / median dark current
    scaling: list[tuple[float, float]] = field(default_factory=list)  # (exposure, peak signal)
    visible: bool = False  # stronger than one sub's noise: needs exactly matching darks


@dataclass
class DarkCurrent(Section):
    unit: str = "e⁻/s"
    points: list[DarkPoint] = field(default_factory=list)
    current: Value = MISSING  # best estimate, e⁻/s/px
    reference_temperature: float = math.nan
    estimator: str = ""
    slope_fits: list[dict] = field(default_factory=list)  # per temperature group
    doubling: Value = MISSING  # °C
    activation: Value = MISSING  # eV
    temp_points: np.ndarray | None = None  # [n, 3]: temperature, current, error
    map: np.ndarray | None = None  # per-pixel dark current, full resolution
    map_absolute: bool = False  # False: excess over the median (offset unknown)
    map_label: str = ""
    map_exposure: float = 0.0
    map_frames: int = 0
    sub_noise: float = math.nan  # noise of one sub of the map's darks, in map units
    clamped: bool = False  # the camera holds the dark level constant
    median_pixel: float = math.nan
    mean_pixel: float = math.nan
    hist: tuple[np.ndarray, np.ndarray] | None = None
    glow: AmpGlow | None = None
    read_noise_bound: Value = MISSING  # e⁻, only without bias frames


def _set_signal(st: SetStats, fs: FrameSet, offset: Offset) -> tuple[float, float]:
    """Dark signal above the offset (mean over planes, ADU) and its error."""
    level, err = plane_level(st)
    if offset.exact:
        sig = level - offset.level
        e = np.sqrt(err**2 + offset.error**2)
    elif offset.source == "raw black level" and np.isfinite(st.black_meta).any():
        # Black measured on this frame's own masked border (Canon); masked pixels also
        # collect some dark current, so this is a lower limit of the visible-area current.
        black = np.nanmedian(st.black_meta, axis=0)
        sig, e = level - black, err
    else:
        return math.nan, math.nan
    return float(np.mean(sig)), float(np.sqrt(np.sum(e**2)) / len(e))


def _pick_sets(data: ModeData) -> list[tuple[FrameSet, SetStats]]:
    """Every dark sub set; masters only for exposures that no sub set covers.

    A master made from the same subs adds no information, and its temperature is often
    missing, so it would otherwise appear as a separate point.
    """
    subs = data.of(FrameType.DARK, subs=True)
    covered = {round(fs.exposure or 0, 2) for fs, _ in subs}
    masters = [
        (fs, st)
        for fs, st in data.of(FrameType.DARK, subs=False)
        if round(fs.exposure or 0, 2) not in covered
    ]
    return sorted(subs + masters, key=lambda p: (p[0].exposure or 0, p[0].temperature or 0))


def dark_current(
    data: ModeData, gain: Gain, rn: ReadNoise, offset: Offset, fmt: DataFormat
) -> DarkCurrent:
    result = DarkCurrent()
    sets = [(fs, st) for fs, st in _pick_sets(data) if (fs.exposure or 0) > 0]
    if not sets:
        result.status = Status.UNAVAILABLE
        result.method = "Needs dark frames."
        result.unlock.append("Capture 10 or more darks at your usual sub length and temperature.")
        return result
    g = gain.gain if gain.gain.ok else Value(1.0, 0.0, "e⁻/ADU")
    if not gain.gain.ok:
        result.unit = "ADU/s"
        result.note(Level.INFO, "Gain unknown at this setting, so dark current is in ADU/s.")
    unit = result.unit
    bias_src = noise_source(data)
    bias_var = None
    if bias_src is not None and bias_src[0].kind is FrameType.BIAS:
        bias_var = np.mean(bias_src[1].pair_var, axis=0)

    for fs, st in sets:
        t = float(fs.exposure)
        signal, err = _set_signal(st, fs, offset)
        level_current = Value(math.nan)
        if math.isfinite(signal):
            value = signal * g.value / t
            rel = quadrature(err / signal if signal else math.nan, g.error / g.value)
            level_current = Value(value, abs(value) * rel, unit + "/px")
        noise_current = Value(math.nan)
        if st.n_pairs > 0 and gain.gain.ok:
            var = np.mean(st.pair_var, axis=0)
            var_err = (
                np.std(st.pair_var, axis=0, ddof=1) / math.sqrt(st.n_pairs)
                if st.n_pairs > 1
                else var * math.sqrt(2 / st.pair_npix[0])
            )
            if bias_var is not None:
                excess = float(np.mean(var - bias_var))
                value = g.value**2 * excess / t
                rho = float(np.nanmean(np.abs(np.concatenate([st.pair_rho_x, st.pair_rho_y]))))
                # Spatial filtering in the camera removes variance: then this is a lower limit.
                bound = "lower" if rho > 0.05 else None
                noise_current = Value(
                    value, g.value**2 * float(np.mean(var_err)) / t, "e⁻/s/px", bound
                )
                if bound:
                    result.note(
                        Level.WARN,
                        f"{fs.label}: neighbouring pixels' noise is correlated ({rho:.2f}), so "
                        "the camera filtered these darks (e.g. Sony's long-exposure 'star "
                        "eater'). Filtering removes noise, so noise-based dark current is a "
                        "lower limit and some hot pixels are hidden.",
                    )
            else:
                q = fmt.container_scale**2 / 12
                total = float(np.mean(var)) - q
                noise_current = Value(g.value**2 * total / t, math.nan, "e⁻/s/px", "upper")
                result.read_noise_bound = Value(
                    math.sqrt(max(total, 0)) * g.value, math.nan, "e⁻", "upper"
                )
        result.points.append(
            DarkPoint(
                fs.label,
                t,
                fs.temperature if fs.temperature is not None else math.nan,
                fs.is_master,
                st.n_combined or st.n_frames,
                signal,
                err,
                level_current,
                noise_current,
            )
        )

    _slopes(result, g, unit)
    _choose(result, offset)
    _temperature(result)
    _map_and_glow(result, data, sets, g, offset, rn.combined_adu.value)
    return result


def _groups(points: list[DarkPoint]) -> list[list[DarkPoint]]:
    """Points within 2 °C of each other (or all, when temperatures are unknown)."""
    groups: list[list[DarkPoint]] = []
    for p in sorted(points, key=lambda p: p.temperature if math.isfinite(p.temperature) else 0):
        if groups and (
            not math.isfinite(p.temperature) or abs(p.temperature - groups[-1][0].temperature) <= 2
        ):
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


def _slopes(result: DarkCurrent, g: Value, unit: str) -> None:
    for group in _groups(result.points):
        pts = [p for p in group if math.isfinite(p.signal_adu)]
        exposures = {round(p.exposure, 2) for p in pts}
        if len(exposures) < 2:
            continue
        t = np.array([p.exposure for p in pts])
        s = np.array([p.signal_adu for p in pts])
        e = np.array([max(p.signal_err, 1e-6) for p in pts])
        line = fit_line(t, s, e if len(pts) > 2 else None)
        slope_err = (
            line.slope_err
            if math.isfinite(line.slope_err)
            else float(math.sqrt(e[0] ** 2 + e[-1] ** 2) / np.ptp(t))
        )
        temps = [p.temperature for p in pts if math.isfinite(p.temperature)]
        result.slope_fits.append(
            {
                "temperature": float(np.mean(temps)) if temps else math.nan,
                "exposure": t,
                "signal": s,
                "error": e,
                "slope": line.slope,
                "intercept": line.intercept,
                "current": Value(
                    line.slope * g.value,
                    quadrature(slope_err * g.value, line.slope * g.error),
                    unit + "/px",
                ),
            }
        )


def _is_clamped(points: list[DarkPoint]) -> bool:
    """The dark *level* stays at the bias while the dark *noise* grows: level clamping.

    Some cameras (e.g. Canon DSLRs) hold the black level of the image constant whatever the
    exposure, which removes the dark signal from the mean but not its shot noise.
    """
    pairs = [
        (p.current, p.noise_current)
        for p in points
        if p.current.ok and p.noise_current.ok and p.noise_current.bound != "upper"
    ]
    if not pairs:
        return False
    return all(
        n.value > 3 * max(n.error, 0) and n.value > 0 and level.value < 0.3 * n.value
        for level, n in pairs
    )


def _choose(result: DarkCurrent, offset: Offset) -> None:
    longest = max(result.points, key=lambda p: p.exposure)
    result.reference_temperature = longest.temperature
    result.clamped = _is_clamped(result.points)
    if result.clamped:
        best = max((p for p in result.points if p.noise_current.ok), key=lambda p: p.exposure)
        result.current = best.noise_current
        result.reference_temperature = best.temperature
        result.estimator = "noise"
        result.status = Status.MEASURED
        result.method = (
            "The camera holds the dark level at the bias (black clamping), so dark current is "
            "measured from the shot noise it adds: D = g²·(σ²_dark − σ²_bias)/t (EMVA 1288 "
            f"§7.1), from {best.label}."
        )
        result.note(
            Level.INFO,
            "Dark level clamped by the camera: the mean of a dark doesn't "
            "rise with exposure, but its noise does. Dark frames still remove hot pixels "
            "and pattern; the clamped level also means the sky level in lights is "
            "measured against the bias.",
        )
        return
    if result.slope_fits:
        fit = max(result.slope_fits, key=lambda f: np.ptp(f["exposure"]))
        result.current = fit["current"]
        result.reference_temperature = fit["temperature"]
        result.estimator = "slope"
        result.status = Status.MEASURED
        n = len(fit["exposure"])
        temp = f", {fit['temperature']:.0f} °C" if math.isfinite(fit["temperature"]) else ""
        result.method = (
            f"Slope of mean dark signal vs exposure ({n} exposures{temp}), × gain. The fit's "
            "intercept absorbs bias/zero-exposure differences."
        )
    elif longest.current.ok:
        result.current = longest.current
        result.estimator = "level"
        result.status = Status.MEASURED if offset.exact else Status.LIMITED
        result.method = (
            f"(mean dark − {offset.source}) / exposure × gain, from "
            f"{longest.label}. Assumes the bias equals a zero-length dark."
        )
        if offset.source == "raw black level":
            result.note(
                Level.INFO,
                "Offset taken from the camera's masked-pixel black level. "
                "Masked pixels collect dark current too, so this underestimates it.",
            )
        result.unlock.append(
            "Darks at two or more exposure times (e.g. 60 s and your sub "
            "length) give the dark current as a slope, independent of the bias."
        )
    elif longest.noise_current.ok:
        result.current = longest.noise_current
        result.estimator = "noise"
        result.status = Status.LIMITED
        if longest.noise_current.bound:
            result.method = (
                f"No bias frames, so the dark signal can't be separated from the "
                f"offset. The temporal noise of {longest.label} bounds read noise and "
                "dark current together: RN² + D·t ≤ g²·σ²_dark."
            )
            result.unlock.append(
                "Bias frames (or darks at a second, much shorter exposure) at "
                "this gain, offset and temperature."
            )
        else:
            result.method = "Increase of temporal noise over bias: D = g²(σ²_dark − σ²_bias)/t."
    else:
        result.status = Status.UNAVAILABLE
        result.method = "Needs an offset reference (bias) or gain to measure dark current."
        result.unlock.append("Bias frames at this setting.")
    # Cross-check level/slope against the noise estimate when both exist.
    noise = [
        p.noise_current for p in result.points if p.noise_current.ok and not p.noise_current.bound
    ]
    if result.estimator in ("slope", "level") and noise:
        best = max((p for p in result.points if p.noise_current.ok), key=lambda p: p.exposure)
        a, b = result.current, best.noise_current
        if a.ok and b.ok and a.value > 0:
            sigma = quadrature(a.error, b.error)
            if abs(a.value - b.value) > 3 * sigma and abs(a.value - b.value) > 0.25 * abs(a.value):
                result.note(
                    Level.WARN,
                    f"Signal-based ({a.format()}) and noise-based "
                    f"({b.format()}) dark current disagree. Common causes: the camera "
                    "clamps the dark level (signal low), RTS or hot pixels (noise high), "
                    "or the bias differs from a zero-exposure dark.",
                )


def _temperature(result: DarkCurrent) -> None:
    """Doubling temperature and activation energy from currents at different temperatures."""
    pts = []
    if result.slope_fits and result.estimator != "noise":
        pts = [(f["temperature"], f["current"]) for f in result.slope_fits]
    for p in result.points:  # sets at temperatures without a slope fit
        value = p.noise_current if result.estimator == "noise" else p.current
        if (
            math.isfinite(p.temperature)
            and value.ok
            and not value.bound
            and not any(abs(p.temperature - t) <= 2 for t, _ in pts)
        ):
            pts.append((p.temperature, value))
    pts = [(t, v) for t, v in pts if math.isfinite(t) and v.ok and v.value > 0]
    if len(pts) < 2 or max(t for t, _ in pts) - min(t for t, _ in pts) < 3:
        if any(math.isfinite(p.temperature) for p in result.points):
            result.unlock.append(
                "Darks at three or more sensor temperatures (≥ 5 °C apart) "
                "give the doubling temperature."
            )
        return
    temps = np.array([t for t, _ in pts])
    cur = np.array([v.value for _, v in pts])
    err = np.array(
        [v.error if math.isfinite(v.error) and v.error > 0 else 0.1 * v.value for _, v in pts]
    )
    result.temp_points = np.column_stack([temps, cur, err])
    log_err = err / cur
    line = fit_line(temps, np.log(cur), log_err if len(pts) > 2 else None)
    if line.slope > 0:
        doubling = math.log(2) / line.slope
        d_err = (
            doubling * line.slope_err / line.slope if math.isfinite(line.slope_err) else math.nan
        )
        result.doubling = Value(doubling, d_err, "°C")
    kelvin = temps + 273.15
    arr = fit_line(1 / kelvin, np.log(cur / kelvin**1.5), log_err if len(pts) > 2 else None)
    result.activation = Value(-arr.slope * BOLTZMANN_EV, arr.slope_err * BOLTZMANN_EV, "eV")


def _sub_noise(st: SetStats, is_master: bool, rn_adu: float) -> float:
    """Typical temporal noise of one sub (ADU), measured per pixel.

    A master has no per-frame noise left; read noise, the floor of a sub's noise, stands in.
    """
    if st.var_map is not None and not is_master:
        return float(np.sqrt(np.median(st.var_map)))
    return rn_adu


def _map_and_glow(
    result: DarkCurrent, data: ModeData, sets, g: Value, offset: Offset, rn_adu: float
) -> None:
    """Per-pixel dark current from the darks with the most signal, and amp glow."""
    fs, st = max(
        sets, key=lambda p: (p[0].exposure or 0) * math.sqrt(max(p[1].n_frames, p[1].n_combined))
    )
    t = float(fs.exposure)
    image = st.mean_map.astype(np.float32)
    if offset.map is not None and offset.map.shape == image.shape:
        signal = image - offset.map
        result.map_absolute = True
    elif offset.exact or offset.source == "raw black level":
        signal = image.copy()
        for i, (_, plane) in enumerate(st.layout.planes(signal)):
            black = (
                np.nanmedian(st.black_meta[:, i])
                if offset.source == "raw black level" and np.isfinite(st.black_meta[:, i]).any()
                else offset.level[i]
            )
            plane -= np.float32(black)
        result.map_absolute = True
    else:
        signal = image.copy()
        for _, plane in st.layout.planes(signal):
            plane -= np.float32(np.median(plane))
    dmap = signal * np.float32(g.value / t)
    result.map = dmap
    result.map_label = fs.label
    result.map_exposure = t
    result.map_frames = st.n_combined if fs.is_master else st.n_frames
    result.sub_noise = _sub_noise(st, fs.is_master, rn_adu) * g.value / t
    result.median_pixel = float(np.median(dmap))
    result.mean_pixel = float(np.mean(dmap))
    lo, hi = np.percentile(dmap, [0.01, 99.999])
    span = max(hi - lo, 1e-9)
    edges = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 400)
    counts, _ = np.histogram(dmap, bins=edges)
    result.hist = (edges, counts)
    result.glow = amp_glow(dmap, data.layout.step, result.median_pixel)
    result.glow.visible = result.glow.location != "none" and result.glow.peak >= result.sub_noise
    # How the glow scales with exposure (dark scaling validity): peak glow signal per exposure,
    # at the map's temperature (glow also depends on temperature).
    for fs2, st2 in sets:
        if (
            fs.temperature is not None
            and fs2.temperature is not None
            and abs(fs2.temperature - fs.temperature) > 2
        ):
            continue
        excess = st2.mean_map.astype(np.float32)
        lum = _luminance(excess, data.layout.step)
        ls = large_scale(lum, 64)
        peak = float(ls.max() - np.median(ls))
        result.glow.scaling.append((float(fs2.exposure), peak))


def _luminance(image: np.ndarray, step: int) -> np.ndarray:
    """Average of the colour planes at plane resolution (the image itself for mono)."""
    if step == 1:
        return image
    return bin_image(image, 2)


def amp_glow(dmap: np.ndarray, step: int, median_current: float) -> AmpGlow:
    lum = _luminance(dmap, step)
    factor = max(1, max(lum.shape) // 480)
    ls = large_scale(lum, 64)
    base = float(
        np.median(ls[ls.shape[0] // 4 : -ls.shape[0] // 4, ls.shape[1] // 4 : -ls.shape[1] // 4])
    )
    excess = ls - base
    binned = bin_image(excess, factor)
    peak_idx = np.unravel_index(int(np.argmax(excess)), excess.shape)
    peak = float(excess[peak_idx])
    h, w = excess.shape
    band_h, band_w = max(1, h // 20), max(1, w // 20)
    edges = {
        "top": float(excess[:band_h].mean()),
        "bottom": float(excess[-band_h:].mean()),
        "left": float(excess[:, :band_w].mean()),
        "right": float(excess[:, -band_w:].mean()),
    }
    noise = float(np.std(excess[h // 4 : -h // 4, w // 4 : -w // 4])) or 1e-12
    threshold = max(5 * noise, 0.25 * abs(median_current))
    significant = excess > threshold
    y, x = peak_idx
    location = "none"
    if peak > threshold:
        vert = "top" if y < h / 3 else "bottom" if y > 2 * h / 3 else ""
        horiz = "left" if x < w / 3 else "right" if x > 2 * w / 3 else ""
        location = (
            f"{vert}-{horiz} corner"
            if vert and horiz
            else f"{vert or horiz} edge"
            if vert or horiz
            else "centre"
        )
    return AmpGlow(
        map=binned,
        bin_factor=factor * step,
        peak=peak,
        peak_xy=(float(x * step), float(y * step)),
        location=location,
        edge_excess=edges,
        area_fraction=float(significant.mean()),
        relative=peak / median_current if median_current > 0 else math.nan,
    )


def summarise_glow(glow: AmpGlow | None, unit: str) -> str:
    if glow is None or glow.location == "none":
        return "No significant amp glow."
    rel = f" ({glow.relative:.1f}× the median)" if math.isfinite(glow.relative) else ""
    return (
        f"Amp glow at the {glow.location}: +{glow.peak:.3g} {unit}{rel}, "
        f"{glow.area_fraction:.1%} of the sensor."
    )


def scaling_check(glow: AmpGlow | None) -> list[tuple[float, float, float]]:
    """(exposure, peak glow, peak per second) — constant per-second values mean it scales."""
    if glow is None:
        return []
    return [(t, p, p / t if t else math.nan) for t, p in glow.scaling]
