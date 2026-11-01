"""Checks on the data itself and on how well it serves calibration.

Each check states what was tested, the evidence, and a verdict. They catch the traps that make
sensor numbers (or calibrated images) wrong: offset clipping, lossy or filtered raw data, masters
that aren't what their header says, drifting bias, temperature mismatches, flickering flats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dsa.analysis.common import Level
from dsa.analysis.dark import DarkCurrent
from dsa.analysis.data import ModeData
from dsa.analysis.gain import Gain
from dsa.analysis.levels import DataFormat, Offset
from dsa.analysis.noise import ReadNoise
from dsa.catalog import format_exposure
from dsa.frames import Format, FrameType
from dsa.stats import central_roi, large_scale, robust_sigma


@dataclass
class Check:
    key: str
    title: str
    level: Level
    summary: str
    detail: str = ""


def run_checks(
    data: ModeData, fmt: DataFormat, offset: Offset, rn: ReadNoise, gain: Gain, dark: DarkCurrent
) -> list[Check]:
    checks: list[Check] = []
    checks += _format_checks(fmt)
    checks += _clipping(data)
    checks += _correlation(data)
    checks += _masters(data, rn)
    checks += _bias_stability(data, rn, gain)
    checks += _temperatures(data)
    checks += _flats(data, gain)
    checks += _dark_scaling(dark)
    checks += _calibration_coverage(data)
    checks += _black_level(data)
    order = {Level.PROBLEM: 0, Level.WARN: 1, Level.INFO: 2, Level.OK: 3}
    return sorted(checks, key=lambda c: order[c.level])


def _format_checks(fmt: DataFormat) -> list[Check]:
    out = []
    if fmt.integer:
        if fmt.container_scale > 1:
            out.append(
                Check(
                    "format",
                    "Number format",
                    Level.INFO,
                    f"{round(fmt.adc_bits)}-bit data stored ×{fmt.container_scale}",
                    fmt.notes[0].text if fmt.notes else "",
                )
            )
        else:
            out.append(
                Check(
                    "format",
                    "Number format",
                    Level.OK,
                    f"Every code used ({fmt.adc_bits:.1f} bits up to {fmt.white_level:.0f})",
                )
            )
    if fmt.lossy:
        out.append(
            Check(
                "lossy",
                "Lossy raw compression",
                Level.WARN,
                "Compressed raw: noise statistics are approximate",
                fmt.lossy_note,
            )
        )
    return out


def _clipping(data: ModeData) -> list[Check]:
    worst, label = 0.0, ""
    for fs, st in data.all(subs=True):
        if fs.kind in (FrameType.BIAS, FrameType.DARK, FrameType.DARK_FLAT):
            frac = float(st.frac_low.max())
            if frac >= worst:
                worst, label = frac, fs.label
    if not label:
        return []
    if worst > 0.005:
        level, text = Level.PROBLEM, f"{worst:.2%} of pixels at zero in {label}"
    elif worst > 1e-4:
        level, text = Level.WARN, f"{worst:.3%} of pixels at zero in {label}"
    else:
        level, text = Level.OK, "No pixels clipped at zero"
    detail = (
        "When the offset is too low, the noise distribution of bias and darks is cut at "
        "zero: read noise and dark current are underestimated and faint signal is lost. "
        "Raise the offset until no pixels sit at 0."
    )
    return [Check("clipping", "Offset clipping", level, text, detail)]


def _correlation(data: ModeData) -> list[Check]:
    worst, label = 0.0, ""
    for fs, st in data.all(subs=True):
        if st.n_pairs == 0 or fs.kind is FrameType.LIGHT:  # lights change between frames
            continue
        rho = float(np.nanmean(np.abs(np.concatenate([st.pair_rho_x, st.pair_rho_y]))))
        if rho > worst:
            worst, label = rho, fs.label
    if not label:
        return []
    detail = (
        "Read and shot noise are independent from pixel to pixel, so the difference of two "
        "frames should show no correlation between neighbouring same-colour pixels. "
        "Correlation means the camera filtered or compressed the raw data (e.g. Sony's "
        "'star eater' on long exposures), which lowers the measured variance and inflates "
        "the gain."
    )
    if worst > 0.05:
        return [
            Check(
                "filtering",
                "In-camera filtering",
                Level.WARN,
                f"Neighbouring pixels correlated ({worst:.2f}) in {label}",
                detail,
            )
        ]
    return [
        Check(
            "filtering",
            "In-camera filtering",
            Level.OK,
            f"Pixel noise independent (|ρ| ≤ {worst:.3f})",
            detail,
        )
    ]


def _hp_variance(image: np.ndarray, layout) -> float:
    name = layout.names[0]
    plane = layout.plane(image, name).astype(np.float32)
    hp = plane - large_scale(plane, 32)
    return robust_sigma(hp[central_roi(hp.shape)]) ** 2


def _masters(data: ModeData, rn: ReadNoise) -> list[Check]:
    out = []
    for fs in data.sets:
        if not fs.quantitative:
            out.append(
                Check(
                    f"master-{fs.key}",
                    fs.label,
                    Level.INFO,
                    "Listed, not used for measurements",
                    f"{fs.frames[0].path.name} was made by "
                    f"{fs.frames[0].software or 'another tool'}, which rescales values and moves "
                    "the black point to 0, so its numbers aren't raw ADU. The sub-frames are "
                    "analysed instead.",
                )
            )
    for fs, st in data.all(subs=False):
        subs = [
            (f, s)
            for f, s in data.of(fs.kind)
            if fs.exposure is None
            or f.exposure is None
            or abs((f.exposure or 0) - (fs.exposure or 0)) <= 0.05 * max(fs.exposure or 0, 1e-6)
        ]
        if fs.kind is FrameType.FLAT:
            subs = [(f, s) for f, s in subs if f.filter == fs.filter]
        title = f"{fs.label}"
        claimed = fs.n_combined
        if not subs:
            out.append(
                Check(
                    f"master-{fs.key}",
                    title,
                    Level.INFO,
                    f"Master{' of ' + str(claimed) if claimed else ''} (no subs to compare)",
                    f"{fs.frames[0].path.name}: used where no sub-frames exist.",
                )
            )
            continue
        sf, ss = max(subs, key=lambda p: p[1].n_frames)
        ratio = float(np.median(st.mean_map) / max(np.median(ss.mean_map), 1e-9))
        lines = [f"Level: {ratio:.3f}× the {sf.label}."]
        level = Level.OK
        if not 0.97 < ratio < 1.03:
            if fs.kind is FrameType.FLAT and fs.frames[0].format is Format.FITS:
                lines.append("A normalised or offset-subtracted master flat; fine for calibration.")
            else:
                level = Level.WARN
                lines.append(
                    "The master's level differs from the subs: different settings, a "
                    "scaled master, or a different session."
                )
        if fs.kind in (FrameType.BIAS, FrameType.DARK) and ss.n_frames >= 3 and rn.adu:
            temporal = float(np.mean(ss.pair_var[:, 0]))
            fpn = _hp_variance(ss.mean_map, ss.layout) - temporal / ss.n_frames
            master_var = _hp_variance(st.mean_map, st.layout)
            excess = master_var - fpn
            if excess > 0:
                n_eff = temporal / excess
                lines.append(
                    f"Noise implies ≈ {n_eff:.0f} frames averaged"
                    f"{f' (header: {claimed})' if claimed else ''}."
                )
                if claimed and not 0.5 < n_eff / claimed < 2:
                    level = Level.WARN
                    lines.append(
                        "A master averages random noise down by √N; this one doesn't "
                        "match its frame count (different subs, or another rejection/"
                        "combination method)."
                    )
        out.append(Check(f"master-{fs.key}", title, level, lines[0], " ".join(lines[1:])))
    return out


def _bias_stability(data: ModeData, rn: ReadNoise, gain: Gain) -> list[Check]:
    bias = [(fs, st) for fs, st in data.of(FrameType.BIAS) if st.n_frames >= 3]
    if not bias:
        return []
    out = []
    levels = []
    for fs, st in bias:
        means = st.frame_mean.mean(axis=1)
        levels.append((fs, float(np.median(means))))
        drift = float(np.ptp(means))
        ref = rn.combined_adu.value if rn.combined_adu.ok else 1.0
        t = st.times[np.isfinite(st.times)]
        span = f" over {(t.max() - t.min()) / 60:.0f} min" if t.size > 1 else ""
        level = Level.WARN if drift > 0.5 * ref else Level.OK
        out.append(
            Check(
                f"bias-drift-{fs.key}",
                "Bias stability",
                level,
                f"Level varies {drift:.2f} ADU{span} ({drift / ref:.0%} of read noise)",
                f"{fs.label}: frame-mean drift. Stable bias is what makes a master bias "
                "(or bias-subtracted darks) valid across a session.",
            )
        )
    if len(levels) > 1:
        values = [v for _, v in levels]
        spread = max(values) - min(values)
        text = ", ".join(f"{fs.folder.name}: {v:.1f}" for fs, v in levels)
        level = (
            Level.WARN
            if spread > 2 * (rn.combined_adu.value if rn.combined_adu.ok else 1)
            else Level.OK
        )
        out.append(
            Check(
                "bias-sessions",
                "Bias between sessions",
                level,
                f"Mean levels differ by {spread:.1f} ADU across {len(levels)} sets",
                text,
            )
        )
    return out


def _temperatures(data: ModeData) -> list[Check]:
    out = []
    for fs, st in data.all(subs=True):
        temps = st.temps[np.isfinite(st.temps)]
        if temps.size < 2 or fs.kind is FrameType.FLAT:
            continue
        spread = float(np.ptp(temps))
        cooled = fs.frames[0].set_temperature is not None
        limit = 1.0 if cooled else 5.0
        if spread > limit:
            out.append(
                Check(
                    f"temp-{fs.key}",
                    "Temperature stability",
                    Level.WARN,
                    f"{fs.label}: sensor temperature spans {spread:.1f} °C",
                    "Dark current roughly doubles every 5–7 °C; frames meant to match "
                    "should be taken at one temperature.",
                )
            )
    lights = data.of(FrameType.LIGHT)
    darks = data.of(FrameType.DARK, subs=None)
    for fs, _ in lights:
        if fs.temperature is None or not darks:
            continue
        same = [
            d
            for d, _ in darks
            if d.exposure
            and fs.exposure
            and abs(d.exposure - fs.exposure) <= 0.05 * fs.exposure
            and d.temperature is not None
        ]
        if not same:
            continue
        best = min(same, key=lambda d: abs(d.temperature - fs.temperature))
        diff = best.temperature - fs.temperature
        cooled = fs.frames[0].set_temperature is not None
        limit = 1.5 if cooled else 4.0
        level = Level.WARN if abs(diff) > limit else Level.OK
        out.append(
            Check(
                f"temp-match-{fs.key}",
                "Dark/light temperature",
                level,
                f"{fs.label}: best dark is {diff:+.1f} °C away ({best.label})",
            )
        )
    return out


def _flats(data: ModeData, gain: Gain) -> list[Check]:
    out = []
    for sg in gain.per_set:
        issues, level = [], Level.OK
        if math.isfinite(sg.level_fraction):
            issues.append(f"level {sg.level_fraction:.0%} of full scale")
            if not 0.2 <= sg.level_fraction <= 0.8:
                level = Level.WARN
        if sg.saturated > 1e-4:
            issues.append(f"{sg.saturated:.2%} clipped")
            level = Level.PROBLEM if sg.saturated > 1e-3 else Level.WARN
        if sg.stability > 0.01:
            issues.append(f"frame-to-frame level varies {sg.stability:.1%}")
            level = max(level, Level.WARN, key=_rank)
        out.append(
            Check(
                f"flat-{sg.label}",
                "Flat quality",
                level,
                f"{sg.label}: " + ", ".join(issues),
                "Good flats sit at 30–60 % of full scale, unclipped, with a steady light source.",
            )
        )
    for fs, st in data.of(FrameType.FLAT):
        if st.n_frames < 3:
            continue
        rows = st.row_means[0]
        dev = rows - rows.mean(axis=0, keepdims=True)
        dev -= dev.mean(axis=1, keepdims=True)
        row_var = float(np.mean(dev**2))
        expected = (
            float(np.mean(st.pair_var[:, 0])) / st.col_means[0].shape[1] if st.n_pairs else math.nan
        )
        if math.isfinite(expected) and row_var > 4 * expected:
            out.append(
                Check(
                    f"flicker-{fs.key}",
                    "Flat banding",
                    Level.WARN,
                    f"{fs.label}: row brightness varies frame to frame "
                    f"({math.sqrt(row_var / expected):.1f}× shot noise)",
                    "Typical of a pulsed (PWM) light panel read out by a rolling shutter. "
                    "Longer flat exposures or a different panel brightness fix it.",
                )
            )
        exposure = fs.exposure or 0
        if fs.frames[0].format is Format.RAW and 0 < exposure < 1 / 100:
            out.append(
                Check(
                    f"shutter-{fs.key}",
                    "Short flat exposure",
                    Level.INFO,
                    f"{fs.label}: {format_exposure(exposure)}",
                    "Very short exposures with a mechanical shutter can expose the frame "
                    "unevenly; 1/50 s or longer is safer.",
                )
            )
    return out


def _rank(level: Level) -> int:
    return {Level.OK: 0, Level.INFO: 1, Level.WARN: 2, Level.PROBLEM: 3}[level]


def _dark_scaling(dark: DarkCurrent) -> list[Check]:
    out = []
    for fit in dark.slope_fits:
        t = fit["exposure"]
        s = fit["signal"]
        if len(t) < 2:
            continue
        intercept = fit["intercept"]
        rel = intercept / max(float(np.max(s)), 1e-9)
        level = Level.WARN if abs(rel) > 0.2 else Level.OK
        out.append(
            Check(
                "dark-scaling",
                "Dark scaling",
                level,
                f"Dark signal is linear in exposure; intercept {intercept:+.2f} ADU",
                "A large intercept means the bias is not a zero-length dark (the camera "
                "adds a fixed offset or glow per exposure); scale darks with care.",
            )
        )
    glow = dark.glow
    if glow is not None and len(glow.scaling) >= 2 and glow.visible:
        rates = [p / t for t, p in glow.scaling if t > 0]
        spread = (max(rates) - min(rates)) / max(np.mean(rates), 1e-12)
        level = Level.WARN if spread > 0.3 else Level.OK
        out.append(
            Check(
                "glow-scaling",
                "Amp glow scaling",
                level,
                f"Glow per second varies {spread:.0%} between exposures",
                "If glow doesn't scale with exposure, darks must match the light "
                "exposure exactly (no dark scaling/optimisation).",
            )
        )
    return out


def _calibration_coverage(data: ModeData) -> list[Check]:
    out = []
    darks = data.of(FrameType.DARK, subs=None)
    flats = data.of(FrameType.FLAT, subs=None)
    bias = data.of(FrameType.BIAS, subs=None)
    for fs, _ in data.of(FrameType.LIGHT):
        missing = []
        if not any(
            d.exposure and fs.exposure and abs(d.exposure - fs.exposure) <= 0.05 * fs.exposure
            for d, _ in darks
        ):
            missing.append(f"darks at {format_exposure(fs.exposure or 0)}")
        # DSLR flats needn't match the lights' ISO (and then live in another mode).
        if not any(f.filter == fs.filter for f, _ in flats) and data.mode.iso is None:
            missing.append(f"flats{' for ' + fs.filter if fs.filter else ''}")
        if not bias:
            missing.append("bias (or dark flats) for the flats")
        level = Level.WARN if missing else Level.OK
        summary = "complete" if not missing else "missing " + ", ".join(missing)
        out.append(
            Check(f"cover-{fs.key}", "Calibration coverage", level, f"{fs.label}: {summary}")
        )
    return out


def _black_level(data: ModeData) -> list[Check]:
    values = [st.black_meta for _, st in data.all(subs=True) if np.isfinite(st.black_meta).any()]
    if not values:
        return []
    allv = np.concatenate(values, axis=0)
    spread = float(np.nanmax(allv) - np.nanmin(allv))
    if spread < 2:
        return []
    return [
        Check(
            "black",
            "Black level varies",
            Level.INFO,
            f"Raw black level spans {spread:.0f} ADU between frames",
            "The camera measures the black level on masked pixels for every frame. DSA "
            "uses measured bias frames where they exist; otherwise each frame's own black.",
        )
    ]
