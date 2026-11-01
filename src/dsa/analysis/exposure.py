"""Optimal sub-exposure from the measured read noise and the sky brightness in your lights.

Stacking many short subs costs one read noise per sub; long subs risk satellites, guiding errors
and clipped stars. Robin Glover's criterion picks the shortest sub whose noise is within a
fraction x of a single infinitely long exposure (sky-limited):

    (1 + x)² = 1 + R² / (P·t)   ⇒   t = R² / (P·((1 + x)² − 1))   (≈ 10·R²/P for x = 5 %)

with R the read noise (e⁻) and P the sky flux (e⁻/s/px), measured from the lights' background
above offset and dark signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dsa.analysis.common import Level, Section, Status, Value
from dsa.analysis.dark import DarkCurrent
from dsa.analysis.data import ModeData
from dsa.analysis.gain import Gain
from dsa.analysis.levels import Offset
from dsa.analysis.noise import ReadNoise
from dsa.frames import FrameType

PENALTIES = (0.02, 0.05, 0.10)


def optimal_exposure(read_noise_e: float, flux: float, penalty: float) -> float:
    """Sub length (s) at which read noise adds ``penalty`` to the sky-limited noise."""
    if not (flux > 0 and read_noise_e > 0):
        return math.nan
    return read_noise_e**2 / (flux * ((1 + penalty) ** 2 - 1))


def noise_penalty(read_noise_e: float, flux: float, exposure: float, dark: float = 0.0) -> float:
    """Fractional extra noise vs a sky-limited exposure (``dark`` e⁻/s adds its own noise)."""
    if not (flux > 0 and exposure > 0):
        return math.nan
    return math.sqrt(1 + (read_noise_e**2 + dark * exposure) / (flux * exposure)) - 1


@dataclass
class SkyPoint:
    label: str
    filter: str | None
    exposure: float
    n: int
    sky_adu: float  # background above offset and dark signal
    flux: Value  # e⁻/s/px, mean over colour planes
    plane_flux: list[float]
    penalty: float  # read-noise penalty of the sub length you used
    penalty_with_dark: float
    swamp: float  # sky electrons per sub / read noise²
    optimal: dict[float, float]  # penalty -> seconds
    clipped: float  # fraction of pixels at the white level (star cores)
    optimal_assumed: dict[float, float] = field(default_factory=dict)  # with published RN


@dataclass
class SubExposure(Section):
    read_noise_e: float = math.nan
    read_noise_bound: bool = False
    assumed_read_noise: float = math.nan  # published value, used when only a bound exists
    dark_current: float = math.nan  # e⁻/s/px
    points: list[SkyPoint] = field(default_factory=list)


def sub_exposure(
    data: ModeData,
    gain: Gain,
    rn: ReadNoise,
    dark: DarkCurrent,
    offset: Offset,
    published_read_noise: float | None = None,
) -> SubExposure:
    """Sky flux from the lights and the sub length Glover's criterion recommends.

    ``published_read_noise`` (e⁻) is used, and labelled, when only an upper limit of the read
    noise could be measured.
    """
    result = SubExposure()
    result.read_noise_e = rn.combined_e.value if rn.combined_e.ok else math.nan
    result.read_noise_bound = rn.combined_e.bound == "upper"
    if result.read_noise_bound and published_read_noise:
        result.assumed_read_noise = published_read_noise
    known_dark = dark.current.ok and dark.unit == "e⁻/s" and not dark.current.bound
    dc = dark.current.value if known_dark else 0.0
    result.dark_current = dc
    lights = data.of(FrameType.LIGHT)
    if not (gain.gain.ok and rn.combined_e.ok):
        result.status = Status.UNAVAILABLE
        result.method = "Needs gain (flats) and read noise (bias) at this setting."
        if not lights:
            result.unlock.append("Light frames at this setting give the sky brightness.")
        return result
    if not lights:
        result.status = Status.LIMITED
        result.method = (
            "No light frames at this setting: enter your sky flux in the calculator, "
            "or add a few lights to measure it."
        )
        result.unlock.append("A few light frames at this setting (any target) measure your sky.")
        return result
    if not offset.known:
        result.status = Status.UNAVAILABLE
        result.method = "Needs an offset reference (bias) to measure the sky level."
        return result

    g = gain.gain.value
    darks = data.of(FrameType.DARK, subs=None)
    for fs, st in lights:
        t = float(fs.exposure or 0)
        if t <= 0:
            continue
        light_temp = fs.temperature if fs.temperature is not None else 0.0
        reference, how = _dark_reference(darks, offset, dark, dc, t, g, light_temp)
        if how:
            result.note(Level.INFO, f"{fs.label}: {how}")
        # The darkest frame's median is the sky least affected by twilight, moon or clouds.
        sky_planes = np.min(st.frame_median, axis=0) - reference
        sky = float(np.mean(sky_planes))
        plane_flux = [float(v * g / t) for v in sky_planes]
        flux = sky * g / t
        rel = gain.gain.error / g if math.isfinite(gain.gain.error) else math.nan
        r = result.read_noise_e
        assumed = result.assumed_read_noise
        result.points.append(
            SkyPoint(
                label=fs.label,
                filter=fs.filter,
                exposure=t,
                n=st.n_frames,
                sky_adu=sky,
                flux=Value(flux, abs(flux) * rel, "e⁻/s/px"),
                plane_flux=plane_flux,
                penalty=noise_penalty(r, flux, t),
                penalty_with_dark=noise_penalty(r, flux, t, dc),
                swamp=flux * t / r**2 if r > 0 else math.nan,
                optimal={x: optimal_exposure(r, flux, x) for x in PENALTIES},
                clipped=float(np.mean(st.frac_high)),
                optimal_assumed=(
                    {x: optimal_exposure(assumed, flux, x) for x in PENALTIES}
                    if math.isfinite(assumed)
                    else {}
                ),
            )
        )
    if result.read_noise_bound:
        tail = (
            f"; with the published read noise ({result.assumed_read_noise:.2f} e⁻) they are "
            "shorter, as shown."
            if math.isfinite(result.assumed_read_noise)
            else "."
        )
        result.note(
            Level.INFO,
            "Read noise is only an upper limit here (no bias frames), so the recommended sub "
            "lengths are upper limits too" + tail,
        )
    if not result.points:
        result.status = Status.UNAVAILABLE
        return result
    result.status = Status.MEASURED
    result.method = (
        "Sky flux = (darkest frame's median − offset − dark signal) × gain / exposure; "
        "Glover's criterion t = R² / (P·((1+x)² − 1))."
    )
    for p in result.points:
        if p.flux.value <= 0:
            result.note(
                Level.WARN,
                f"{p.label}: no sky signal above the offset — check the bias "
                "matches the lights (gain, offset, temperature).",
            )
        if p.optimal.get(0.05, 0) > 3600:
            result.note(
                Level.INFO,
                f"{p.label}: the sky adds very little signal ({p.flux.format()}), so subs stay "
                "read-noise limited for over an hour; sub length is set by tracking, satellites "
                "and dark current instead. Check these lights are representative.",
            )
        if p.clipped > 1e-4:
            result.note(
                Level.INFO,
                f"{p.label}: {p.clipped:.3%} of pixels are clipped (star "
                "cores); longer subs clip more stars.",
            )
    return result


def _dark_reference(
    darks: list,
    offset: Offset,
    dark: DarkCurrent,
    dc: float,
    t: float,
    gain: float,
    light_temp: float,
) -> tuple[np.ndarray, str]:
    """Level (per plane, ADU) a light of exposure ``t`` would have without any sky.

    A dark of the same exposure is the direct answer. Otherwise: the offset plus the dark
    current over the exposure (minus what the offset already holds when it is itself a dark).
    """
    matching = [
        (fs, st)
        for fs, st in darks
        if fs.exposure and abs(fs.exposure - t) <= 0.05 * t and fs.quantitative
    ]
    if matching:
        # Same exposure and the closest temperature (dark current depends strongly on it).
        _, st = min(
            matching, key=lambda p: (abs((p[0].temperature or 0) - light_temp), -p[1].n_frames)
        )
        # Medians, like the lights' sky level: dark noise is skewed, so mean ≠ median.
        return np.mean(st.frame_median, axis=0), ""
    if dark.clamped:
        return offset.level, ""
    extra = max(t - offset.exposure, 0.0)
    if dc > 0:
        return offset.level + dc * extra / gain, ""
    if extra > 0:
        return offset.level, "dark current unknown, so the sky estimate includes the dark signal"
    return offset.level, ""
