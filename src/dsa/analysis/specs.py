"""Published sensor data and the comparison of measurements against it.

Values come from docs/research.md (manufacturer curves, photonstophotos.net by Bill Claff, and
independent measurements), with their sources. Gains are in **native ADC units** (e⁻ per ADC
step); measurements are converted with the detected container scale before comparing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from dsa.analysis.common import Value


@dataclass(frozen=True)
class Curve:
    """A published curve over the camera's setting (ISO, or the driver's gain value)."""

    setting: tuple[float, ...]
    gain: tuple[float, ...]  # e⁻/ADU, native
    read_noise: tuple[float, ...]  # e⁻
    full_well: tuple[float, ...] = ()  # e⁻ (often the ADC-limited saturation)
    source: str = ""
    url: str = ""


@dataclass(frozen=True)
class SensorSpec:
    name: str
    pattern: str  # regex on the camera name
    sensor: str
    pixel_size: float  # µm
    adc_bits: int
    setting_name: str  # "ISO" or "Gain"
    curves: dict[str, Curve]  # by readout mode ("" = the only one)
    dark_temps: tuple[float, ...] = ()
    dark_current: tuple[float, ...] = ()  # e⁻/s/px
    dark_source: str = ""
    notes: tuple[str, ...] = ()

    def curve_for(self, readout: str | None) -> Curve | None:
        if len(self.curves) == 1:
            return next(iter(self.curves.values()))
        key = _readout_key(readout)
        return self.curves.get(key)


def _readout_key(readout: str | None) -> str:
    text = (readout or "").lower()
    if "hdr" in text:
        return "HDR"
    if "high" in text or "hcg" in text:
        return "HCG"
    if "low" in text or "lcg" in text:
        return "LCG"
    return ""


_TOUPTEK_SETTINGS = (100, 177, 316, 562, 1000, 1778, 3162, 5623, 10000)
_TOUPTEK_URL = "https://touptekastro.com/en-eu/products/atr585m"

SPECS: tuple[SensorSpec, ...] = (
    SensorSpec(
        name="ToupTek ATR585M",
        pattern=r"ATR585|585M",
        sensor="Sony IMX585 (STARVIS 2, BSI)",
        pixel_size=2.9,
        adc_bits=12,
        setting_name="Gain",
        curves={
            "HCG": Curve(
                _TOUPTEK_SETTINGS,
                (1.0, 0.56, 0.31, 0.18, 0.10, 0.06, 0.03, 0.02, 0.01),
                (0.79, 0.72, 0.59, 0.54, 0.52, 0.52, 0.50, 0.52, 0.54),
                (4100, 2300, 1300, 740, 420, 230, 135, 75, 43),
                "ToupTek ATR585M HCG chart (−10 °C), read by eye",
                _TOUPTEK_URL,
            ),
            "LCG": Curve(
                _TOUPTEK_SETTINGS,
                (9.5, 5.45, 3.0, 1.72, 0.95, 0.55, 0.30, 0.17, 0.10),
                (2.74, 3.04, 2.39, 2.40, 2.21, 2.18, 1.91, 1.92, 2.02),
                (38900, 22000, 12000, 7000, 4000, 2200, 1250, 700, 400),
                "ToupTek ATR585M LCG chart (−10 °C), read by eye",
                _TOUPTEK_URL,
            ),
            "HDR": Curve(
                _TOUPTEK_SETTINGS,
                (0.44, 0.25, 0.14, 0.08, 0.04, 0.02, 0.01, 0.01, 0.005),
                (0.64, 0.55, 0.55, 0.54, 0.51, 0.51, 0.53, 0.51, 0.55),
                (28000, 16500, 9000, 5100, 2800, 1600, 900, 500, 300),
                "ToupTek ATR585M HDR chart (16-bit ADU), read by eye",
                _TOUPTEK_URL,
            ),
        },
        dark_temps=(-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30),
        dark_current=(
            0.000386,
            0.000738,
            0.00126,
            0.00201,
            0.00435,
            0.00845,
            0.0162,
            0.0330,
            0.0656,
            0.1245,
            0.2495,
            0.512,
        ),
        dark_source="ZWO ASI585MM Pro manual (same IMX585 sensor)",
        notes=(
            "ToupTek gain is a percentage: 100 = 1× (0 dB).",
            "Vendor 'full well' at low gain is the ADC-limited saturation capacity.",
        ),
    ),
    SensorSpec(
        name="Canon EOS 600D",
        pattern=r"600D|T3i|Kiss X5",
        sensor="Canon 18 MP APS-C CMOS",
        pixel_size=4.3,
        adc_bits=14,
        setting_name="ISO",
        curves={
            "": Curve(
                (100, 200, 400, 800, 1600, 3200, 6400),
                (1.88, 0.93, 0.47, 0.23, 0.117, 0.058, 0.029),
                (14.72, 7.84, 4.96, 3.63, 3.07, 2.81, 2.64),
                (26796, 13300, 6650, 3300, 1650, 830, 410),
                "photonstophotos.net (Bill Claff)",
                "https://www.photonstophotos.net/Charts/RN_e.htm",
            )
        },
    ),
    SensorSpec(
        name="Canon EOS 500D",
        pattern=r"500D|T1i|Kiss X3",
        sensor="Canon 15 MP APS-C CMOS",
        pixel_size=4.7,
        adc_bits=14,
        setting_name="ISO",
        curves={
            "": Curve(
                (100, 200, 400, 800, 1600),
                (1.36, 0.68, 0.34, 0.17, 0.085),
                (15.8, 7.91, 4.74, 3.38, 2.74),
                (22282, 11141, 5571, 2785, 1393),
                "Subaru Telescope all-sky camera page (O. Guyon); gains likely "
                "extrapolated 2× per stop",
                "https://subarutelescope.org/staff/guyon/09allskysurvey.web/"
                "10detector.web/content.html",
            )
        },
        dark_temps=(3.0,),
        dark_current=(0.0044,),
        dark_source="Subaru page, upper limit 'to be confirmed'",
        notes=("Not in the photonstophotos database; only one independent source exists.",),
    ),
    SensorSpec(
        name="Sony A6400",
        pattern=r"ILCE-6400|A6400|α6400",
        sensor="Sony 24 MP APS-C CMOS (dual conversion gain)",
        pixel_size=3.92,
        adc_bits=14,
        setting_name="ISO",
        curves={
            "": Curve(
                (100, 125, 160, 200, 250, 320, 400, 500, 800, 1600, 3200, 6400, 12800),
                (2.30, 1.82, 1.44, 1.15, 0.91, 0.72, 0.57, 0.46, 0.29, 0.144, 0.072, 0.036, 0.018),
                (3.25, 3.05, 2.93, 2.91, 2.77, 2.75, 1.17, 1.12, 1.09, 1.05, 1.03, 0.78, 0.75),
                (36443, 28900, 22800, 18200, 14400, 11400, 9000, 7300, 4600, 2280, 1140, 570, 285),
                "photonstophotos.net (Bill Claff)",
                "https://www.photonstophotos.net/Charts/RN_e.htm",
            )
        },
        notes=(
            "Conversion gain switches between ISO 320 and 400.",
            "Compressed raw (cRAW) is lossy; bulb and continuous modes are 12-bit.",
        ),
    ),
)


def find_spec(camera: str) -> SensorSpec | None:
    return next((s for s in SPECS if re.search(s.pattern, camera, re.IGNORECASE)), None)


def interpolate(xs: tuple[float, ...], ys: tuple[float, ...], x: float) -> tuple[float, bool]:
    """Log-log interpolation; returns (value, extrapolated)."""
    if not xs or not ys or x <= 0:
        return math.nan, False
    lx, ly = np.log(np.array(xs, float)), np.log(np.array(ys, float))
    extrapolated = not xs[0] <= x <= xs[-1]
    if extrapolated:
        i = (0, 1) if x < xs[0] else (-2, -1)
        slope = (ly[i[1]] - ly[i[0]]) / (lx[i[1]] - lx[i[0]])
        return float(math.exp(ly[i[0]] + slope * (math.log(x) - lx[i[0]]))), True
    return float(math.exp(np.interp(math.log(x), lx, ly))), False


def published_read_noise(
    spec: SensorSpec | None, setting: float | None, readout: str | None
) -> float | None:
    """Published read noise (e⁻) at a setting, if the camera and setting are covered."""
    curve = spec.curve_for(readout) if spec is not None else None
    if curve is None or setting is None:
        return None
    value, extrapolated = interpolate(curve.setting, curve.read_noise, setting)
    return None if extrapolated or not math.isfinite(value) else value


def dark_at(spec: SensorSpec, temperature: float) -> tuple[float, bool]:
    """Published dark current at ``temperature`` (log-linear in T)."""
    if not spec.dark_temps or not math.isfinite(temperature):
        return math.nan, False
    t = np.array(spec.dark_temps, float)
    d = np.log(np.array(spec.dark_current, float))
    if t.size == 1:
        return float(spec.dark_current[0]), abs(temperature - t[0]) > 3
    extrapolated = not t[0] <= temperature <= t[-1]
    return float(math.exp(np.interp(temperature, t, d))), extrapolated


@dataclass
class Comparison:
    quantity: str
    measured: Value
    published: float
    unit: str
    source: str
    ratio: float
    agrees: bool | None  # None: cannot judge (bound, extrapolated)
    tolerance: float
    note: str = ""


@dataclass
class SpecReport:
    spec: SensorSpec | None
    setting: float | None
    readout: str = ""
    rows: list[Comparison] = field(default_factory=list)


def compare(
    spec: SensorSpec | None,
    setting: float | None,
    readout: str | None,
    *,
    gain_native: Value,
    read_noise_e: Value,
    saturation_e: Value,
    dark_current: Value,
    dark_temperature: float,
    dark_unit: str,
) -> SpecReport:
    report = SpecReport(spec, setting, _readout_key(readout))
    if spec is None or setting is None:
        return report
    curve = spec.curve_for(readout)
    if curve is not None:
        for quantity, measured, ys, unit, tol in (
            ("Gain", gain_native, curve.gain, "e⁻/ADU (native)", 0.10),
            ("Read noise", read_noise_e, curve.read_noise, "e⁻", 0.25),
            ("Saturation", saturation_e, curve.full_well, "e⁻", 0.20),
        ):
            if not measured.ok or not ys:
                continue
            published, extra = interpolate(curve.setting, ys, setting)
            report.rows.append(_row(quantity, measured, published, unit, curve.source, tol, extra))
    if dark_current.ok and dark_unit == "e⁻/s" and spec.dark_temps:
        published, extra = dark_at(spec, dark_temperature)
        if extra:  # outside the published temperature range: a ratio would mean nothing
            return report
        row = _row("Dark current", dark_current, published, "e⁻/s/px", spec.dark_source, 1.0, extra)
        row.note = (row.note + " " if row.note else "") + (
            f"Published value at {dark_temperature:.0f} °C (interpolated in temperature)."
        )
        report.rows.append(row)
    return report


def _row(
    quantity: str,
    measured: Value,
    published: float,
    unit: str,
    source: str,
    tolerance: float,
    extrapolated: bool,
) -> Comparison:
    ratio = measured.value / published if published else math.nan
    agrees: bool | None
    if measured.bound == "upper":
        agrees = None if ratio >= 1 - tolerance else False
    elif extrapolated:
        agrees = None
    else:
        agrees = abs(math.log(ratio)) <= math.log(1 + tolerance) if math.isfinite(ratio) else None
    note = "Published curve extrapolated to this setting." if extrapolated else ""
    if measured.bound == "upper":
        note = "Measured value is an upper limit."
    return Comparison(quantity, measured, published, unit, source, ratio, agrees, tolerance, note)
