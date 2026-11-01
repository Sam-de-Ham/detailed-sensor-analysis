"""Synthetic sensor: frames generated from known parameters, the ground truth for tests.

The model follows the standard linear camera model (EMVA 1288 / Janesick):

    electrons = Poisson(photon signal * (1 + PRNU) * vignetting * dust) + Poisson(dark current * t)
    ADU       = offset + (electrons + read noise + fixed-pattern offsets) / gain, quantised

with the effects the analysis has to see through: per-pixel dark current spread, hot pixels,
amp glow, row/column fixed pattern, temporal row noise, random telegraph (RTS) pixels, full-well
and ADC clipping, a 12-bit ADC stored in a 16-bit container, and an optional Bayer mosaic with
different channel responses.

All parameters are in electrons and native ADC units; the "file" ADU differs from native ADU by
``container_scale`` (16 for 12-bit data written as 16-bit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from dsa.cfa import CfaLayout


@dataclass(frozen=True)
class Dust:
    x: float
    y: float
    radius: float  # outer radius of the shadow, pixels
    depth: float  # fractional light loss in the shadow ring (0.02 = 2 %)


@dataclass(frozen=True)
class SensorModel:
    width: int = 640
    height: int = 480
    cfa: str | None = None
    adc_bits: int = 12
    container_bits: int = 16  # data is stored left-shifted into this many bits
    gain: float = 1.0  # e-/ADU, native ADC units
    offset: float = 50.0  # native ADU
    read_noise: float = 1.5  # e- rms, white per-pixel temporal noise
    row_noise: float = 0.0  # e- rms, temporal noise common to a whole row
    row_fpn: float = 0.0  # e- rms, fixed offset per row
    column_fpn: float = 0.0  # e- rms, fixed offset per column
    dsnu: float = 0.5  # e- rms, fixed per-pixel offset (bias non-uniformity)
    full_well: float = 50_000.0  # e-
    nonlinearity: float = 0.0  # fractional compression at full well (S * (1 - k S / FW))
    dark_current: float = 0.01  # e-/s/px median at reference temperature
    dark_ref_temp: float = -10.0  # deg C
    dark_doubling: float = 6.0  # deg C per doubling
    dark_spread: float = 0.3  # log-normal sigma of per-pixel dark current
    hot_fraction: float = 1e-4
    hot_factor: float = 200.0  # hot pixels have this many times the median dark current
    amp_glow: float = 0.0  # e-/s excess at the glow centre
    amp_glow_scale: float = 80.0  # pixels, e-folding distance from the glow centre
    amp_glow_at: tuple[float, float] = (1.0, 0.5)  # (x, y) as fractions of width/height
    rts_fraction: float = 0.0
    rts_amplitude: float = 6.0  # e-, step between the two RTS states
    prnu: float = 0.01  # fractional rms
    vignetting: float = 0.3  # fractional light loss in the far corner
    optical_centre: tuple[float, float] = (0.5, 0.5)  # fractions of width/height
    dust: tuple[Dust, ...] = ()
    channel_response: dict[str, float] = field(
        default_factory=lambda: {"R": 0.55, "G1": 1.0, "G2": 1.0, "B": 0.7}
    )
    clamp_dark: bool = False  # camera removes the mean dark signal (Canon-style black clamp)
    smoothing: float = 0.0  # in-camera filter: weight of the 3x3 same-colour neighbour mean
    seed: int = 1234

    @property
    def container_scale(self) -> int:
        return 2 ** (self.container_bits - self.adc_bits)

    @property
    def file_gain(self) -> float:
        """e- per ADU as stored in the file."""
        return self.gain / self.container_scale

    @property
    def file_offset(self) -> float:
        return self.offset * self.container_scale

    @property
    def white_level(self) -> int:
        return (2**self.adc_bits - 1) * self.container_scale

    def dark_current_at(self, temperature: float) -> float:
        return self.dark_current * 2 ** ((temperature - self.dark_ref_temp) / self.dark_doubling)


class SyntheticSensor:
    """A sensor instance: the fixed patterns are drawn once, frames are drawn per call."""

    def __init__(self, model: SensorModel) -> None:
        self.model = model
        m = model
        rng = np.random.default_rng(m.seed)
        shape = (m.height, m.width)
        self.layout = CfaLayout(m.cfa)

        # Fixed offsets (e-): per-pixel DSNU, per-row and per-column patterns.
        self.fixed_offset = (
            rng.normal(0, m.dsnu, shape)
            + rng.normal(0, m.row_fpn, (m.height, 1))
            + rng.normal(0, m.column_fpn, (1, m.width))
        ).astype(np.float32)

        # Relative dark current per pixel (median 1), hot pixels, amp glow (e-/s, at ref temp).
        relative = rng.lognormal(0.0, m.dark_spread, shape) if m.dark_spread > 0 else np.ones(shape)
        relative /= np.median(relative)
        self.hot = rng.random(shape) < m.hot_fraction
        relative[self.hot] = m.hot_factor * rng.uniform(0.5, 2.0, int(self.hot.sum()))
        self.dark_relative = relative.astype(np.float32)
        yy, xx = np.mgrid[0 : m.height, 0 : m.width].astype(np.float32)
        gx, gy = m.amp_glow_at[0] * (m.width - 1), m.amp_glow_at[1] * (m.height - 1)
        distance = np.hypot(xx - gx, yy - gy)
        self.glow = (m.amp_glow * np.exp(-distance / m.amp_glow_scale)).astype(np.float32)

        # Photo response: PRNU, vignetting, dust, channel response.
        self.prnu = rng.normal(0, m.prnu, shape).astype(np.float32)
        cx, cy = m.optical_centre[0] * (m.width - 1), m.optical_centre[1] * (m.height - 1)
        r = np.hypot(xx - cx, yy - cy)
        r_corner = max(
            np.hypot(c[0] - cx, c[1] - cy)
            for c in ((0, 0), (m.width - 1, 0), (0, m.height - 1), (m.width - 1, m.height - 1))
        )
        self.vignetting = (1 - m.vignetting * (r / r_corner) ** 2).astype(np.float32)
        self.dust = np.ones(shape, dtype=np.float32)
        for d in m.dust:
            rd = np.hypot(xx - d.x, yy - d.y)
            # A donut: full depth in the ring, half depth in the centre (central obstruction).
            ring = (rd <= d.radius) & (rd >= 0.5 * d.radius)
            core = rd < 0.5 * d.radius
            self.dust[ring] *= 1 - d.depth
            self.dust[core] *= 1 - d.depth / 2
        response = np.ones(shape, dtype=np.float32)
        if m.cfa:
            for name, plane in self.layout.planes(response):
                plane[...] = m.channel_response.get(name, 1.0)
        self.response = response
        self.flat_field = (1 + self.prnu) * self.vignetting * self.dust * self.response

        self.rts = rng.random(shape) < m.rts_fraction
        self._rng = np.random.default_rng(m.seed + 1)

    # ----------------------------------------------------------------------------- frames --

    def dark_rate(self, temperature: float) -> np.ndarray:
        """True per-pixel dark current (e-/s) at ``temperature``."""
        scale = self.model.dark_current_at(temperature) / self.model.dark_current
        return (self.dark_relative * self.model.dark_current + self.glow) * scale

    def frame(
        self,
        exposure: float = 0.0,
        temperature: float | None = None,
        flat_level: float = 0.0,
        sky_rate: float = 0.0,
        stars: int = 0,
    ) -> np.ndarray:
        """One frame as stored in the file (uint16).

        ``flat_level`` is the mean photo-electron signal of the brightest (G) plane at the optical
        centre; ``sky_rate`` is a uniform sky flux in e-/s/px; ``stars`` adds point sources.
        """
        m, rng = self.model, self._rng
        temperature = m.dark_ref_temp if temperature is None else temperature
        shape = (m.height, m.width)
        signal = np.zeros(shape, dtype=np.float64)
        if flat_level > 0:
            signal += flat_level * self.flat_field
        if sky_rate > 0 or stars:
            sky = np.full(shape, sky_rate * exposure, dtype=np.float64)
            if stars:
                sky += self._stars(stars, exposure)
            signal += sky * self.flat_field
        electrons = rng.poisson(signal).astype(np.float64) if signal.any() else signal
        if exposure > 0:
            dark = self.dark_rate(temperature) * exposure
            electrons += rng.poisson(dark)
            if m.clamp_dark:
                electrons -= float(np.median(dark))  # the level goes, its shot noise stays
        electrons = np.minimum(electrons, m.full_well)
        if m.nonlinearity:
            electrons = electrons * (1 - m.nonlinearity * electrons / m.full_well)
        noise = rng.normal(0, m.read_noise, shape)
        if m.row_noise:
            noise += rng.normal(0, m.row_noise, (m.height, 1))
        if m.rts_fraction:
            state = rng.random(shape) < 0.5
            noise += np.where(self.rts & state, m.rts_amplitude, 0.0)
        adu = m.offset + (electrons + noise + self.fixed_offset) / m.gain
        if m.smoothing:
            adu = self._smooth(adu)
        adu = np.clip(np.rint(adu), 0, 2**m.adc_bits - 1)
        return (adu * m.container_scale).astype(np.uint16)

    def _smooth(self, image: np.ndarray) -> np.ndarray:
        """Mix each pixel with the mean of its same-colour neighbours (noise reduction)."""
        from scipy.ndimage import uniform_filter

        out = image.copy()
        w = self.model.smoothing
        for name, plane in self.layout.planes(image):
            self.layout.plane(out, name)[...] = (1 - w) * plane + w * uniform_filter(plane, 3)
        return out

    def _stars(self, count: int, exposure: float) -> np.ndarray:
        m, rng = self.model, self._rng
        image = np.zeros((m.height, m.width))
        xs = rng.uniform(5, m.width - 5, count).astype(int)
        ys = rng.uniform(5, m.height - 5, count).astype(int)
        flux = rng.pareto(1.5, count) * 200 * exposure
        image[ys, xs] = flux
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(image, 1.5) * 2 * math.pi * 1.5**2


# --------------------------------------------------------------------------- FITS output --


def write_fits(path: Path, data: np.ndarray, header: dict[str, object]) -> None:
    """Write a uint16 frame the way NINA does (BZERO 32768, top-down rows)."""
    from astropy.io import fits

    hdu = fits.PrimaryHDU(data)
    for key, value in header.items():
        hdu.header[key] = value
    hdu.header["ROWORDER"] = "TOP-DOWN"  # astropy stores uint16 as int16 + BZERO 32768
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


def write_session(
    sensor: SyntheticSensor,
    root: Path,
    *,
    camera: str = "SIMCAM",
    gain_setting: float = 100,
    bias: int = 0,
    darks: dict[float, int] | None = None,
    dark_temps: tuple[float, ...] = (),
    flats: dict[str, tuple[float, int]] | None = None,
    flat_series: tuple[tuple[float, float], ...] = (),
    lights: dict[str, tuple[float, int, float]] | None = None,
    temperature: float | None = None,
) -> None:
    """Write a NINA-style capture session of synthetic frames under ``root``.

    ``darks`` maps exposure -> count (at ``temperature`` and every ``dark_temps`` value);
    ``flats`` maps filter -> (flat level e-, count); ``lights`` maps filter ->
    (exposure, count, sky e-/s).
    """
    m = sensor.model
    temperature = m.dark_ref_temp if temperature is None else temperature
    clock = datetime(2026, 1, 1, 22, 0, 0)

    def header(kind: str, exposure: float, temp: float, filt: str | None = None) -> dict:
        nonlocal clock
        clock += timedelta(seconds=exposure + 2)
        h: dict[str, object] = {
            "IMAGETYP": kind,
            "EXPTIME": exposure,
            "EXPOSURE": exposure,
            "INSTRUME": camera,
            "GAIN": gain_setting,
            "OFFSET": int(m.offset),
            "XBINNING": 1,
            "YBINNING": 1,
            "CCD-TEMP": temp,
            "SET-TEMP": temp,
            "XPIXSZ": 2.9,
            "YPIXSZ": 2.9,
            "DATE-LOC": clock.isoformat(),
            "DATE-OBS": clock.isoformat(),
        }
        if m.cfa:
            h["BAYERPAT"] = m.cfa
        if filt:
            h["FILTER"] = filt
        return h

    for i in range(bias):
        write_fits(
            root / "BIAS" / f"bias_{i:04d}.fits",
            sensor.frame(0.0, temperature),
            header("BIAS", 0.0, temperature),
        )
    for temp in (temperature, *dark_temps):
        for exposure, count in (darks or {}).items():
            for i in range(count):
                name = f"dark_{exposure:g}s_{temp:+g}C_{i:04d}.fits"
                write_fits(
                    root / "DARK" / name,
                    sensor.frame(exposure, temp),
                    header("DARK", exposure, temp),
                )
    for filt, (level, count) in (flats or {}).items():
        for i in range(count):
            write_fits(
                root / "FLAT" / f"flat_{filt}_{i:04d}.fits",
                sensor.frame(0.5, temperature, flat_level=level),
                header("FLAT", 0.5, temperature, filt),
            )
    # A photon-transfer series: (exposure s, mean e-) pairs, two frames each, filter "PTC".
    for k, (exposure, level) in enumerate(flat_series):
        for i in range(2):
            h = header("FLAT", exposure, temperature, "PTC")
            write_fits(
                root / f"PTC/{k:02d}" / f"flat_{k:02d}_{i}.fits",
                sensor.frame(exposure, temperature, flat_level=level),
                h,
            )
    for filt, (exposure, count, sky) in (lights or {}).items():
        for i in range(count):
            write_fits(
                root / "LIGHT" / f"light_{filt}_{i:04d}.fits",
                sensor.frame(exposure, temperature, sky_rate=sky, stars=40),
                header("LIGHT", exposure, temperature, filt),
            )


def colour_model(**changes: object) -> SensorModel:
    """A DSLR-like Bayer sensor: 14-bit, no container scaling, higher black level."""
    base = SensorModel(
        cfa="RGGB",
        adc_bits=14,
        container_bits=14,
        gain=0.8,
        offset=1024,
        read_noise=3.0,
        full_well=13_000,
        dark_current=0.05,
        dark_ref_temp=20.0,
    )
    return replace(base, **changes)
