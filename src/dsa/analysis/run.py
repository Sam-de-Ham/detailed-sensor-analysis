"""Run every analysis the data supports for one camera, mode by mode."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event

from dsa.analysis.common import Level, Note, Status, Value
from dsa.analysis.dark import DarkCurrent, dark_current
from dsa.analysis.data import ModeData
from dsa.analysis.exposure import SubExposure, sub_exposure
from dsa.analysis.flat import FlatField, flat_field
from dsa.analysis.gain import DynamicRange, Gain, dynamic_range, measure_gain, scale_gain
from dsa.analysis.levels import DataFormat, Offset, data_format, find_offset
from dsa.analysis.noise import PatternNoise, ReadNoise, pattern_noise, read_noise
from dsa.analysis.pixels import Defects, find_defects
from dsa.analysis.quality import Check, run_checks
from dsa.analysis.specs import (
    SensorSpec,
    SpecReport,
    compare,
    find_spec,
    published_read_noise,
)
from dsa.catalog import Catalog, FrameSet, Mode
from dsa.measure import Cancelled, SetStats, StatsCache, pixel_series

Progress = Callable[[float, str], None]


@dataclass
class ModeReport:
    mode: Mode
    sets: list[FrameSet]
    format: DataFormat
    offset: Offset
    read_noise: ReadNoise
    pattern: PatternNoise
    gain: Gain
    dynamic_range: DynamicRange
    dark: DarkCurrent
    flat: FlatField
    defects: Defects
    exposure: SubExposure
    checks: list[Check]
    specs: SpecReport
    failures: list[Note] = field(default_factory=list)  # sets that could not be read

    @property
    def setting_value(self) -> float | None:
        return float(self.mode.iso) if self.mode.iso is not None else self.mode.gain


@dataclass
class CameraReport:
    camera: str
    spec: SensorSpec | None
    modes: list[ModeReport]
    pixel_size: float | None
    focal_ratio: float | None


@dataclass
class Settings:
    max_frames: int = 50  # frames per set used for statistics
    rts: bool = True  # second pass over bias/darks for telegraph-noise pixels


def _set_metadata(sets: list[FrameSet], attr: str) -> float | None:
    values = [getattr(f, attr) for fs in sets for f in fs.frames if getattr(f, attr)]
    return float(values[0]) if values else None


def measure_mode(
    sets: list[FrameSet],
    cache: StatsCache,
    settings: Settings,
    progress: Progress | None = None,
    cancel: Event | None = None,
) -> tuple[dict[str, SetStats], list[Note]]:
    stats: dict[str, SetStats] = {}
    failures: list[Note] = []
    total = sum(min(fs.count, settings.max_frames) for fs in sets) or 1
    done = 0
    for fs in sets:
        if not fs.quantitative:
            continue
        n = min(fs.count, settings.max_frames)

        def sub_progress(fraction: float, text: str, base: int = done, n: int = n) -> None:
            if progress is not None:
                progress((base + fraction * n) / total, text)

        try:
            stats[fs.key] = cache.measure(fs, settings.max_frames, sub_progress, cancel)
        except Cancelled:
            raise
        except Exception as err:  # one unreadable set must not sink the whole analysis
            failures.append(Note(Level.PROBLEM, f"{fs.label} ({fs.folder}): {err}"))
        done += n
    return stats, failures


def analyse_mode(
    data: ModeData,
    spec: SensorSpec | None,
    pixel_size: float | None,
    focal_ratio: float | None,
    settings: Settings,
    cancel: Event | None = None,
) -> ModeReport:
    fmt = data_format(data)
    offset = find_offset(data)
    rn = read_noise(data, fmt)
    gain = measure_gain(data, fmt, rn)
    return _finish(data, fmt, offset, rn, gain, spec, pixel_size, focal_ratio, settings, cancel)


def _finish(
    data: ModeData,
    fmt: DataFormat,
    offset: Offset,
    rn: ReadNoise,
    gain: Gain,
    spec: SensorSpec | None,
    pixel_size: float | None,
    focal_ratio: float | None,
    settings: Settings,
    cancel: Event | None,
) -> ModeReport:
    rn.to_electrons(gain.gain)
    pattern = pattern_noise(data, rn)
    dark = dark_current(data, gain, rn, offset, fmt)
    if not rn.combined_e.ok and dark.read_noise_bound.ok:
        rn.combined_e = dark.read_noise_bound
    dr = dynamic_range(gain, rn, fmt, offset)
    flat = flat_field(data, gain, pixel_size, focal_ratio)
    reader = (lambda frames, ys, xs: pixel_series(frames, ys, xs, cancel)) if settings.rts else None
    defects = find_defects(data, dark, flat, rn, gain, reader)
    setting = float(data.mode.iso) if data.mode.iso is not None else data.mode.gain
    assumed = published_read_noise(spec, setting, data.mode.readout)
    exposure = sub_exposure(data, gain, rn, dark, offset, assumed)
    checks = run_checks(data, fmt, offset, rn, gain, dark)
    specs = compare(
        spec,
        setting,
        data.mode.readout,
        gain_native=gain.native,
        read_noise_e=rn.combined_e,
        saturation_e=dr.saturation_e,
        dark_current=dark.current,
        dark_temperature=dark.reference_temperature,
        dark_unit=dark.unit,
    )
    return ModeReport(
        data.mode,
        data.sets,
        fmt,
        offset,
        rn,
        pattern,
        gain,
        dr,
        dark,
        flat,
        defects,
        exposure,
        checks,
        specs,
    )


def analyse_camera(
    catalog: Catalog,
    camera: str,
    cache: StatsCache,
    settings: Settings | None = None,
    progress: Progress | None = None,
    cancel: Event | None = None,
) -> CameraReport:
    """Measure and analyse every mode of ``camera``.

    Modes without flats borrow the gain of a measured mode of the same camera and readout,
    scaled by the ratio of settings (ISO or gain), and say so.
    """
    settings = settings or Settings()
    spec = find_spec(camera)
    modes = catalog.modes(camera)
    all_sets = [fs for m in modes for fs in catalog.sets_for(m)]
    pixel_size = _set_metadata(all_sets, "pixel_size") or (spec.pixel_size if spec else None)
    focal_ratio = _set_metadata(all_sets, "focal_ratio")
    total = sum(min(fs.count, settings.max_frames) for fs in all_sets) or 1
    done = 0
    prepared = []
    for mode in modes:
        sets = catalog.sets_for(mode)
        weight = sum(min(fs.count, settings.max_frames) for fs in sets)

        def mode_progress(
            fraction: float, text: str, base: int = done, w: int = weight, mode: Mode = mode
        ) -> None:
            if progress is not None:
                progress(0.9 * (base + fraction * w) / total, f"{mode.setting}: {text}")

        stats, failures = measure_mode(sets, cache, settings, mode_progress, cancel)
        done += weight
        data = ModeData(mode, sets, stats)
        if not stats:
            continue
        fmt = data_format(data)
        offset = find_offset(data)
        rn = read_noise(data, fmt)
        gain = measure_gain(data, fmt, rn)
        prepared.append((data, fmt, offset, rn, gain, failures))

    _borrow_gains([(p[0].mode, p[4]) for p in prepared])
    reports = []
    for i, (data, fmt, offset, rn, gain, failures) in enumerate(prepared):
        if progress is not None:
            progress(0.9 + 0.1 * i / max(len(prepared), 1), f"{data.mode.setting}: analysing")
        report = _finish(
            data, fmt, offset, rn, gain, spec, pixel_size, focal_ratio, settings, cancel
        )
        report.failures = failures
        reports.append(report)
    if progress is not None:
        progress(1.0, "Done")
    return CameraReport(camera, spec, reports, pixel_size, focal_ratio)


def _setting(mode: Mode) -> float | None:
    return float(mode.iso) if mode.iso is not None else mode.gain


def _borrow_gains(pairs: list[tuple[Mode, Gain]]) -> None:
    measured = [(m, g) for m, g in pairs if g.gain.ok and not g.estimated_from]
    for mode, gain in pairs:
        if gain.gain.ok or _setting(mode) is None:
            continue
        candidates = [
            (m, g)
            for m, g in measured
            if m.readout == mode.readout and m.binning == mode.binning and _setting(m)
        ]
        if not candidates:
            continue
        ref_mode, ref_gain = min(
            candidates, key=lambda p: abs(math.log(_setting(p[0]) / _setting(mode)))
        )
        scale_gain(gain, ref_gain, _setting(ref_mode), _setting(mode), ref_mode.setting)
        gain.unlock = [u for u in gain.unlock if "flats" not in u.lower()] + [
            "Flats at this setting would measure the gain directly."
        ]


def headline(report: ModeReport) -> list[tuple[str, Value, Status]]:
    """The few numbers that summarise a mode, for overviews and the CLI."""
    rn, g, dr, dk, ff = (
        report.read_noise,
        report.gain,
        report.dynamic_range,
        report.dark,
        report.flat,
    )
    rows = [
        ("Gain", g.gain, g.status),
        ("Read noise", rn.combined_e if rn.combined_e.ok else rn.combined_adu, rn.status),
        ("Saturation", dr.saturation_e, dr.status),
        ("Dynamic range", dr.stops, dr.status),
        ("Dark current", dk.current, dk.status),
        ("PRNU", ff.prnu, ff.status),
    ]
    return rows
