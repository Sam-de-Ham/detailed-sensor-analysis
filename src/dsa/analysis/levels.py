"""The data's number format and the zero-signal (offset) level everything else is measured from.

*Container scaling*: a 12-bit ADC stored in 16-bit files (ToupTek, ZWO, QHY via NINA) only
uses every 16th code. Quantisation noise and "native" gain are in ADC units, so the scale has to
be known. It is detected from the code histogram rather than trusted from metadata.

*Offset*: bias (or dark-flat) frames give it directly. Without them, DSLR raws carry a black
level in their metadata; as a last resort the shortest dark is used, which includes some dark
signal and is flagged as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dsa.analysis.common import Level, Note
from dsa.analysis.data import ModeData
from dsa.frames import FrameType
from dsa.measure import SetStats
from dsa.stats import central_roi, clipped_stats


@dataclass
class DataFormat:
    integer: bool
    container_scale: int  # 16 for 12-bit data in 16-bit files, 1 when every code is used
    adc_bits: float  # effective ADC bits (log2 of usable codes)
    white_level: float  # highest code (file ADU)
    lossy: bool  # lossy raw compression (Sony cRAW)
    lossy_note: str = ""
    notes: list[Note] = field(default_factory=list)


def container_scale(hist: np.ndarray, threshold: float = 0.99) -> int:
    """Largest power-of-two step that (almost) all codes are multiples of.

    The threshold tolerates a few pixels the camera rewrote (ToupTek's defect correction leaves
    ~0.02 % of values off the 16-step grid).
    """
    total = hist.sum()
    if total == 0:
        return 1
    for k in range(8, 0, -1):
        step = 2**k
        if hist[::step].sum() >= threshold * total:
            return step
    return 1


def code_step(hist: np.ndarray, centre: float, width: float) -> float:
    """Median spacing of occupied codes within ``centre ± width`` (the local quantisation step)."""
    lo = max(0, int(centre - width))
    hi = min(hist.size - 1, int(centre + width))
    window = hist[lo : hi + 1]
    occupied = np.flatnonzero(window > max(1, window.max() * 1e-4))
    if occupied.size < 3:
        return math.nan
    return float(np.median(np.diff(occupied)))


def data_format(data: ModeData) -> DataFormat:
    sub_stats = [st for _, st in data.all(subs=True)]
    integer_hists = [st.code_hist for st in sub_stats if st.code_hist is not None]
    first = sub_stats[0] if sub_stats else next(st for _, st in data.all())
    white = max((st.white_level for st in sub_stats), default=first.white_level)
    if not integer_hists:
        return DataFormat(
            False,
            1,
            math.nan,
            white,
            False,
            notes=[
                Note(
                    Level.INFO,
                    "Only floating-point masters: the raw number format cannot be checked.",
                )
            ],
        )
    hist = np.sum([h.sum(axis=0) for h in integer_hists], axis=0)
    lossy = any(fs.frames[0].compressed for fs, _ in data.all(subs=True))
    # Lossy raw decodes through a tone curve whose code spacing grows with signal; that is
    # compression, not a container scale.
    scale = 1 if lossy else container_scale(hist)
    usable = (white + 1) / scale
    bits = math.log2(usable) if usable > 1 else math.nan
    fmt = DataFormat(True, scale, bits, white, lossy)
    if scale > 1:
        fmt.notes.append(
            Note(
                Level.INFO,
                f"Values are multiples of {scale}: a {round(bits)}-bit ADC stored as 16-bit. "
                f"1 native ADU = {scale} file ADU; quantisation noise is {scale}/√12 = "
                f"{scale / math.sqrt(12):.1f} file ADU rms.",
            )
        )
    if lossy:
        fmt.lossy_note = (
            "Lossy compressed raw (Sony cRAW): a tone curve and block delta coding add "
            "signal-dependent quantisation. Variance-based gain and noise are biased; DSA corrects "
            "for the measured code spacing, but uncompressed raw is needed for exact results."
        )
        fmt.notes.append(Note(Level.WARN, fmt.lossy_note))
    return fmt


def quantisation_variance(stats: SetStats, plane: int, level: float, sigma: float) -> float:
    """Variance added by quantisation at ``level`` (file ADU²), from the local code spacing."""
    if stats.code_hist is None:
        return 0.0
    step = code_step(stats.code_hist[plane], level, max(4 * sigma, 8))
    return step**2 / 12 if math.isfinite(step) else 0.0


# ------------------------------------------------------------------------------ offset --


@dataclass
class Offset:
    """Zero-signal level per colour plane, file ADU."""

    source: str  # "bias", "master bias", "dark flat", "raw black level", "short dark", "none"
    level: np.ndarray  # per plane, central-ROI clipped mean
    error: np.ndarray  # per plane, standard error (systematic for metadata/dark sources)
    map: np.ndarray | None  # full-resolution offset map (bias mean), if from frames
    exact: bool  # True when measured from zero-exposure frames
    label: str = ""
    stats: SetStats | None = None
    exposure: float = 0.0  # > 0 when the reference is a dark: it then includes dark signal

    @property
    def known(self) -> bool:
        return self.source != "none"


def plane_level(stats: SetStats, image: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Clipped mean and its standard error per plane over the central ROI of a mean map."""
    image = stats.mean_map if image is None else image
    levels, errors = [], []
    for _, plane in stats.layout.planes(image):
        roi = plane[central_roi(plane.shape)]
        clip = clipped_stats(roi[::2, ::2])
        levels.append(clip.mean)
        errors.append(clip.std / math.sqrt(max(clip.n, 1)))
    return np.array(levels), np.array(errors)


def find_offset(data: ModeData, exposure: float | None = None) -> Offset:
    """Best available zero-signal level. ``exposure`` prefers a matching dark flat."""
    if exposure is not None:
        for fs, st in data.of(FrameType.DARK_FLAT):
            if fs.exposure and abs(fs.exposure - exposure) <= 0.05 * exposure:
                level, err = plane_level(st)
                return Offset("dark flat", level, err, st.mean_map, True, fs.label, st)
    bias = data.best(FrameType.BIAS)
    if bias is not None:
        fs, st = bias
        level, err = plane_level(st)
        return Offset(
            "master bias" if fs.is_master else "bias", level, err, st.mean_map, True, fs.label, st
        )
    blacks = [st.black_meta for _, st in data.all(subs=True) if np.isfinite(st.black_meta).any()]
    if blacks:
        values = np.nanmedian(np.concatenate(blacks, axis=0), axis=0)
        spread = np.nanstd(np.concatenate(blacks, axis=0), axis=0)
        return Offset(
            "raw black level",
            values,
            np.maximum(spread, 0.5),
            None,
            False,
            "black level from raw metadata",
        )
    darks = [(fs, st) for fs, st in data.of(FrameType.DARK, subs=None)]
    if darks:
        fs, st = min(darks, key=lambda d: (d[0].exposure or 0, d[0].is_master))
        level, err = plane_level(st)
        return Offset("short dark", level, err, None, False, fs.label, st, fs.exposure or 0.0)
    n = len(data.planes)
    return Offset("none", np.full(n, np.nan), np.full(n, np.nan), None, False)
