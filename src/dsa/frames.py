"""Frame records: what a file is, how it was captured, and how to read its pixels.

Metadata comes from FITS headers (NINA, Siril, ...) and camera EXIF (Canon CR2, Sony ARW, any
LibRaw format). Folder and file names are a fallback for files without a frame-type header,
which is every DSLR raw.

Pixels are always returned as a 2-D array of raw sensor values, top row first, with the colour
filter layout that applies to that orientation. Nothing is debayered, black-subtracted or
rescaled: the analyses need the numbers the ADC produced. Masters stored as normalised floats
(Siril writes 0..1) are scaled back to ADU.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from dsa.cfa import CfaLayout

FITS_EXTENSIONS = frozenset({".fit", ".fits", ".fts"})
RAW_EXTENSIONS = frozenset(
    {".cr2", ".cr3", ".arw", ".nef", ".nrw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
)
TIFF_EXTENSIONS = frozenset({".tif", ".tiff"})
FRAME_EXTENSIONS = FITS_EXTENSIONS | RAW_EXTENSIONS | TIFF_EXTENSIONS


class FrameType(StrEnum):
    BIAS = "bias"
    DARK = "dark"
    FLAT = "flat"
    DARK_FLAT = "dark_flat"
    LIGHT = "light"

    @property
    def label(self) -> str:
        return {"dark_flat": "Dark flat"}.get(self.value, self.value.capitalize())


class Format(StrEnum):
    FITS = "fits"
    RAW = "raw"
    TIFF = "tiff"


@dataclass(frozen=True)
class FrameInfo:
    """Everything known about one file without reading its pixels."""

    path: Path
    format: Format
    kind: FrameType | None  # None: could not be classified from metadata or path
    kind_source: str  # "header", "folder", "filename", "content" or "unknown"
    camera: str | None
    width: int
    height: int
    cfa: str | None = None
    exposure: float | None = None
    iso: int | None = None
    gain: float | None = None  # camera gain *setting* (FITS GAIN), not e-/ADU
    offset: float | None = None
    readout: str | None = None
    binning: int = 1
    temperature: float | None = None
    set_temperature: float | None = None
    filter: str | None = None
    timestamp: datetime | None = None
    is_master: bool = False
    n_combined: int | None = None
    white_level: float | None = None  # from raw metadata, in file ADU
    bit_depth: int | None = None  # nominal container bits
    pixel_size: float | None = None  # micrometres
    focal_ratio: float | None = None
    compressed: bool | None = None  # lossy raw compression (e.g. Sony cRAW)
    software: str | None = None
    size: int = 0
    mtime: float = 0.0

    @property
    def layout(self) -> CfaLayout:
        return CfaLayout(self.cfa)

    @property
    def setting(self) -> str:
        """Human-readable sensor setting: ``ISO 1600`` or ``gain 100 / offset 500``."""
        if self.iso is not None:
            return f"ISO {self.iso}"
        parts = []
        if self.gain is not None:
            parts.append(f"gain {self.gain:g}")
        if self.offset is not None:
            parts.append(f"offset {self.offset:g}")
        if self.readout:
            parts.append(self.readout)
        return " / ".join(parts) or "default"


# ---------------------------------------------------------------- normalisation --


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def normalize_camera(name: str | None) -> str | None:
    """Drop connection suffixes drivers append (``"ATR585M(USB2.0)"`` -> ``"ATR585M"``)."""
    if not name:
        return None
    text = re.sub(r"\s*[(\[][^)\]]*[)\]]\s*$", "", str(name)).strip()
    return " ".join(text.split()) or None


_FILTER_ALIASES = {
    "l": "L",
    "lum": "L",
    "luminance": "L",
    "r": "R",
    "red": "R",
    "g": "G",
    "green": "G",
    "b": "B",
    "blue": "B",
    "h": "Ha",
    "ha": "Ha",
    "halpha": "Ha",
    "o": "OIII",
    "o3": "OIII",
    "oiii": "OIII",
    "s": "SII",
    "s2": "SII",
    "sii": "SII",
}


def normalize_filter(name: str | None) -> str | None:
    if name is None or not str(name).strip():
        return None
    return _FILTER_ALIASES.get(_key(name), str(name).strip())


def _type_from_word(word: str) -> tuple[FrameType | None, bool]:
    """Classify an IMAGETYP value, folder name or file-name token. Returns (type, is_master).

    Prefix matching accepts the names people actually use: ``Bias1600``, ``biases``, ``Darks``,
    ``FLAT``, ``Lighs`` (sic), ``MasterOffset_ISO3200``, ``Dark Frame``.
    """
    key = re.sub(r"[^a-z]", "", word.lower())
    is_master = key.startswith("master")
    key = key.removeprefix("master")
    if key.startswith(("darkflat", "flatdark")):
        return FrameType.DARK_FLAT, is_master
    if key.startswith(("bias", "offset", "zero")):
        return FrameType.BIAS, is_master
    if key.startswith("dark") or key == "d":
        return FrameType.DARK, is_master
    if key.startswith("flat"):
        return FrameType.FLAT, is_master
    if key.startswith(("light", "lighs", "object", "science")):
        return FrameType.LIGHT, is_master
    return None, is_master


def classify_path(path: Path) -> tuple[FrameType | None, str, bool]:
    """Frame type from the file name, then the nearest typed folder."""
    for token in re.split(r"[_\-\s.]+", path.stem):
        kind, master = _type_from_word(token)
        if kind is not None:
            return kind, "filename", master
    for part in reversed(path.parent.parts[-4:]):
        kind, master = _type_from_word(part)
        if kind is not None:
            return kind, "folder", master
    return None, "unknown", path.stem.lower().startswith("master")


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def parse_timestamp(value: Any) -> datetime | None:
    """ISO (FITS, NINA's 7-digit fractions) or EXIF ``YYYY:MM:DD HH:MM:SS`` timestamps."""
    if not value:
        return None
    text = str(value).strip().rstrip("Z")
    text = re.sub(r"^(\d{4}):(\d{2}):(\d{2}) ", r"\1-\2-\3T", text)
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


# ------------------------------------------------------------------------- FITS --


def _fits_info(path: Path, stat_size: int, mtime: float) -> FrameInfo:
    from astropy.io import fits

    header = fits.getheader(path, 0)
    naxis = int(header.get("NAXIS", 0))
    if naxis < 2:
        raise ValueError("no image data")
    if naxis == 3 and int(header.get("NAXIS3", 1)) > 1:
        raise ValueError("colour (3-plane) image: not raw sensor data")
    kind, master = _type_from_word(str(header.get("IMAGETYP", "")))
    source = "header"
    if kind is None:
        kind, source, master = classify_path(path)
    n_combined = header.get("STACKCNT", header.get("NCOMBINE"))
    is_master = master or n_combined is not None
    pattern = _text(header.get("BAYERPAT"))
    pattern = (
        pattern.upper() if pattern and pattern.upper() in {"RGGB", "BGGR", "GRBG", "GBRG"} else None
    )
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    bitpix = int(header.get("BITPIX", 16))
    exposure = _float(header.get("EXPTIME", header.get("EXPOSURE")))
    if exposure is not None and not 0 <= exposure < 1e6:  # garbage seen in Siril output (5e-315)
        exposure = None
    iso = _float(header.get("ISOSPEED"))
    gain = None if iso else _float(header.get("GAIN"))
    focal = _float(header.get("FOCRATIO"))
    return FrameInfo(
        path=path,
        format=Format.FITS,
        kind=kind,
        kind_source=source,
        camera=normalize_camera(_text(header.get("INSTRUME"))),
        width=width,
        height=height,
        cfa=pattern,
        exposure=exposure,
        iso=int(iso) if iso else None,
        gain=gain,
        offset=_float(header.get("OFFSET")),
        readout=_text(header.get("READOUTM")),
        binning=int(_float(header.get("XBINNING")) or 1),
        temperature=_float(header.get("CCD-TEMP")),
        set_temperature=_float(header.get("SET-TEMP")),
        filter=normalize_filter(_text(header.get("FILTER"))),
        timestamp=parse_timestamp(header.get("DATE-OBS") or header.get("DATE-LOC")),
        is_master=is_master,
        n_combined=int(n_combined) if n_combined is not None else None,
        bit_depth=abs(bitpix) if bitpix > 0 else None,
        pixel_size=_float(header.get("XPIXSZ")),
        focal_ratio=focal if focal and focal > 0 else None,
        software=_text(header.get("SWCREATE") or header.get("PROGRAM")),
        size=stat_size,
        mtime=mtime,
    )


def _read_fits(info: FrameInfo) -> np.ndarray:
    from astropy.io import fits

    with fits.open(info.path, memmap=False) as hdul:
        hdu = hdul[0]
        header = hdu.header
        data = hdu.data
        if data.ndim == 3:
            data = data[0]
        rows_bottom_up = str(header.get("ROWORDER", "")).strip().upper() == "BOTTOM-UP"
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32, copy=False)
        # Siril stores float images normalised to 0..1 (of the 16-bit range).
        finite = data[np.isfinite(data)]
        if finite.size and float(np.max(finite)) <= 2.0:
            data = data * np.float32(65535.0)
    elif data.dtype != np.uint16 and data.min() >= 0 and data.max() <= 65535:
        data = data.astype(np.uint16)
    if rows_bottom_up:
        data = data[::-1]
    return np.ascontiguousarray(data)


# -------------------------------------------------------------------------- RAW --


def _raw_pattern(raw: Any) -> str | None:
    """CFA pattern of the visible area from LibRaw (``None`` for non-Bayer sensors)."""
    desc = raw.color_desc.decode()
    pattern = raw.raw_pattern
    if len(desc) < 3 or pattern is None or pattern.shape != (2, 2):
        return None
    letters = "".join(desc[i] for i in pattern.ravel())
    return letters if letters in {"RGGB", "BGGR", "GRBG", "GBRG"} else None


def _exif(path: Path, details: bool) -> dict[str, Any]:
    import exifread

    with path.open("rb") as fh:
        return dict(exifread.process_file(fh, details=details, debug=False))


def _exif_value(tags: dict[str, Any], *names: str) -> Any:
    for name in names:
        tag = tags.get(name)
        if tag is None:
            continue
        values = getattr(tag, "values", None)
        if isinstance(values, list) and values:
            return values[0]
        return str(tag)
    return None


def _ratio(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "num") and hasattr(value, "den"):
        return value.num / value.den if value.den else None
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None


def _raw_info(path: Path, stat_size: int, mtime: float) -> FrameInfo:
    import rawpy

    raw = rawpy.RawPy()
    try:
        raw.open_file(str(path))  # parses metadata only; unpacking pixels is 3x slower
        width, height = raw.sizes.width, raw.sizes.height
        pattern = _raw_pattern(raw)
        white = float(raw.white_level)
    finally:
        raw.close()
    make_is_canon = path.suffix.lower() in {".cr2", ".cr3"}
    tags = _exif(path, details=make_is_canon)  # Canon keeps the sensor temperature in MakerNote
    model = _text(_exif_value(tags, "Image Model"))
    make = _text(_exif_value(tags, "Image Make"))
    camera = model
    if make and model and not model.lower().startswith(make.split()[0].lower()):
        camera = f"{make.split()[0].title()} {model}"
    kind, source, master = classify_path(path)
    temperature = _canon_temperature(_exif_value(tags, "MakerNote CameraTemperature"))
    # Sony writes the same compression tag for lossy cRAW and uncompressed raw; the data rate
    # tells them apart (cRAW is ~8 bits per pixel, uncompressed 16).
    bits_per_pixel = stat_size * 8 / max(width * height, 1)
    return FrameInfo(
        path=path,
        format=Format.RAW,
        kind=kind,
        kind_source=source,
        camera=normalize_camera(camera),
        width=width,
        height=height,
        cfa=pattern,
        exposure=_ratio(_exif_value(tags, "EXIF ExposureTime")),
        iso=int(_float(_exif_value(tags, "EXIF ISOSpeedRatings")) or 0) or None,
        temperature=temperature,
        timestamp=parse_timestamp(_exif_value(tags, "EXIF DateTimeOriginal", "Image DateTime")),
        is_master=master,
        white_level=white,
        bit_depth=max(8, math.ceil(math.log2(white + 1))) if white > 0 else None,
        compressed=path.suffix.lower() == ".arw" and bits_per_pixel < 11,
        software=_text(_exif_value(tags, "Image Software")),
        size=stat_size,
        mtime=mtime,
    )


def _canon_temperature(value: Any) -> float | None:
    """Canon's MakerNote reports the sensor temperature as e.g. ``'29 C'``."""
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(value)) if value is not None else None
    temperature = float(match.group(1)) if match else None
    return temperature if temperature is not None and -40 < temperature < 80 else None


def _read_raw(info: FrameInfo) -> Pixels:
    import rawpy

    with rawpy.imread(str(info.path)) as raw:
        data = np.array(raw.raw_image_visible, dtype=np.uint16, copy=True)
        levels = raw.black_level_per_channel
        colours = raw.raw_colors_visible[:2, :2]
        # LibRaw's per-channel black (Canon: measured on the masked border of *this* frame).
        black = {
            name: float(levels[colours[dy, dx]])
            for name, (dy, dx) in CfaLayout(info.cfa).offsets.items()
        }
    return Pixels(data, black)


# ------------------------------------------------------------------------- TIFF --


def _tiff_info(path: Path, stat_size: int, mtime: float) -> FrameInfo:
    """Single-channel TIFFs: only masters (DSS ``MasterOffset_ISO1600.tif``) are accepted."""
    import tifffile

    kind, source, master = classify_path(path)
    if not master or kind is None:
        raise ValueError("TIFF that is not a named calibration master (likely a processed image)")
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if page.samplesperpixel != 1:
            raise ValueError("colour TIFF: not raw sensor data")
        height, width = page.shape[:2]
        bits = page.bitspersample
    iso = re.search(r"ISO(\d+)", path.stem, re.IGNORECASE)
    exposure = re.search(r"_(\d+(?:\.\d+)?)s(?:_|$)", path.stem)
    description = path.with_name(path.stem + ".Description.txt")
    n_combined = None
    if description.exists():
        text = description.read_text(errors="replace")
        files = re.findall(r"\.(?:cr2|arw|nef|dng|fits?)\b", text, re.IGNORECASE)
        n_combined = len(files) or None
    return FrameInfo(
        path=path,
        format=Format.TIFF,
        kind=kind,
        kind_source=source,
        camera=None,
        width=width,
        height=height,
        exposure=float(exposure.group(1)) if exposure else None,
        iso=int(iso.group(1)) if iso else None,
        is_master=True,
        n_combined=n_combined,
        bit_depth=bits,
        software="DeepSkyStacker" if description.exists() else None,
        size=stat_size,
        mtime=mtime,
    )


def _read_tiff(info: FrameInfo) -> np.ndarray:
    import tifffile

    data = tifffile.imread(info.path)
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
        if float(np.nanmax(data)) <= 2.0:
            data = data * np.float32(65535.0)
    return np.ascontiguousarray(data)


# --------------------------------------------------------------------- public API --


def read_info(path: Path) -> FrameInfo:
    """Read a file's metadata (no pixels). Raises ``ValueError`` for files that aren't frames."""
    stat = path.stat()
    suffix = path.suffix.lower()
    if suffix in FITS_EXTENSIONS:
        return _fits_info(path, stat.st_size, stat.st_mtime)
    if suffix in RAW_EXTENSIONS:
        return _raw_info(path, stat.st_size, stat.st_mtime)
    if suffix in TIFF_EXTENSIONS:
        return _tiff_info(path, stat.st_size, stat.st_mtime)
    raise ValueError(f"unsupported file type {suffix}")


@dataclass(frozen=True)
class Pixels:
    """Raw sensor values, top row first (uint16 for subs, float32 for float masters)."""

    data: np.ndarray
    black_level: dict[str, float] | None = None  # per colour plane, from raw metadata


def read_pixels(info: FrameInfo) -> Pixels:
    if info.format is Format.FITS:
        return Pixels(_read_fits(info))
    if info.format is Format.RAW:
        return _read_raw(info)
    return Pixels(_read_tiff(info))


def with_kind(info: FrameInfo, kind: FrameType, source: str) -> FrameInfo:
    return replace(info, kind=kind, kind_source=source)
