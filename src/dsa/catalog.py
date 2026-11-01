"""Find calibration frames in a folder tree and group them for analysis.

The hierarchy is camera → *mode* (one sensor setting: ISO, or gain/offset/readout mode) →
*frame sets* (frames of one type taken together: a bias series, 300 s darks at −5 °C, the Ha
flats of one session). Analyses work on the sets of one mode, because gain, read noise and
offset all change with the setting.

The scan never writes anywhere near the data. It tolerates the mess real libraries contain:
processing output next to raw frames, the same frames copied into several projects, masters
from different tools, DSLR raws whose only type information is the folder name.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from threading import Event

from dsa.frames import FRAME_EXTENSIONS, Format, FrameInfo, FrameType, read_info

# Processing output that sits next to raw data: Siril/our own work folders and products.
IGNORED_DIRS = frozenset({"process", "cache", "autostack", ".autostack", ".dsa-cache"})
DERIVED_PREFIXES = ("r_", "pp_", "bkg_", "rs_", "drz_")

Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class Mode:
    """One operating point of one sensor: frames of a mode share gain and offset."""

    camera: str
    iso: int | None
    gain: float | None
    offset: float | None
    readout: str | None
    binning: int
    width: int
    height: int
    cfa: str | None

    @property
    def setting(self) -> str:
        if self.iso is not None:
            return f"ISO {self.iso}"
        parts = []
        if self.gain is not None:
            parts.append(f"Gain {self.gain:g}")
        if self.offset is not None:
            parts.append(f"offset {self.offset:g}")
        if self.readout:
            parts.append(self.readout)
        if self.binning > 1:
            parts.append(f"bin {self.binning}")
        return " · ".join(parts) or "Default"

    @property
    def key(self) -> str:
        return _digest(repr(astuple_mode(self)))[:12]


def astuple_mode(mode: Mode) -> tuple:
    return tuple(getattr(mode, f.name) for f in fields(mode))


@dataclass
class FrameSet:
    """Frames of one type captured together (or one master)."""

    mode: Mode
    kind: FrameType
    frames: list[FrameInfo]
    exposure: float | None
    filter: str | None
    folder: Path
    is_master: bool

    @property
    def key(self) -> str:
        members = "|".join(f"{f.path}:{f.size}:{f.mtime}" for f in self.frames)
        return _digest(members)[:16]

    @property
    def count(self) -> int:
        return len(self.frames)

    @property
    def temperature(self) -> float | None:
        temps = [f.temperature for f in self.frames if f.temperature is not None]
        return statistics.fmean(temps) if temps else None

    @property
    def temperature_range(self) -> tuple[float, float] | None:
        temps = [f.temperature for f in self.frames if f.temperature is not None]
        return (min(temps), max(temps)) if temps else None

    @property
    def quantitative(self) -> bool:
        """False for masters whose values were rescaled by their tool (DeepSkyStacker TIFFs move
        the black point to 0 and rescale), so they can't be compared with raw ADU."""
        return not (self.is_master and self.frames[0].format is Format.TIFF)

    @property
    def n_combined(self) -> int | None:
        return self.frames[0].n_combined if self.is_master else None

    @property
    def label(self) -> str:
        parts = [("Master " + self.kind.label.lower()) if self.is_master else self.kind.label]
        if self.filter:
            parts.append(self.filter)
        if self.exposure is not None and self.kind is not FrameType.BIAS:
            parts.append(format_exposure(self.exposure))
        if (t := self.temperature) is not None:
            parts.append(f"{t:+.0f} °C")
        if self.is_master:
            if self.n_combined:
                parts.append(f"of {self.n_combined}")
        else:
            parts.append(f"{self.count} subs")
        return " · ".join(parts)


@dataclass
class Catalog:
    roots: list[Path]
    frames: list[FrameInfo] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    duplicates: list[tuple[Path, Path]] = field(default_factory=list)  # (dropped, kept)
    sets: list[FrameSet] = field(default_factory=list)

    @property
    def cameras(self) -> list[str]:
        return sorted({s.mode.camera for s in self.sets})

    def modes(self, camera: str) -> list[Mode]:
        modes = {s.mode for s in self.sets if s.mode.camera == camera}
        return sorted(modes, key=_mode_order)

    def sets_for(self, mode: Mode) -> list[FrameSet]:
        return [s for s in self.sets if s.mode == mode]


def format_exposure(seconds: float) -> str:
    if seconds <= 0:
        return "0 s"
    if seconds < 0.5:
        return f"1/{round(1 / seconds):d} s"
    return f"{seconds:.3g} s" if seconds < 10 else f"{seconds:.0f} s"


def _mode_order(mode: Mode) -> tuple:
    return (mode.iso or 0, mode.gain or 0, mode.offset or 0, mode.readout or "", mode.binning)


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- discovery --


def iter_frame_files(root: Path) -> Iterator[Path]:
    """Every candidate frame file under ``root``, skipping processing output folders."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d.lower() not in IGNORED_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            if Path(name).suffix.lower() in FRAME_EXTENSIONS:
                yield Path(dirpath) / name


class MetadataCache:
    """Remembers parsed metadata per file (keyed by path, size and mtime) between scans."""

    VERSION = 1

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._entries: dict[str, dict] = {}
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("version") == self.VERSION:
                    self._entries = data["entries"]
            except (OSError, ValueError, KeyError):
                self._entries = {}

    def get(self, path: Path, size: int, mtime: float) -> FrameInfo | str | None:
        entry = self._entries.get(str(path))
        if not entry or entry["size"] != size or entry["mtime"] != mtime:
            return None
        if "skip" in entry:
            return entry["skip"]
        return _info_from_dict(entry["info"])

    def put(self, path: Path, size: int, mtime: float, value: FrameInfo | str) -> None:
        entry: dict = {"size": size, "mtime": mtime}
        if isinstance(value, str):
            entry["skip"] = value
        else:
            entry["info"] = _info_to_dict(value)
        self._entries[str(path)] = entry

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"version": self.VERSION, "entries": self._entries}), encoding="utf-8"
        )
        tmp.replace(self.path)


def _info_to_dict(info: FrameInfo) -> dict:
    data = asdict(info)
    data["path"] = str(info.path)
    data["timestamp"] = info.timestamp.isoformat() if info.timestamp else None
    return data


def _info_from_dict(data: dict) -> FrameInfo:
    data = dict(data)
    data["path"] = Path(data["path"])
    data["format"] = Format(data["format"])
    data["kind"] = FrameType(data["kind"]) if data["kind"] else None
    data["timestamp"] = datetime.fromisoformat(data["timestamp"]) if data["timestamp"] else None
    return FrameInfo(**data)


def _read_one(path: Path, cache: MetadataCache) -> FrameInfo | str:
    try:
        stat = path.stat()
    except OSError as err:
        return f"unreadable: {err}"
    cached = cache.get(path, stat.st_size, stat.st_mtime)
    if cached is not None:
        return cached
    try:
        result: FrameInfo | str = read_info(path)
    except Exception as err:  # one corrupt file must not abort a scan of thousands
        result = str(err) or type(err).__name__
    cache.put(path, stat.st_size, stat.st_mtime, result)
    return result


def scan(
    roots: Iterable[Path],
    cache_path: Path | None = None,
    progress: Progress | None = None,
    cancel: Event | None = None,
    workers: int = 8,
) -> Catalog:
    """Index every frame under ``roots`` and group them into modes and frame sets."""
    roots = [Path(r) for r in roots]
    catalog = Catalog(roots)
    paths = [p for root in roots for p in iter_frame_files(root)]
    cache = MetadataCache(cache_path)
    infos: list[FrameInfo] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (path, result) in enumerate(
            zip(paths, pool.map(lambda p: _read_one(p, cache), paths), strict=True)
        ):
            if cancel is not None and cancel.is_set():
                break
            if progress is not None and (i % 25 == 0 or i == len(paths) - 1):
                progress((i + 1) / max(len(paths), 1), f"Reading metadata {i + 1}/{len(paths)}")
            if isinstance(result, str):
                catalog.skipped.append((path, result))
            else:
                infos.append(result)
    cache.save()
    kept, skipped = _filter_products(infos)
    catalog.skipped.extend(skipped)
    kept = _inherit_camera(kept)
    catalog.frames, catalog.duplicates = _deduplicate(kept)
    catalog.sets = group_sets(catalog.frames)
    return catalog


# ----------------------------------------------------------------------- cleanup --


def _filter_products(infos: list[FrameInfo]) -> tuple[list[FrameInfo], list[tuple[Path, str]]]:
    """Drop processing products: registered/calibrated frames, stacked lights, unknown types."""
    kept, skipped = [], []
    for info in infos:
        stem = info.path.stem.lower()
        if stem.startswith(DERIVED_PREFIXES) and not info.is_master:
            skipped.append((info.path, "processing output (calibrated or registered frame)"))
        elif info.kind is FrameType.LIGHT and info.is_master:
            skipped.append((info.path, "stacked image, not a calibration master"))
        elif info.kind is None:
            skipped.append((info.path, "frame type unknown (no IMAGETYP, no typed folder)"))
        else:
            kept.append(info)
    return kept, skipped


def _inherit_camera(infos: list[FrameInfo]) -> list[FrameInfo]:
    """Masters without camera metadata (DSS TIFFs) belong to the camera of nearby subs.

    Searches the master's folder first, then its parent, for subs with matching ISO and size.
    """
    by_folder: dict[Path, list[FrameInfo]] = defaultdict(list)
    for info in infos:
        if info.camera:
            by_folder[info.path.parent].append(info)
    out = []
    for info in infos:
        if info.camera is None:
            for folder in (info.path.parent, info.path.parent.parent):
                donors = [
                    d
                    for d in by_folder.get(folder, [])
                    if (d.width, d.height) == (info.width, info.height)
                    and (info.iso is None or d.iso == info.iso)
                ]
                if not donors:
                    # Same folder, any frame of the same size (e.g. a bias master next to flats).
                    donors = [
                        d
                        for d in by_folder.get(folder, [])
                        if (d.width, d.height) == (info.width, info.height)
                    ]
                if donors:
                    d = donors[0]
                    info = replace(
                        info,
                        camera=d.camera,
                        cfa=info.cfa or d.cfa,
                        iso=info.iso if info.iso is not None else d.iso,
                        white_level=info.white_level or d.white_level,
                    )
                    break
        out.append(info)
    return out


def _identity(info: FrameInfo) -> tuple:
    """Two files with this identity hold the same capture (copies in several projects)."""
    stamp = info.timestamp.isoformat() if info.timestamp else info.path.name
    return (info.camera, info.kind, info.is_master, stamp, info.exposure, info.size)


def _content_digest(path: Path, chunk: int = 1 << 18) -> str:
    """Hash of a slice from the middle of the file: distinguishes frames, cheap to read."""
    with path.open("rb") as fh:
        fh.seek(max(0, path.stat().st_size // 2 - chunk // 2))
        return hashlib.sha1(fh.read(chunk)).hexdigest()


def _deduplicate(infos: list[FrameInfo]) -> tuple[list[FrameInfo], list[tuple[Path, Path]]]:
    """Drop copies of the same file. Metadata alone is not enough: EXIF times have one-second
    resolution and Sony cRAW files have a fixed size, so burst frames share both. Candidates
    with equal metadata are compared by content.
    """
    seen: dict[tuple, list[tuple[str, FrameInfo]]] = defaultdict(list)
    kept, dropped = [], []
    for info in sorted(infos, key=lambda i: str(i.path).lower()):
        candidates = seen[_identity(info)]
        digest = _content_digest(info.path) if candidates else ""
        if candidates and not candidates[0][0]:
            candidates[0] = (_content_digest(candidates[0][1].path), candidates[0][1])
        original = next((other for d, other in candidates if d == digest), None)
        if original is not None:
            dropped.append((info.path, original.path))
        else:
            candidates.append((digest, info))
            kept.append(info)
    return kept, dropped


# ---------------------------------------------------------------------- grouping --


def mode_of(info: FrameInfo) -> Mode | None:
    if info.camera is None:
        return None
    return Mode(
        camera=info.camera,
        iso=info.iso,
        gain=info.gain if info.iso is None else None,
        offset=info.offset if info.iso is None else None,
        readout=info.readout,
        binning=info.binning,
        width=info.width,
        height=info.height,
        cfa=info.cfa,
    )


def _exposure_bucket(exposure: float | None) -> float | None:
    return None if exposure is None else float(f"{exposure:.3g}")


def _set_key(info: FrameInfo) -> tuple:
    """Frames sharing this key are one set.

    Sets never span folders (different nights, different ambient temperature for DSLRs).
    Flats are not split by exposure: DSLR flats in aperture priority vary slightly from frame
    to frame, and NINA uses one exposure per filter anyway.
    """
    exposure = None if info.kind is FrameType.FLAT else _exposure_bucket(info.exposure)
    set_temp = round(info.set_temperature) if info.set_temperature is not None else None
    master_id = str(info.path) if info.is_master else None
    return (info.kind, info.filter, exposure, set_temp, info.path.parent, master_id)


def group_sets(infos: list[FrameInfo]) -> list[FrameSet]:
    groups: dict[tuple[Mode, tuple], list[FrameInfo]] = defaultdict(list)
    for info in infos:
        mode = mode_of(info)
        if mode is not None:
            groups[(mode, _set_key(info))].append(info)
    sets = []
    for (mode, key), frames in groups.items():
        frames.sort(key=lambda f: (f.timestamp or datetime.min, str(f.path)))
        exposures = [f.exposure for f in frames if f.exposure is not None]
        sets.append(
            FrameSet(
                mode=mode,
                kind=key[0],
                frames=frames,
                exposure=statistics.median(exposures) if exposures else None,
                filter=key[1],
                folder=frames[0].path.parent,
                is_master=frames[0].is_master,
            )
        )
    kind_order = {k: i for i, k in enumerate(FrameType)}
    sets.sort(
        key=lambda s: (
            s.mode.camera,
            _mode_order(s.mode),
            kind_order[s.kind],
            s.is_master,
            s.filter or "",
            s.exposure or 0,
            str(s.folder),
        )
    )
    return sets
