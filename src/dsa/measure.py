"""Stream the frames of a set once and keep every statistic the analyses need.

Reading pixels is the expensive part (a 24 MP raw takes ~0.3 s to decode, a set can be tens of
GB), so each set is read exactly once, frames are decoded ahead on worker threads, and the
result (:class:`SetStats`) is cached on disk keyed by the files' identities.

Per frame and colour plane: level statistics, clipping fractions and row/column mean profiles.
Per pair of consecutive frames (0-1, 2-3, ...; disjoint so pairs are independent): the mean and
the variance of their difference, which is free of every fixed pattern; on flats also per tile,
which turns the illumination falloff into a range of signal levels. Per pixel: the temporal mean
and variance over the whole set (Welford-style, float64).
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path
from threading import Event

import numpy as np

from dsa.catalog import FrameSet
from dsa.cfa import CfaLayout
from dsa.frames import FrameInfo, FrameType, Pixels, read_pixels
from dsa.stats import central_roi, clipped_stats, tiles

STATS_VERSION = 3
TILE = 32  # tile edge in plane pixels for the flat-field photon transfer
CODE_HISTOGRAM_FRAMES = 4

Progress = Callable[[float, str], None]


class Cancelled(Exception):
    pass


@dataclass
class SetStats:
    """Everything measured from one frame set. Arrays are indexed ``[frame/pair, plane, ...]``."""

    set_key: str
    kind: str
    exposure: float
    is_master: bool
    n_combined: int
    n_total: int  # frames in the set
    cfa: str | None
    height: int
    width: int
    white_level: float  # highest code the data can reach (file ADU)
    planes: list[str]
    integer_data: bool
    # per frame [n, planes]
    frame_mean: np.ndarray
    frame_median: np.ndarray
    frame_std: np.ndarray  # spatial, clipped, central ROI (includes fixed pattern)
    frame_min: np.ndarray
    frame_max: np.ndarray
    frac_low: np.ndarray  # fraction of pixels at (or below) 0 ADU: clipped by too low an offset
    frac_high: np.ndarray  # fraction at or above the white level
    black_meta: np.ndarray  # black level from raw metadata (NaN if none)
    times: np.ndarray  # [n] unix seconds (NaN unknown)
    temps: np.ndarray  # [n] deg C (NaN unknown)
    row_means: list[np.ndarray]  # per plane [n, rows]
    col_means: list[np.ndarray]  # per plane [n, cols]
    # per pair [pairs, planes]
    pair_mean: np.ndarray
    pair_dmean: np.ndarray
    pair_var: np.ndarray  # temporal variance per frame = clipped var(diff) / 2, central ROI
    pair_rejected: np.ndarray
    pair_npix: np.ndarray
    pair_rho_x: np.ndarray  # lag-1 correlation of the difference image along rows
    pair_rho_y: np.ndarray
    tile_mean: np.ndarray | None  # [pairs, planes, ty, tx] flats only
    tile_var: np.ndarray | None
    # per pixel
    mean_map: np.ndarray  # float32 [h, w]
    var_map: np.ndarray | None  # float32 [h, w], temporal (ddof=1), needs >= 2 frames
    code_hist: np.ndarray | None  # [planes, 65536] counts of each code (integer data only)
    paths: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.frame_mean.shape[0])

    @property
    def n_pairs(self) -> int:
        return int(self.pair_mean.shape[0])

    @property
    def layout(self) -> CfaLayout:
        return CfaLayout(self.cfa)

    # ----------------------------------------------------------------- persistence --

    def save(self, path: Path) -> None:
        arrays, meta = {}, {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                arrays[f.name] = value
            elif f.name in ("row_means", "col_means"):
                for i, arr in enumerate(value):
                    arrays[f"{f.name}_{i}"] = arr
            else:
                meta[f.name] = value
        meta["version"] = STATS_VERSION
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, __meta__=np.array(json.dumps(meta)), **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> SetStats | None:
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["__meta__"]))
                if meta.pop("version", None) != STATS_VERSION:
                    return None
                kwargs = dict(meta)
                for f in fields(cls):
                    if f.name in ("row_means", "col_means"):
                        kwargs[f.name] = [data[f"{f.name}_{i}"] for i in range(len(meta["planes"]))]
                    elif f.name not in meta:
                        kwargs[f.name] = data.get(f.name, None)
                return cls(**kwargs)
        except (OSError, ValueError, KeyError):
            return None


# ----------------------------------------------------------------------- reading --


def iter_pixels(
    frames: Sequence[FrameInfo], workers: int = 3, cancel: Event | None = None
) -> Iterator[tuple[FrameInfo, Pixels]]:
    """Decode frames in order, ``workers`` frames ahead of the consumer."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue: deque[tuple[FrameInfo, Future[Pixels]]] = deque()
        pending = iter(frames)
        for info in pending:
            queue.append((info, pool.submit(read_pixels, info)))
            if len(queue) >= workers:
                break
        while queue:
            if cancel is not None and cancel.is_set():
                for _, fut in queue:
                    fut.cancel()
                raise Cancelled
            info, fut = queue.popleft()
            nxt = next(pending, None)
            if nxt is not None:
                queue.append((nxt, pool.submit(read_pixels, nxt)))
            yield info, fut.result()


def pick_frames(frames: Sequence[FrameInfo], limit: int) -> list[FrameInfo]:
    """At most ``limit`` frames, evenly spread over the set (keeps drift visible)."""
    if len(frames) <= limit:
        return list(frames)
    idx = np.unique(np.linspace(0, len(frames) - 1, limit).round().astype(int))
    return [frames[i] for i in idx]


def white_level_of(info: FrameInfo, data: np.ndarray) -> float:
    if info.white_level:
        return float(info.white_level)
    if np.issubdtype(data.dtype, np.integer):
        return float(np.iinfo(data.dtype).max)
    return 65535.0


# ------------------------------------------------------------------------ measure --


def _lag_correlation(diff: np.ndarray, sigma: float, centre: float) -> tuple[float, float]:
    """Lag-1 correlation along x and y of a (winsorised) difference image."""
    if not sigma > 0:
        return math.nan, math.nan
    d = np.clip(diff - centre, -5 * sigma, 5 * sigma)
    var = float(np.mean(d * d))
    if var <= 0:
        return math.nan, math.nan
    rx = float(np.mean(d[:, :-1] * d[:, 1:])) / var
    ry = float(np.mean(d[:-1, :] * d[1:, :])) / var
    return rx, ry


class _Accumulator:
    def __init__(self, fs: FrameSet, frames: list[FrameInfo], first: np.ndarray) -> None:
        self.layout = CfaLayout(fs.frames[0].cfa)
        self.planes = list(self.layout.names)
        self.integer = bool(np.issubdtype(first.dtype, np.integer))
        self.ref = first.astype(np.float32)
        self.s1 = np.zeros(first.shape, dtype=np.float64)
        self.s2 = np.zeros(first.shape, dtype=np.float64)
        self.n = 0
        self.per_frame: list[dict] = []
        self.pairs: list[dict] = []
        self.rows: list[list[np.ndarray]] = [[] for _ in self.planes]
        self.cols: list[list[np.ndarray]] = [[] for _ in self.planes]
        self.hist = np.zeros((len(self.planes), 65536), dtype=np.int64) if self.integer else None
        self.previous: np.ndarray | None = None
        self.want_tiles = fs.kind is FrameType.FLAT
        self.tile_means: list[np.ndarray] = []
        self.tile_vars: list[np.ndarray] = []
        self.white = 0.0

    def add(self, info: FrameInfo, pixels: Pixels) -> None:
        data = pixels.data
        if data.shape != self.ref.shape:
            raise ValueError(f"{info.path.name}: size {data.shape} differs from the set")
        x = data.astype(np.float32)
        d = (x - self.ref).astype(np.float64)
        self.s1 += d
        self.s2 += d * d
        self.n += 1
        self.white = max(self.white, white_level_of(info, data))
        stats: dict[str, list[float]] = {
            k: [] for k in ("mean", "median", "std", "min", "max", "low", "high", "black")
        }
        for i, (name, plane) in enumerate(self.layout.planes(x)):
            roi = plane[central_roi(plane.shape)]
            clip = clipped_stats(roi[::2, ::2])  # spatial stats: subsample is plenty
            lo, hi = float(plane.min()), float(plane.max())
            stats["mean"].append(float(plane.mean(dtype=np.float64)))
            stats["median"].append(float(np.median(roi)))
            stats["std"].append(clip.std)
            stats["min"].append(lo)
            stats["max"].append(hi)
            stats["low"].append(float(np.mean(plane <= 0)) if lo <= 0 else 0.0)
            stats["high"].append(float(np.mean(plane >= self.white)) if self.integer else 0.0)
            black = (pixels.black_level or {}).get(name)
            stats["black"].append(math.nan if black is None else black)
            self.rows[i].append(plane.mean(axis=1, dtype=np.float64).astype(np.float32))
            self.cols[i].append(plane.mean(axis=0, dtype=np.float64).astype(np.float32))
            if self.hist is not None and self.n <= CODE_HISTOGRAM_FRAMES:
                self.hist[i] += np.bincount(self.layout.plane(data, name).ravel(), minlength=65536)
        stats["time"] = [info.timestamp.timestamp() if info.timestamp else math.nan]
        stats["temp"] = [math.nan if info.temperature is None else info.temperature]
        self.per_frame.append(stats)

        if self.previous is None:
            self.previous = x
            return
        self._add_pair(self.previous, x)
        self.previous = None

    def _add_pair(self, a: np.ndarray, b: np.ndarray) -> None:
        diff = a - b
        total = 0.5 * (a + b)
        pair: dict[str, list[float]] = {
            k: [] for k in ("mean", "dmean", "var", "rej", "npix", "rx", "ry")
        }
        tm, tv = [], []
        for name in self.planes:
            dp = self.layout.plane(diff, name)
            sp = self.layout.plane(total, name)
            roi = central_roi(dp.shape)
            clip = clipped_stats(dp[roi])
            pair["mean"].append(float(sp[roi].mean(dtype=np.float64)))
            pair["dmean"].append(float(clip.mean))
            pair["var"].append(clip.std**2 / 2)
            pair["rej"].append(clip.rejected)
            pair["npix"].append(float(clip.n))
            rx, ry = _lag_correlation(dp[roi], clip.std, clip.mean)
            pair["rx"].append(rx)
            pair["ry"].append(ry)
            if self.want_tiles:
                tm.append(tiles(sp, TILE).mean(axis=2, dtype=np.float64).astype(np.float32))
                tv.append(
                    (tiles(dp, TILE).var(axis=2, ddof=1, dtype=np.float64) / 2).astype(np.float32)
                )
        self.pairs.append(pair)
        if self.want_tiles:
            self.tile_means.append(np.stack(tm))
            self.tile_vars.append(np.stack(tv))

    def finish(self, fs: FrameSet, frames: list[FrameInfo]) -> SetStats:
        n = self.n
        mean_map = (self.ref + self.s1 / n).astype(np.float32)
        var_map = None
        if n >= 2:
            var_map = ((self.s2 - self.s1**2 / n) / (n - 1)).clip(min=0).astype(np.float32)

        def frame_arr(key: str) -> np.ndarray:
            return np.array([s[key] for s in self.per_frame], dtype=np.float64)

        def pair_arr(key: str) -> np.ndarray:
            if not self.pairs:
                return np.zeros((0, len(self.planes)))
            return np.array([p[key] for p in self.pairs], dtype=np.float64)

        return SetStats(
            set_key=fs.key,
            kind=fs.kind.value,
            exposure=float(fs.exposure or 0.0),
            is_master=fs.is_master,
            n_combined=int(fs.n_combined or (1 if not fs.is_master else 0)),
            n_total=fs.count,
            cfa=fs.frames[0].cfa,
            height=int(self.ref.shape[0]),
            width=int(self.ref.shape[1]),
            white_level=self.white,
            planes=self.planes,
            integer_data=self.integer,
            frame_mean=frame_arr("mean"),
            frame_median=frame_arr("median"),
            frame_std=frame_arr("std"),
            frame_min=frame_arr("min"),
            frame_max=frame_arr("max"),
            frac_low=frame_arr("low"),
            frac_high=frame_arr("high"),
            black_meta=frame_arr("black"),
            times=frame_arr("time")[:, 0],
            temps=frame_arr("temp")[:, 0],
            row_means=[np.stack(r) for r in self.rows],
            col_means=[np.stack(c) for c in self.cols],
            pair_mean=pair_arr("mean"),
            pair_dmean=pair_arr("dmean"),
            pair_var=pair_arr("var"),
            pair_rejected=pair_arr("rej"),
            pair_npix=pair_arr("npix"),
            pair_rho_x=pair_arr("rx"),
            pair_rho_y=pair_arr("ry"),
            tile_mean=np.stack(self.tile_means) if self.tile_means else None,
            tile_var=np.stack(self.tile_vars) if self.tile_vars else None,
            mean_map=mean_map,
            var_map=var_map,
            code_hist=self.hist,
            paths=[str(f.path) for f in frames],
        )


def measure_set(
    fs: FrameSet,
    max_frames: int = 50,
    progress: Progress | None = None,
    cancel: Event | None = None,
) -> SetStats:
    """Read the frames of ``fs`` (at most ``max_frames``) and measure them."""
    frames = pick_frames(fs.frames, max_frames)
    acc: _Accumulator | None = None
    for i, (info, pixels) in enumerate(iter_pixels(frames, cancel=cancel)):
        if acc is None:
            acc = _Accumulator(fs, frames, pixels.data)
        acc.add(info, pixels)
        if progress is not None:
            progress((i + 1) / len(frames), f"{fs.label}: frame {i + 1}/{len(frames)}")
    assert acc is not None
    return acc.finish(fs, frames)


def pixel_series(
    frames: Sequence[FrameInfo], rows: np.ndarray, cols: np.ndarray, cancel: Event | None = None
) -> np.ndarray:
    """Values of selected pixels in every frame: ``[n_frames, n_pixels]`` (for RTS analysis)."""
    out = np.empty((len(frames), rows.size), dtype=np.float32)
    for i, (_, pixels) in enumerate(iter_pixels(frames, cancel=cancel)):
        out[i] = pixels.data[rows, cols]
    return out


class StatsCache:
    """On-disk cache of :class:`SetStats`, one ``.npz`` per set and frame limit."""

    def __init__(self, root: Path | None) -> None:
        self.root = root

    def _path(self, fs: FrameSet, max_frames: int) -> Path | None:
        if self.root is None:
            return None
        return self.root / "sets" / f"{fs.key}-{max_frames}-v{STATS_VERSION}.npz"

    def get(self, fs: FrameSet, max_frames: int) -> SetStats | None:
        path = self._path(fs, max_frames)
        return SetStats.load(path) if path is not None and path.exists() else None

    def put(self, fs: FrameSet, max_frames: int, stats: SetStats) -> None:
        path = self._path(fs, max_frames)
        if path is not None:
            stats.save(path)

    def measure(
        self,
        fs: FrameSet,
        max_frames: int = 50,
        progress: Progress | None = None,
        cancel: Event | None = None,
    ) -> SetStats:
        cached = self.get(fs, max_frames)
        if cached is not None:
            return cached
        stats = measure_set(fs, max_frames, progress, cancel)
        self.put(fs, max_frames, stats)
        return stats
