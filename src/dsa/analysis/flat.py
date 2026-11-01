"""Flat-field analysis: pixel response non-uniformity (PRNU), vignetting, dust, colour response.

A flat combines three things at different spatial scales: pixel-to-pixel sensitivity (PRNU,
1-2 px), dust shadows (tens to hundreds of px) and vignetting (the whole frame). They are
separated by scale:

* PRNU (EMVA 1288 eq. 67): spatial rms of the averaged, offset-subtracted flat after dividing
  out a ~16 px smooth version, as a fraction of the signal, with the temporal noise left in an
  N-frame average (σ²_t / N) removed.
* Vignetting: a heavily median-binned flat (dust and defects removed) fitted with a radial
  polynomial around a free optical centre.
* Dust: dips in the ratio of the flat to a ~500 px background; dust shadows are donuts whose
  diameter is the dust's distance from the sensor divided by the focal ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, optimize, signal

from dsa.analysis.common import MISSING, Level, Section, Status, Value, weighted_mean
from dsa.analysis.data import ModeData
from dsa.analysis.gain import Gain
from dsa.analysis.levels import Offset, find_offset
from dsa.catalog import FrameSet
from dsa.cfa import CfaLayout
from dsa.frames import FrameType
from dsa.measure import SetStats
from dsa.stats import (
    bin_image,
    central_roi,
    clipped_stats,
    large_scale,
    polynomial_surface,
    robust_sigma,
)

PRNU_SCALE = 16
MIN_LEVEL = 0.05  # flats dimmer than this fraction of full scale are too noisy for PRNU/defects
DUST_BACKGROUND = 400  # px; median background, wider than a ring's annulus
# Matched-filter significance needed per template (z / robust sigma of the response).
Z_MIN = {"spot": 7.0, "ring": 5.0}


@dataclass
class DustMote:
    x: float  # full-resolution pixels
    y: float
    diameter: float  # full-resolution pixels
    depth: float  # peak light loss, fraction
    mean_depth: float
    donut: bool  # ring-shaped shadow (out-of-focus dust)
    distance_mm: float = math.nan  # from the sensor, if the focal ratio is known
    filters: list[str] = field(default_factory=list)


@dataclass
class Vignetting:
    map: np.ndarray  # normalised (max 1), binned
    bin_factor: int
    centre: tuple[float, float]  # optical centre, full-resolution pixels
    centre_offset: tuple[float, float]  # from the geometric centre, pixels
    corners: dict[str, float]  # remaining light fraction at each corner
    worst_corner: str
    falloff_stops: float  # at the worst corner
    asymmetry_x: float  # right edge / left edge − 1
    asymmetry_y: float  # bottom edge / top edge − 1
    radius: np.ndarray  # radial profile, full-resolution pixels
    profile: np.ndarray
    model: np.ndarray
    residual_rms: float
    plane_corners: dict[str, float] = field(default_factory=dict)  # mean corner per colour


@dataclass
class FlatSetResult:
    label: str
    filter: str | None
    is_master: bool
    n: int
    level: float  # median signal above offset, ADU
    level_fraction: float  # of the usable range (offset .. white level)
    prnu: list[Value]  # per plane, %
    prnu_combined: Value
    prnu_map: np.ndarray | None  # %, full resolution mosaic
    vignetting: Vignetting | None
    dust: list[DustMote]
    channel_ratios: dict[str, float] = field(default_factory=dict)
    offset_source: str = ""
    dust_map: np.ndarray | None = None  # light loss (fraction) relative to the background
    dust_scale: int = 1  # full-resolution pixels per dust_map pixel


@dataclass
class FlatField(Section):
    planes: list[str] = field(default_factory=list)
    sets: list[FlatSetResult] = field(default_factory=list)
    prnu: Value = MISSING
    dust: list[DustMote] = field(default_factory=list)  # merged over filters


def _calibrated(st: SetStats, offset: Offset) -> np.ndarray:
    image = st.mean_map.astype(np.float32)
    if offset.map is not None and offset.map.shape == image.shape and offset.exact:
        return image - offset.map
    out = image.copy()
    for i, (_, plane) in enumerate(st.layout.planes(out)):
        if math.isfinite(offset.level[i]):
            plane -= np.float32(offset.level[i])
    return out


def flat_field(
    data: ModeData, gain: Gain, pixel_size: float | None, focal_ratio: float | None
) -> FlatField:
    result = FlatField(planes=data.planes)
    chosen = _choose_sets(data)
    if not chosen:
        result.status = Status.UNAVAILABLE
        result.method = "Needs flat frames."
        result.unlock.append("Capture flats (10 or more per filter).")
        return result
    for fs, st in chosen:
        offset = find_offset(data, exposure=fs.exposure)
        result.sets.append(_analyse_set(fs, st, offset, data.layout, pixel_size, focal_ratio))
    usable = [s for s in result.sets if s.prnu_combined.ok and s.level_fraction >= MIN_LEVEL]
    for s in result.sets:
        if s.level_fraction < MIN_LEVEL:
            result.note(
                Level.WARN,
                f"{s.label}: the flat is at {s.level_fraction:.1%} of full "
                "scale, so its shot noise swamps PRNU; it is shown but not used for PRNU "
                "or defective pixels.",
            )
    values = [s.prnu_combined for s in usable if not s.is_master]
    if not values:
        values = [s.prnu_combined for s in usable]
    if values:
        mean, err, _ = weighted_mean(
            np.array([v.value for v in values]), np.array([v.error for v in values])
        )
        # Filters differ in wavelength, and PRNU depends on it: the spread is real, report it.
        spread = float(np.std([v.value for v in values], ddof=1)) if len(values) > 1 else math.nan
        result.prnu = Value(mean, max(err, spread) if math.isfinite(spread) else err, "%")
    result.dust = _merge_dust(result.sets)
    result.status = Status.MEASURED
    if all(s.is_master for s in result.sets):
        result.status = Status.LIMITED
        result.note(
            Level.INFO,
            "Only master flats: PRNU assumes the master still contains the "
            "offset and that its frame count is right.",
        )
    result.method = (
        "PRNU: EMVA 1288 on the averaged flat, high-pass at ~16 px, temporal residue "
        "removed. Vignetting: radial fit with free centre to a median-binned flat. "
        "Dust: dips below a ~500 px background."
    )
    if not focal_ratio:
        result.unlock.append(
            "Set the focal ratio (FOCRATIO in NINA's telescope settings) to "
            "estimate how far each dust mote is from the sensor."
        )
    return result


def _choose_sets(data: ModeData) -> list[tuple[FrameSet, SetStats]]:
    """Sub sets per filter; a master only for filters without subs."""
    subs = [(fs, st) for fs, st in data.of(FrameType.FLAT, subs=True) if st.n_frames >= 2]
    filters = {fs.filter for fs, _ in subs}
    masters = [
        (fs, st) for fs, st in data.of(FrameType.FLAT, subs=False) if fs.filter not in filters
    ]
    return subs + masters


def _analyse_set(
    fs: FrameSet,
    st: SetStats,
    offset: Offset,
    layout: CfaLayout,
    pixel_size: float | None,
    focal_ratio: float | None,
) -> FlatSetResult:
    flat = _calibrated(st, offset)
    n = st.n_combined if fs.is_master else st.n_frames
    prnu, hp_planes, levels = [], {}, {}
    for i, (name, plane) in enumerate(layout.planes(flat)):
        level = float(np.median(plane[central_roi(plane.shape)]))
        levels[name] = level
        smooth = large_scale(plane, PRNU_SCALE)
        hp = plane / np.maximum(smooth, 1e-6)
        hp_planes[name] = hp
        roi = central_roi(hp.shape, 0.8)
        clip = clipped_stats(hp[roi])
        s2 = clip.std**2
        temporal = 0.0
        if not fs.is_master and st.var_map is not None:
            # Residual temporal noise of the N-frame average, relative to the local signal
            # (larger towards vignetted corners), averaged like the spatial variance.
            rel_var = layout.plane(st.var_map, name)[roi] / np.maximum(smooth[roi], 1e-6) ** 2
            temporal = float(np.mean(rel_var)) / max(n, 1)
        if offset.map is not None and offset.stats is not None and offset.stats.n_pairs:
            # The subtracted offset map is itself an average with temporal noise left in it.
            off = offset.stats
            off_var = float(np.mean(off.pair_var[:, i])) / off.n_frames
            temporal += off_var * float(np.mean(1 / np.maximum(smooth[roi], 1e-6) ** 2))
        value = math.sqrt(max(s2 - temporal, 0.0)) * 100
        err = value * math.sqrt(1 / (2 * max(clip.n, 2)))
        prnu.append(Value(value, err, "%"))
    combined = Value(
        float(np.sqrt(np.mean([p.value**2 for p in prnu]))),
        float(np.mean([p.error for p in prnu])),
        "%",
    )
    prnu_map = (layout.merge(hp_planes, flat.shape) - 1) * 100
    lum = _luminance(flat, layout, levels)
    vig = vignetting(lum, layout, flat)
    dust, dust_map, dust_scale = find_dust(lum, layout.step, pixel_size, focal_ratio)
    for d in dust:
        d.filters = [fs.filter or "—"]
    ratios = {}
    if not layout.is_mono:
        g = (levels["G1"] + levels["G2"]) / 2
        ratios = {
            "R/G": levels["R"] / g,
            "B/G": levels["B"] / g,
            "G1/G2": levels["G1"] / levels["G2"],
        }
    level = float(np.mean(list(levels.values())))
    usable = st.white_level - float(np.nanmean(offset.level))
    return FlatSetResult(
        fs.label,
        fs.filter,
        fs.is_master,
        n,
        level,
        level / usable if usable > 0 else math.nan,
        prnu,
        combined,
        prnu_map.astype(np.float32),
        vig,
        dust,
        ratios,
        offset.source,
        dust_map,
        dust_scale,
    )


def _luminance(flat: np.ndarray, layout: CfaLayout, levels: dict[str, float]) -> np.ndarray:
    """Plane-normalised average at plane resolution: colour response divided out."""
    if layout.is_mono:
        return flat / max(levels["L"], 1e-6)
    planes = [layout.plane(flat, name) / max(levels[name], 1e-6) for name in layout.names]
    h = min(p.shape[0] for p in planes)
    w = min(p.shape[1] for p in planes)
    return np.mean([p[:h, :w] for p in planes], axis=0)


# --------------------------------------------------------------------------- vignetting --


def _radial(params: np.ndarray, x: np.ndarray, y: np.ndarray, norm: float) -> np.ndarray:
    x0, y0, a, b1, b2, b3 = params
    r2 = ((x - x0) ** 2 + (y - y0) ** 2) / norm**2
    return a * (1 + b1 * r2 + b2 * r2**2 + b3 * r2**3)


def vignetting(lum: np.ndarray, layout: CfaLayout, flat: np.ndarray) -> Vignetting | None:
    step = layout.step
    factor = max(1, max(lum.shape) // 192)
    small = bin_image(lum, factor, "median")
    small = ndimage.median_filter(small, size=3, mode="nearest")
    if small.size < 16:
        return None
    h, w = small.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # Bin centres in full-resolution pixels.
    scale = factor * step
    xs, ys = (xx + 0.5) * scale, (yy + 0.5) * scale
    full_w, full_h = w * scale, h * scale
    norm = math.hypot(full_w, full_h) / 2
    peak = float(np.percentile(small, 99.5))
    start = np.array([full_w / 2, full_h / 2, peak, -0.2, 0.0, 0.0])
    fit = optimize.least_squares(
        lambda p: (_radial(p, xs, ys, norm) - small).ravel(),
        start,
        loss="soft_l1",
        f_scale=0.005 * peak,
        bounds=([-full_w, -full_h, 0, -5, -5, -5], [2 * full_w, 2 * full_h, 10 * peak, 5, 5, 5]),
    )
    model = _radial(fit.x, xs, ys, norm)
    top = float(model.max())
    normalised = small / top
    corner_n = max(1, min(h, w) // 12)
    # Corners from the fitted model at the sensor's corner pixels: the binned map's corner bins
    # sit half a bin inwards and are noisier.
    corner_xy = {
        "top-left": (0, 0),
        "top-right": (full_w, 0),
        "bottom-left": (0, full_h),
        "bottom-right": (full_w, full_h),
    }
    corner_values = {
        k: float(_radial(fit.x, np.array(x, float), np.array(y, float), norm) / top)
        for k, (x, y) in corner_xy.items()
    }
    worst = min(corner_values, key=corner_values.get)
    mid_h, mid_w = slice(h // 3, 2 * h // 3), slice(w // 3, 2 * w // 3)
    left, right = (
        float(np.median(normalised[mid_h, :corner_n])),
        float(np.median(normalised[mid_h, -corner_n:])),
    )
    top_e, bottom = (
        float(np.median(normalised[:corner_n, mid_w])),
        float(np.median(normalised[-corner_n:, mid_w])),
    )
    r = np.hypot(xs - fit.x[0], ys - fit.x[1]).ravel()
    edges = np.linspace(0, r.max(), 41)
    idx = np.digitize(r, edges) - 1
    radius, profile, model_p = [], [], []
    for k in range(40):
        sel = idx == k
        if sel.sum() >= 3:
            radius.append(float(r[sel].mean()))
            profile.append(float(np.median(normalised.ravel()[sel])))
            model_p.append(float(np.mean(model.ravel()[sel] / top)))
    plane_corners = {}
    if not layout.is_mono:
        for name, plane in layout.planes(flat):
            pb = bin_image(plane, max(1, max(plane.shape) // 96), "median")
            c = max(1, min(pb.shape) // 12)
            centre = float(
                np.median(
                    pb[
                        pb.shape[0] // 3 : 2 * pb.shape[0] // 3,
                        pb.shape[1] // 3 : 2 * pb.shape[1] // 3,
                    ]
                )
            )
            corner = np.median(
                np.concatenate(
                    [
                        pb[:c, :c].ravel(),
                        pb[:c, -c:].ravel(),
                        pb[-c:, :c].ravel(),
                        pb[-c:, -c:].ravel(),
                    ]
                )
            )
            plane_corners[name] = float(corner / centre) if centre > 0 else math.nan
    return Vignetting(
        map=normalised.astype(np.float32),
        bin_factor=scale,
        centre=(float(fit.x[0]), float(fit.x[1])),
        centre_offset=(float(fit.x[0] - full_w / 2), float(fit.x[1] - full_h / 2)),
        corners=corner_values,
        worst_corner=worst,
        falloff_stops=-math.log2(max(corner_values[worst], 1e-6)),
        asymmetry_x=right / left - 1 if left > 0 else math.nan,
        asymmetry_y=bottom / top_e - 1 if top_e > 0 else math.nan,
        radius=np.array(radius),
        profile=np.array(profile),
        model=np.array(model_p),
        residual_rms=float(np.sqrt(np.mean((small - model) ** 2)) / top),
        plane_corners=plane_corners,
    )


# --------------------------------------------------------------------------------- dust --


def find_dust(
    lum: np.ndarray, step: int, pixel_size: float | None, focal_ratio: float | None
) -> tuple[list[DustMote], np.ndarray, int]:
    """Dust shadows as connected dips in the flat relative to its large-scale background.

    Two scales: sharp motes (sensor-window dust, a few % deep) at full working resolution, and
    large faint rings (filter dust, hundreds of px across and ~0.1–0.5 % deep) on a 4×-binned
    map, where averaging lowers the noise enough to see them.
    Returns the motes, the depth map they were found in and its scale (full-res px per pixel).
    """
    work_factor = max(1, max(lum.shape) // 2000)
    work = bin_image(lum, work_factor) if work_factor > 1 else lum.astype(np.float32)
    scale = work_factor * step
    # A polynomial takes out vignetting (median filters bias a curved surface), then a median
    # background of the flattened residual takes out what the polynomial missed.
    residual = work / np.maximum(polynomial_surface(work), 1e-6)
    background = large_scale(residual, max(64, DUST_BACKGROUND // scale))
    depth = 1 - residual / np.maximum(background, 1e-6)
    fine = ndimage.gaussian_filter(depth, 1.5)
    motes = _detect(fine, scale, floor=0.003, pixel_size=pixel_size, focal_ratio=focal_ratio)
    coarse = bin_image(depth, 4)
    for mote in _find_rings(coarse, 4 * scale, pixel_size, focal_ratio):
        # A strong mote also excites larger templates around it: keep rings clear of motes.
        if not any(_overlap(mote, m) for m in motes):
            motes.append(mote)
    motes.sort(key=lambda m: -m.depth * m.diameter**2)
    return motes[:200], fine.astype(np.float32), scale


def _templates(radius: float) -> dict[str, np.ndarray]:
    """Zero-mean templates for a shadow of ``radius``: a filled disc and a thin ring (dust far
    from the sensor seen through a centrally obstructed telescope), each against its surroundings.
    """
    size = int(np.ceil(1.4 * radius)) + 1
    yy, xx = np.mgrid[-size : size + 1, -size : size + 1]
    r = np.hypot(xx, yy)
    outside = (r > radius) & (r <= 1.35 * radius)
    shapes = {"spot": r <= radius, "ring": (r > 0.7 * radius) & (r <= radius)}
    out = {}
    for name, inside in shapes.items():
        t = np.zeros(r.shape)
        t[inside] = 1 / inside.sum()
        t[outside] = -1 / outside.sum()
        if name == "ring":
            core = r <= 0.55 * radius
            t[core] = -1 / core.sum()
            t[t < 0] /= 2  # core and surroundings share the negative weight
        out[name] = t
    return out


def _find_rings(
    depth: np.ndarray,
    scale: int,
    pixel_size: float | None,
    focal_ratio: float | None,
) -> list[DustMote]:
    """Large, faint dust shadows by matched filtering with discs and rings over many radii.

    A pixel threshold can't separate a 0.1 % ring from the flat's own large-scale wiggles of
    similar amplitude; a ring template responds to round shadows and averages the rest away.
    """
    # Row and column stripes (fixed pattern, flat-panel banding) would ring through the filter.
    depth = depth - np.median(depth, axis=0, keepdims=True)
    depth = depth - np.median(depth, axis=1, keepdims=True)
    h, w = depth.shape
    radii = np.geomspace(6, min(h, w) / 4, 14)
    best = np.zeros((h, w))
    best_r = np.zeros((h, w))
    best_kind = np.zeros((h, w), dtype=bool)  # True = ring
    for radius in radii:
        for kind, template in _templates(radius).items():
            response = signal.fftconvolve(depth, template[::-1, ::-1], mode="same")
            z = response / (robust_sigma(response[::2, ::2]) or 1e-12)
            # Disc templates also answer to blotchy illumination; rings are far more specific.
            z = z / Z_MIN[kind]
            better = z > best
            best = np.where(better, z, best)
            best_r = np.where(better, radius, best_r)
            best_kind = np.where(better, kind == "ring", best_kind)
    peaks = (best == ndimage.maximum_filter(best, size=9)) & (best > 1)
    motes: list[DustMote] = []
    for y, x in sorted(zip(*np.nonzero(peaks), strict=True), key=lambda p: -best[p]):
        r = best_r[y, x]
        if not (r < x < w - r and r < y < h - r):  # the whole shadow must be on the sensor
            continue
        mote = DustMote(
            x=float(x * scale + scale / 2),
            y=float(y * scale + scale / 2),
            diameter=float(2 * r * scale),
            depth=float(_ring_depth(depth, y, x, r)),
            mean_depth=float(_ring_depth(depth, y, x, r)),
            donut=bool(best_kind[y, x]),
        )
        if any(_overlap(mote, m) for m in motes):
            continue
        if pixel_size and focal_ratio:
            mote.distance_mm = mote.diameter * pixel_size / 1000 * focal_ratio
        motes.append(mote)
    return motes


def _ring_depth(depth: np.ndarray, y: int, x: int, radius: float) -> float:
    """Light lost in the shadow relative to just outside it."""
    size = int(np.ceil(1.4 * radius))
    patch = depth[max(0, y - size) : y + size + 1, max(0, x - size) : x + size + 1]
    yy, xx = np.mgrid[: patch.shape[0], : patch.shape[1]]
    r = np.hypot(yy - min(y, size), xx - min(x, size))
    inside = (r > 0.6 * radius) & (r <= radius)
    outside = (r > radius) & (r <= 1.35 * radius)
    return float(np.median(patch[inside]) - np.median(patch[outside]))


def _overlap(a: DustMote, b: DustMote) -> bool:
    return math.hypot(a.x - b.x, a.y - b.y) < (a.diameter + b.diameter) / 2


def _same_mote(a: DustMote, b: DustMote) -> bool:
    return math.hypot(a.x - b.x, a.y - b.y) < max(12.0, 0.3 * max(a.diameter, b.diameter))


def _detect(
    depth: np.ndarray, scale: int, floor: float, pixel_size: float | None, focal_ratio: float | None
) -> list[DustMote]:
    noise = robust_sigma(depth[::2, ::2])
    threshold = max(5 * noise, floor)
    mask = ndimage.binary_closing(depth > threshold, structure=np.ones((5, 5)))
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    labels, _count = ndimage.label(mask)
    h, w = depth.shape
    margin = 0.03 * min(h, w)  # the background is least certain at the edges and corners
    min_area = max(12, (8 // scale) ** 2)
    motes = []
    for k, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        sel = labels[region] == k
        area = int(sel.sum())
        if area < min_area:
            continue
        d = np.where(sel, depth[region], 0)
        if d.max() < 1.5 * threshold or _touches_border(region, depth.shape):
            continue
        cy, cx = ndimage.center_of_mass(d)
        y0, x0 = region[0].start + cy, region[1].start + cx
        if (
            region[0].start < margin
            or region[1].start < margin
            or region[0].stop > h - margin
            or region[1].stop > w - margin
        ):
            continue
        # Bounding-box size: an annulus's area would understate a ring's diameter.
        diameter = (sel.shape[0] + sel.shape[1]) / 2
        centre_depth = float(depth[round(y0), round(x0)])
        peak = float(d.max())
        mote = DustMote(
            x=float(x0 * scale + scale / 2),
            y=float(y0 * scale + scale / 2),
            diameter=diameter * scale,
            depth=peak,
            mean_depth=float(d[sel].mean()),
            donut=centre_depth < 0.7 * peak and diameter > 6,
        )
        if pixel_size and focal_ratio:
            mote.distance_mm = mote.diameter * pixel_size / 1000 * focal_ratio
        motes.append(mote)
    return motes


def _touches_border(region: tuple[slice, slice], shape: tuple[int, int]) -> bool:
    return (
        region[0].start == 0
        or region[1].start == 0
        or region[0].stop >= shape[0]
        or region[1].stop >= shape[1]
    )


def _merge_dust(sets: list[FlatSetResult]) -> list[DustMote]:
    """Same mote seen through several filters → one entry listing those filters."""
    merged: list[DustMote] = []
    for s in sets:
        for mote in s.dust:
            match = next((m for m in merged if _same_mote(m, mote)), None)
            if match is None:
                merged.append(
                    DustMote(
                        mote.x,
                        mote.y,
                        mote.diameter,
                        mote.depth,
                        mote.mean_depth,
                        mote.donut,
                        mote.distance_mm,
                        list(mote.filters),
                    )
                )
            else:
                match.filters = sorted(set(match.filters) | set(mote.filters))
                match.depth = max(match.depth, mote.depth)
    return merged
