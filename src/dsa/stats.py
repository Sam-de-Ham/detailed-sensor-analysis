"""Robust statistics and small numerical helpers shared by the analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

MAD_TO_SIGMA = 1.482602218505602  # 1 / Phi^-1(3/4): MAD of a normal distribution -> sigma


@dataclass(frozen=True)
class Clipped:
    """Mean and standard deviation of the values left after sigma clipping."""

    mean: float
    std: float
    n: int
    rejected: float  # fraction of values clipped


def robust_sigma(values: np.ndarray) -> float:
    """Standard deviation estimated from the median absolute deviation."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return math.nan
    med = np.median(values)
    return float(np.median(np.abs(values - med)) * MAD_TO_SIGMA)


def clipped_stats(values: np.ndarray, kappa: float = 5.0, iterations: int = 5) -> Clipped:
    """Iterative kappa-sigma clipped mean and standard deviation.

    Starts from median/MAD so a handful of hot pixels can't inflate the first estimate. At
    ``kappa = 5`` a normal distribution loses 6e-7 of its values, so the std needs no correction;
    what *is* removed are hot pixels, cosmic rays and RTS outliers, which are reported separately.
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    total = data.size
    if total == 0:
        return Clipped(math.nan, math.nan, 0, 0.0)
    centre = float(np.median(data))
    sigma = robust_sigma(data)
    kept = data
    for _ in range(iterations):
        if not sigma > 0:
            break
        mask = np.abs(data - centre) <= kappa * sigma
        kept = data[mask]
        new_centre, new_sigma = float(kept.mean()), float(kept.std(ddof=1))
        if math.isclose(new_sigma, sigma, rel_tol=1e-4) and math.isclose(
            new_centre, centre, rel_tol=0, abs_tol=1e-6 * max(1.0, abs(centre))
        ):
            centre, sigma = new_centre, new_sigma
            break
        centre, sigma = new_centre, new_sigma
    if kept.size < 2:
        return Clipped(centre, math.nan, int(kept.size), 1 - kept.size / total)
    return Clipped(
        float(kept.mean()), float(kept.std(ddof=1)), int(kept.size), 1 - kept.size / total
    )


def mean_and_sem(values: np.ndarray) -> tuple[float, float]:
    """Mean and its standard error. The error is NaN for fewer than two values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), math.nan
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def variance_rel_error(n_samples: int) -> float:
    """Relative standard error of a sample variance of normal data, sqrt(2 / (n - 1))."""
    return math.sqrt(2.0 / max(n_samples - 1, 1))


@dataclass(frozen=True)
class LineFit:
    slope: float
    intercept: float
    slope_err: float
    intercept_err: float
    r2: float
    n: int

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.slope * np.asarray(x) + self.intercept


def fit_line(x: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None) -> LineFit:
    """Least-squares straight line with parameter standard errors.

    With ``sigma`` the fit is weighted and the errors are scaled by the reduced chi-square (so
    they reflect the actual scatter, not just the stated errors).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    w = np.ones_like(x)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        good &= np.isfinite(sigma) & (sigma > 0)
        w = np.where(good, 1.0 / np.where(sigma > 0, sigma, 1.0) ** 2, 0.0)
    x, y, w = x[good], y[good], w[good]
    n = x.size
    if n < 2 or np.ptp(x) == 0:
        return LineFit(math.nan, math.nan, math.nan, math.nan, math.nan, n)
    design = np.column_stack([x, np.ones_like(x)])
    weighted = design * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(weighted, y * np.sqrt(w), rcond=None)
    resid = y - design @ coef
    dof = n - 2
    cov = np.linalg.inv(weighted.T @ weighted)
    cov = cov * float(np.sum(w * resid**2) / dof) if dof > 0 else cov * math.nan
    total = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1 - float(np.sum(w * resid**2)) / total if total > 0 else math.nan
    return LineFit(
        float(coef[0]),
        float(coef[1]),
        float(math.sqrt(cov[0, 0])),
        float(math.sqrt(cov[1, 1])),
        r2,
        n,
    )


def fit_proportional(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of ``y = k x`` through the origin and its standard error."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size == 0 or not np.any(x):
        return math.nan, math.nan
    k = float(np.sum(x * y) / np.sum(x * x))
    if x.size < 2:
        return k, math.nan
    resid = y - k * x
    err = math.sqrt(float(np.sum(resid**2)) / (x.size - 1) / float(np.sum(x * x)))
    return k, err


def bin_image(image: np.ndarray, factor: int, func: str = "mean") -> np.ndarray:
    """Downsample by an integer factor (edges that don't fill a block are dropped)."""
    if factor <= 1:
        return np.asarray(image, dtype=np.float32)
    h = image.shape[0] // factor * factor
    w = image.shape[1] // factor * factor
    blocks = image[:h, :w].reshape(h // factor, factor, w // factor, factor)
    if func == "median":
        return np.median(blocks.swapaxes(1, 2).reshape(h // factor, w // factor, -1), axis=2)
    return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def large_scale(image: np.ndarray, scale: int) -> np.ndarray:
    """A smooth, outlier-resistant background of ``image`` at roughly ``scale`` pixels.

    Median-bins by ``scale // 8``, median-filters over ~``scale`` (removing defects and features
    smaller than that), smooths lightly and interpolates back to full size. Equivalent to a
    full-resolution median filter of size ``scale`` on smooth backgrounds, at a fraction of the
    cost.
    """
    factor = max(1, scale // 8)
    h, w = image.shape
    small = bin_image(np.asarray(image, dtype=np.float32), factor, "median")
    size = max(3, min(2 * (scale // factor // 2) + 1, 2 * (min(small.shape) // 2) - 1))
    small = ndimage.median_filter(small, size=size, mode="nearest")
    small = ndimage.gaussian_filter(small, sigma=1.0, mode="nearest")
    zoom = (h / small.shape[0], w / small.shape[1])
    full = ndimage.zoom(small, zoom, order=1, mode="nearest", grid_mode=True)
    return _fit_shape(full, (h, w))


def polynomial_surface(
    image: np.ndarray, degree: int = 6, bins: int = 128, iterations: int = 5
) -> np.ndarray:
    """Robust 2-D Legendre polynomial fit of a smooth surface (e.g. vignetting).

    Fitted to a median-binned copy; points more than 3σ *below* the fit (dust shadows) or 5σ
    above are rejected iteratively, so local dips don't drag the surface down.
    """
    h, w = image.shape
    factor = max(1, max(h, w) // bins)
    small = bin_image(np.asarray(image, dtype=np.float32), factor, "median").astype(np.float64)
    sh, sw = small.shape
    yy, xx = np.mgrid[0:sh, 0:sw]
    u = (xx + 0.5) * factor / w * 2 - 1
    v = (yy + 0.5) * factor / h * 2 - 1
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    design = np.column_stack(
        [
            np.polynomial.legendre.legval(u.ravel(), np.eye(degree + 1)[i])
            * np.polynomial.legendre.legval(v.ravel(), np.eye(degree + 1)[j])
            for i, j in terms
        ]
    )
    target = small.ravel()
    keep = np.isfinite(target)
    coef = np.zeros(len(terms))
    for _ in range(iterations):
        coef, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
        resid = target - design @ coef
        sigma = robust_sigma(resid[keep]) or 1e-12
        new_keep = np.isfinite(target) & (resid > -3 * sigma) & (resid < 5 * sigma)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    fy = (np.arange(h) + 0.5) / h * 2 - 1
    fx = (np.arange(w) + 0.5) / w * 2 - 1
    out = np.zeros((h, w), dtype=np.float64)
    for c, (i, j) in zip(coef, terms, strict=True):
        px = np.polynomial.legendre.legval(fx, np.eye(degree + 1)[i])
        py = np.polynomial.legendre.legval(fy, np.eye(degree + 1)[j])
        out += c * np.outer(py, px)
    return out.astype(np.float32)


def _fit_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = image[: shape[0], : shape[1]]
    pad_h, pad_w = shape[0] - out.shape[0], shape[1] - out.shape[1]
    if pad_h or pad_w:
        out = np.pad(out, ((0, pad_h), (0, pad_w)), mode="edge")
    return out.astype(np.float32, copy=False)


def central_roi(shape: tuple[int, int], fraction: float = 0.5) -> tuple[slice, slice]:
    """The centred window covering ``fraction`` of each dimension."""
    h, w = shape
    dh, dw = int(h * (1 - fraction) / 2), int(w * (1 - fraction) / 2)
    return slice(dh, h - dh), slice(dw, w - dw)


def tiles(image: np.ndarray, size: int) -> np.ndarray:
    """View ``image`` as ``(rows, cols, size*size)`` non-overlapping tiles (copy if needed)."""
    h = image.shape[0] // size * size
    w = image.shape[1] // size * size
    blocks = image[:h, :w].reshape(h // size, size, w // size, size).swapaxes(1, 2)
    return blocks.reshape(h // size, w // size, size * size)
