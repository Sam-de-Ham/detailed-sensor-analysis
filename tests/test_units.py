"""Unit tests of the building blocks: CFA layout, statistics, metadata parsing, formats."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dsa.analysis.common import Value, format_value, weighted_mean
from dsa.analysis.exposure import noise_penalty, optimal_exposure
from dsa.analysis.levels import code_step, container_scale
from dsa.analysis.pixels import bimodality
from dsa.analysis.specs import find_spec, interpolate
from dsa.catalog import format_exposure
from dsa.cfa import CfaLayout
from dsa.frames import FrameType, classify_path, normalize_camera, parse_timestamp
from dsa.stats import clipped_stats, fit_line, large_scale, polynomial_surface, robust_sigma

# ---------------------------------------------------------------------------------- CFA --


def test_cfa_planes_and_merge_round_trip():
    layout = CfaLayout("RGGB")
    image = np.arange(36, dtype=np.float32).reshape(6, 6)
    planes = dict(layout.planes(image))
    assert list(planes) == ["R", "G1", "G2", "B"]
    assert planes["R"][0, 0] == image[0, 0]
    assert planes["G1"][0, 0] == image[0, 1]
    assert planes["G2"][0, 0] == image[1, 0]
    assert planes["B"][0, 0] == image[1, 1]
    assert np.array_equal(layout.merge(planes, image.shape), image)


@pytest.mark.parametrize(("pattern", "red"), [("BGGR", (1, 1)), ("GRBG", (0, 1)), ("GBRG", (1, 0))])
def test_cfa_red_position(pattern, red):
    layout = CfaLayout(pattern)
    assert layout.offsets["R"] == red
    # G1 shares the red row.
    assert layout.offsets["G1"][0] == red[0]


def test_cfa_vertical_flip():
    assert CfaLayout("RGGB").flipped_vertically(4).pattern == "GBRG"
    assert CfaLayout("RGGB").flipped_vertically(5).pattern == "RGGB"
    assert CfaLayout(None).names == ("L",)


# --------------------------------------------------------------------------- statistics --


def test_clipped_stats_ignores_outliers():
    rng = np.random.default_rng(0)
    data = rng.normal(100, 3, 200_000)
    data[:200] = 10_000  # hot pixels
    clip = clipped_stats(data)
    assert clip.mean == pytest.approx(100, abs=0.03)
    assert clip.std == pytest.approx(3, rel=0.01)
    assert clip.rejected == pytest.approx(0.001, abs=2e-4)


def test_robust_sigma_matches_normal():
    rng = np.random.default_rng(1)
    assert robust_sigma(rng.normal(0, 2, 100_000)) == pytest.approx(2, rel=0.02)


def test_fit_line_errors():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 200)
    y = 3 * x + 1 + rng.normal(0, 0.5, x.size)
    line = fit_line(x, y)
    assert line.slope == pytest.approx(3, abs=4 * line.slope_err)
    assert line.slope_err == pytest.approx(0.5 / (x.std() * math.sqrt(x.size)), rel=0.2)


def test_weighted_mean_inflates_error_on_disagreement():
    mean, err, chi2 = weighted_mean(np.array([1.0, 1.0]), np.array([0.1, 0.1]))
    assert (mean, chi2) == (1.0, 0.0)
    assert err == pytest.approx(0.1 / math.sqrt(2))
    _, err_bad, chi2_bad = weighted_mean(np.array([1.0, 2.0]), np.array([0.1, 0.1]))
    assert chi2_bad > 1 and err_bad > err


def test_polynomial_surface_ignores_dips():
    yy, xx = np.mgrid[0:300, 0:400]
    surface = 1 - 0.3 * (((xx - 200) / 250) ** 2 + ((yy - 150) / 250) ** 2)
    image = surface.copy()
    image[100:130, 100:130] *= 0.95  # a dust shadow must not pull the fit down
    fit = polynomial_surface(image.astype(np.float32), degree=4)
    assert np.max(np.abs(fit - surface)) < 0.003


def test_large_scale_removes_small_features():
    image = np.ones((256, 256), dtype=np.float32)
    image[100, 100] = 50  # a hot pixel
    assert np.max(np.abs(large_scale(image, 32) - 1)) < 1e-3


# ------------------------------------------------------------------------------- frames --


@pytest.mark.parametrize(
    ("path", "kind", "master"),
    [
        ("Calibration/a6400/Bias1600/DSC07588.ARW", FrameType.BIAS, False),
        ("x/biases/a.ARW", FrameType.BIAS, False),
        ("25.8.17.1/Lighs/DSC1.ARW", FrameType.LIGHT, False),
        ("x/Darks/MasterDark_ISO3200_40s.tif", FrameType.DARK, True),
        ("x/MasterOffset_ISO1600.tif", FrameType.BIAS, True),
        ("x/FLAT/Ha/f.fits", FrameType.FLAT, False),
        ("x/darkflats/f.fits", FrameType.DARK_FLAT, False),
        ("x/Pleiades/Raw/DSC.ARW", None, False),
    ],
)
def test_classify_path(path, kind, master):
    found, _, is_master = classify_path(Path(path))
    assert found is kind
    assert is_master is master


def test_metadata_normalisation():
    assert normalize_camera("ATR585M(USB2.0)") == "ATR585M"
    assert parse_timestamp("2026-08-23T13:34:51.9871226") is not None
    assert parse_timestamp("2009:01:04 06:55:36").year == 2009
    assert format_exposure(0.00025) == "1/4000 s"
    assert format_exposure(1.88438) == "1.88 s"
    assert format_exposure(300) == "300 s"


# ------------------------------------------------------------------------------ formats --


def test_container_scale_detection():
    rng = np.random.default_rng(3)
    codes = (rng.integers(0, 4096, 100_000) * 16).astype(np.uint16)
    hist = np.bincount(codes, minlength=65536)
    hist[12345] += 5  # a few defect-corrected pixels off the grid
    assert container_scale(hist) == 16
    assert container_scale(np.bincount(rng.integers(0, 16384, 100_000), minlength=65536)) == 1
    assert code_step(hist, 32000, 400) == 16


def test_value_formatting():
    assert format_value(1.23456, 0.012, "e⁻") == "1.235 ± 0.012 e⁻"
    assert format_value(1.23456, 0.05) == "1.23 ± 0.05"
    assert format_value(0.062697, 0.000010) == "0.062697 ± 0.000010"
    assert format_value(4079.08, 0.65) == "4079.1 ± 0.7"
    assert format_value(0.062697, 0.0000001) == "0.062697 ± 0.000001"
    assert Value(2.5, bound="upper", unit="e⁻").format() == "≤ 2.50 e⁻"
    assert format_value(math.nan) == "—"


# ---------------------------------------------------------------------------- formulas --


def test_glover_formula():
    # 5 % extra noise: t = R² / (P · 0.1025), the "C ≈ 10" rule.
    assert optimal_exposure(2.0, 1.0, 0.05) == pytest.approx(4 / 0.1025)
    t = optimal_exposure(1.5, 0.3, 0.05)
    assert noise_penalty(1.5, 0.3, t) == pytest.approx(0.05)


def test_bimodality_separates_telegraph_from_gaussian():
    rng = np.random.default_rng(4)
    gaussian = rng.normal(0, 1, (30, 500))
    telegraph = rng.normal(0, 1, (30, 500)) + 8 * (rng.random((30, 500)) < 0.5)
    sep_g, _, _ = bimodality(gaussian)
    sep_t, step, _ = bimodality(telegraph)
    assert np.percentile(sep_g, 99) < 4.0
    assert np.percentile(sep_t, 5) > 4.0
    assert np.median(step) == pytest.approx(8, rel=0.05)


def test_spec_lookup_and_interpolation():
    spec = find_spec("ATR585M")
    assert spec is not None and spec.curve_for("High Conversion Gain").gain[0] == 1.0
    assert find_spec("Sony ILCE-6400").name == "Sony A6400"
    value, extrapolated = interpolate((100, 200), (2.0, 1.0), 141.42)
    assert value == pytest.approx(math.sqrt(2), rel=1e-3) and not extrapolated
