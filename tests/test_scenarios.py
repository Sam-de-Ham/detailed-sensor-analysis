"""Data situations seen in real libraries: each must give an honest, correct result."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dsa.analysis.common import Level, Status
from tests.sessions import MONO, make_session


@pytest.fixture(scope="module")
def no_bias(tmp_path_factory):
    """A common astro-camera library: one long dark set and flats, no bias frames."""
    model = replace(MONO, rts_fraction=0, amp_glow=0)
    return make_session(
        tmp_path_factory.mktemp("nobias"),
        model,
        darks={300: 12},
        flats={"L": (1500, 10), "R": (1500, 10)},
    )


def test_no_bias_gives_bounds_not_values(no_bias):
    report = no_bias.mode
    m = no_bias.model
    assert report.offset.source == "short dark"
    # Gain still works: the 300 s dark is a good enough offset for bright flats.
    assert report.gain.gain.value == pytest.approx(m.file_gain, rel=0.01)
    rn = report.read_noise
    assert rn.status is Status.LIMITED and rn.combined_e.bound == "upper"
    dark = report.dark
    assert dark.current.bound == "upper"
    # The bounds hold and are not absurdly loose.
    true_rn = np.hypot(m.read_noise, m.row_noise)
    dark_e = float(np.median(no_bias.sensor.dark_rate(m.dark_ref_temp))) * 300
    assert true_rn <= rn.combined_e.value <= 1.1 * np.sqrt(true_rn**2 + dark_e)
    assert dark.current.value >= m.dark_current
    assert any("bias" in u.lower() for u in rn.unlock)


@pytest.fixture(scope="module")
def clamped(tmp_path_factory):
    """A DSLR that holds the dark level at the bias (black clamping)."""
    model = replace(
        MONO, clamp_dark=True, dark_current=0.3, rts_fraction=0, amp_glow=0, hot_fraction=0
    )
    return make_session(
        tmp_path_factory.mktemp("clamp"),
        model,
        bias=16,
        darks={40: 10, 300: 8},
        flats={"L": (1500, 10)},
    )


def test_clamped_dark_level_uses_noise(clamped):
    dark = clamped.mode.dark
    assert dark.clamped
    assert dark.estimator == "noise"
    truth = float(np.mean(clamped.sensor.dark_rate(clamped.model.dark_ref_temp)))
    assert dark.current.value == pytest.approx(truth, rel=0.08)


@pytest.fixture(scope="module")
def low_offset(tmp_path_factory):
    model = replace(MONO, offset=2, rts_fraction=0)
    return make_session(
        tmp_path_factory.mktemp("lowoffset"), model, bias=10, flats={"L": (1500, 6)}
    )


def test_offset_clipping_is_reported(low_offset):
    check = next(c for c in low_offset.mode.checks if c.key == "clipping")
    assert check.level is Level.PROBLEM


@pytest.fixture(scope="module")
def filtered(tmp_path_factory):
    model = replace(MONO, smoothing=0.5, rts_fraction=0)
    return make_session(tmp_path_factory.mktemp("filtered"), model, bias=10, flats={"L": (1500, 8)})


def test_in_camera_filtering_is_detected(filtered):
    check = next(c for c in filtered.mode.checks if c.key == "filtering")
    assert check.level is Level.WARN
    # Smoothing lowers variance, which inflates the photon-transfer gain: the reason to warn.
    assert filtered.mode.gain.gain.value > 1.2 * filtered.model.file_gain


@pytest.fixture(scope="module")
def ptc(tmp_path_factory):
    """A full photon-transfer series up to past full well (pixel-limited, not ADC-limited)."""
    model = replace(
        MONO,
        gain=10.0,
        full_well=30_000,
        read_noise=3.0,
        nonlinearity=0.02,
        rts_fraction=0,
        prnu=0.005,
    )
    levels = [
        500,
        1000,
        2000,
        4000,
        7000,
        10_000,
        14_000,
        18_000,
        22_000,
        26_000,
        30_000,
        34_000,
        38_000,
    ]
    series = tuple((lvl / 1000, lvl) for lvl in levels)
    return make_session(tmp_path_factory.mktemp("ptc"), model, bias=10, flat_series=series)


def test_multi_level_ptc(ptc):
    gain = ptc.mode.gain
    m = ptc.model
    assert gain.multi_level
    # Fitted over the linear range only; the planted 2 % compression biases it by ~1 %.
    assert gain.gain.value == pytest.approx(m.file_gain, rel=0.02)
    assert gain.tile_gain.value == pytest.approx(m.file_gain, rel=0.04)
    # Saturation at the photon-transfer turnover, near the full well (in file ADU).
    sat_e = gain.saturation_signal * m.file_gain
    assert sat_e == pytest.approx(m.full_well, rel=0.2)
    assert ptc.mode.dynamic_range.saturation_kind == "photon transfer"
    lin = gain.linearity[0]
    # 2 % compression at full well → about 1 % half-spread over 5-95 % of the range.
    assert 0.3 < lin.error_pct < 2.0
