"""End-to-end accuracy: the pipeline must recover the parameters synthetic frames were made with.

Tolerances are a few times the statistical uncertainty the pipeline itself reports, so a test
fails on bias, not on noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsa.analysis.common import Status
from dsa.stats import central_roi, clipped_stats


def test_detects_container_scaling(mono):
    fmt = mono.mode.format
    assert fmt.container_scale == 16
    assert round(fmt.adc_bits) == 12


def test_gain(mono):
    gain = mono.mode.gain
    assert gain.status is Status.MEASURED
    assert gain.gain.value == pytest.approx(mono.model.file_gain, rel=0.01)
    assert gain.native.value == pytest.approx(mono.model.gain, rel=0.01)
    assert gain.tile_gain.value == pytest.approx(mono.model.file_gain, rel=0.01)
    assert len(gain.per_set) == 2  # two filters = two independent estimates


def test_read_noise(mono):
    rn = mono.mode.read_noise
    m = mono.model
    # Temporal noise at zero exposure: white read noise plus temporal row noise.
    truth = np.hypot(m.read_noise, m.row_noise)
    assert rn.status is Status.MEASURED
    assert rn.combined_e.value == pytest.approx(truth, rel=0.02)
    assert rn.median_pixel_adu[0] * m.file_gain == pytest.approx(m.read_noise, rel=0.05)


def test_row_noise(mono):
    pattern = mono.mode.pattern
    m = mono.model
    assert pattern.row_temporal[0].value * m.file_gain == pytest.approx(m.row_noise, rel=0.15)
    assert pattern.col_temporal[0].value * m.file_gain < 0.05


def _true_dark(session, temperature):
    rate = session.sensor.dark_rate(temperature)
    return clipped_stats(rate[central_roi(rate.shape)][::2, ::2]).mean


def test_dark_current_and_temperature(mono):
    dark = mono.mode.dark
    assert dark.status is Status.MEASURED
    assert dark.estimator == "slope"
    for fit in dark.slope_fits:
        truth = _true_dark(mono, fit["temperature"])
        assert fit["current"].value == pytest.approx(truth, rel=0.03)
    assert dark.doubling.value == pytest.approx(mono.model.dark_doubling, rel=0.03)
    # Noise-based estimate agrees with the signal-based one.
    for point in dark.points:
        assert point.noise_current.value == pytest.approx(point.current.value, rel=0.05)


def test_amp_glow_located(mono):
    glow = mono.mode.dark.glow
    assert glow.location == "right edge"
    assert glow.peak == pytest.approx(mono.model.amp_glow, rel=0.5)


def test_prnu_vignetting_dust(mono):
    flat = mono.mode.flat
    m = mono.model
    assert flat.prnu.value == pytest.approx(m.prnu * 100, rel=0.03)
    vig = flat.sets[0].vignetting
    for value in vig.corners.values():
        assert value == pytest.approx(1 - m.vignetting, abs=0.02)
    assert abs(vig.centre_offset[0]) < 10 and abs(vig.centre_offset[1]) < 10
    assert len(flat.dust) == 1
    mote, truth = flat.dust[0], m.dust[0]
    assert (mote.x, mote.y) == pytest.approx((truth.x, truth.y), abs=3)
    assert mote.diameter == pytest.approx(2 * truth.radius, rel=0.15)
    assert mote.donut
    assert sorted(mote.filters) == ["Ha", "L"]


def test_defects(mono):
    defects = mono.mode.defects
    hot = defects.get("hot")
    truth = int(mono.sensor.hot.sum())
    assert hot.count == pytest.approx(truth, abs=max(2, 0.05 * truth))
    found = set(zip(hot.ys.tolist(), hot.xs.tolist(), strict=True))
    planted = set(zip(*np.nonzero(mono.sensor.hot), strict=True))
    assert len(found & planted) >= 0.95 * truth
    rts = defects.rts
    planted_rts = int(mono.sensor.rts.sum())
    assert 0.8 * planted_rts <= rts.count <= 1.05 * planted_rts
    assert np.median(rts.amplitudes) == pytest.approx(mono.model.rts_amplitude, rel=0.1)


def test_sub_exposure(mono):
    exposure = mono.mode.exposure
    assert exposure.status is Status.MEASURED
    point = exposure.points[0]
    # Sky flux at the centre is the sky rate times the (vignetted) flat field there.
    ff = mono.sensor.flat_field
    truth = 0.8 * float(np.median(ff[central_roi(ff.shape)]))
    assert point.flux.value == pytest.approx(truth, rel=0.03)


def test_dynamic_range(mono):
    dr = mono.mode.dynamic_range
    m = mono.model
    sat = (m.white_level - m.file_offset) * m.file_gain
    assert dr.saturation_kind == "ADC clip"
    assert dr.saturation_e.value == pytest.approx(sat, rel=0.02)


def test_checks_quiet_on_clean_data(mono):
    from dsa.analysis.common import Level

    noisy = [c for c in mono.mode.checks if c.level in (Level.WARN, Level.PROBLEM)]
    assert noisy == []


# --------------------------------------------------------------------------------- colour --


def test_colour_per_plane(colour):
    report = colour.mode
    m = colour.model
    assert report.format.container_scale == 1
    assert report.gain.planes == ["R", "G1", "G2", "B"]
    for g in report.gain.gain_planes:
        assert g.value == pytest.approx(m.gain, rel=0.02)
    for rn in report.read_noise.adu:
        assert rn.value * m.gain == pytest.approx(m.read_noise, rel=0.03)
    ratios = report.flat.sets[0].channel_ratios
    expected = m.channel_response
    assert ratios["R/G"] == pytest.approx(expected["R"], rel=0.02)
    assert ratios["B/G"] == pytest.approx(expected["B"], rel=0.02)
    assert ratios["G1/G2"] == pytest.approx(1, abs=0.01)
    assert report.flat.prnu.value == pytest.approx(m.prnu * 100, rel=0.05)
