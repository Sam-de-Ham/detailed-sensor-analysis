# Validation

DSA is checked two ways: it must recover the parameters synthetic frames were generated with,
and on real frames it must broadly agree with published measurements of the same sensors.

## 1. Synthetic sensors (automated: `uv run pytest`)

`dsa.synthetic` generates frames from the standard linear camera model with known parameters:
Poisson photon and dark signal, Gaussian read noise, temporal row noise, row/column/pixel fixed
patterns, log-normal per-pixel dark current with hot pixels and amp glow, RTS pixels, PRNU,
vignetting, dust donuts, full-well and ADC clipping, a 12-bit ADC stored ×16, optional Bayer
mosaic, optional black-level clamping, in-camera smoothing and non-linearity. Frames are written
as NINA-style FITS and go through the whole pipeline (scan → measure → analyse).

| Quantity | Planted | Recovered (typical) | Test tolerance |
|---|---|---|---|
| Gain (12-bit ×16 mono) | 0.0625 e⁻/ADU | 0.0624 ± 0.0001 | 1 % |
| Tile-PTC gain | 0.0625 | 0.06246 | 1 % |
| Read noise (white + row) | 1.209 e⁻ | 1.209 ± 0.002 | 2 % |
| Temporal row noise | 0.15 e⁻ | ≈ 0.15 | 15 % |
| Dark current (slope, 3 temperatures) | true clipped mean | within 3 % | 3 % |
| Doubling temperature | 6.0 °C | 6.00 ± 0.002 | 3 % |
| Noise-based vs signal-based dark current | — | agree | 5 % |
| PRNU | 0.80 % | 0.800 % | 3 % |
| Vignetting at the corners | 65 % | 65.0 % | ±2 % |
| Dust mote (position / diameter) | (150, 120) / 56 px | ±1 px / ±5 % | 3 px / 15 % |
| Hot pixels | 55 planted | all found | ±5 %, ≥ 95 % overlap |
| RTS pixels (8 e⁻ steps, 20 frames) | 164 | 80–105 % found | — |
| Per-plane gain, Bayer 14-bit | 0.8 e⁻/ADU | each plane within 2 % | 2 % |
| Channel ratios R/G, B/G, G1/G2 | 0.55, 0.70, 1 | within 2 % | 2 % |
| Multi-level PTC gain (2 % non-linear) | 0.625 | 0.631 (bias from the non-linearity) | 2 % |
| Saturation at the PTC turnover | 30 000 e⁻ FW | 28 500 e⁻ | 20 % |
| Clamped dark level (noise method) | true mean | within 8 % | 8 % |
| No bias frames | — | read noise and dark current reported as upper limits that hold | — |
| Offset clipping, in-camera smoothing | — | flagged | — |

## 2. Real data

Frames from three cameras: a cooled ToupTek ATR585M (NINA FITS), a Sony A6400 (ARW) and a
Canon EOS 500D (CR2), as found in a real astrophotography library — no calibration frames were
taken specially. Published values and their sources: [research.md](research.md).
Gains are compared in native ADC units.

### ToupTek ATR585M (IMX585), gain 100, HCG, −5 °C

| Quantity | Measured | Published | Ratio |
|---|---|---|---|
| Gain (native 12-bit) | 1.0032 ± 0.0002 e⁻/ADU | 1.0 (ToupTek HCG chart) | 1.00 |
| Saturation (ADC clip) | 4079 e⁻ | ~4100 | 0.99 |
| Read noise | ≤ 1.74 e⁻ (no bias frames) | 0.79 | consistent |
| Dark current | ≤ 0.010 e⁻/s | 0.0044 at −5 °C (ZWO ASI585) | consistent |
| PRNU | 0.40 % | — | — |

Six flat sets (R, G, B, Ha, OIII, SII) give gains within 0.1 % of each other. Without bias
frames, read noise and dark current can only be bounded: the 300 s darks' temporal noise is
RN² + D·t ≈ 3.0 e⁻², and the published values (0.79² + 0.0044 × 300 ≈ 1.9 e⁻²) leave room for
RTS pixels and a slightly higher dark current. Amp glow is present but faint (+0.0017 e⁻/s at the
right edge, below a sub's noise), contradicting the "zero amp glow" marketing only mildly.
Vignetting shows an optical centre ~840 px off the sensor centre (130 mm Newtonian).

### Sony A6400 (lossy compressed raw)

| Setting | Quantity | Measured | photonstophotos | Ratio |
|---|---|---|---|---|
| ISO 400 | Gain | 0.591 e⁻/ADU | 0.57 | 1.04 |
| ISO 1600 | Gain | 0.151 | 0.144 | 1.05 |
| ISO 1600 | Read noise | 1.27 e⁻ | 1.05 | 1.21 |
| ISO 1600 | Saturation | 2394 e⁻ | 2280 | 1.05 |
| ISO 3200 | Read noise | 1.14 e⁻ (gain scaled from ISO 1600) | 1.03 | 1.10 |
| ISO 6400 | Read noise | 1.15 e⁻ (gain scaled) | 0.78 | 1.47 |

Discrepancies explained:
- **Read noise ISO 1600 (+21 %).** In DN the difference is 8.4 vs 7.3. Lossy cRAW adds little
  at bias level (code spacing 1 there); sample variation and temperature are plausible causes.
  Within the 25 % tolerance.
- **ISO 6400 (+47 %).** Claff's data drops from 1.03 e⁻ at ISO 3200 to 0.78 e⁻ at ISO 6400,
  inconsistent with an ISO-invariant sensor above its conversion-gain switch; he flags noise
  reduction from ISO 32000 and the step at 6400 is likely processing too. DSA's 1.27 → 1.14 →
  1.15 e⁻ continues the plateau, which is what the sensor architecture predicts.
- The long Sony darks (20 s, 30 s) show neighbour correlation 0.08: in-camera spatial filtering
  ("star eater"). Noise-based dark current there is a lower limit, and DSA says so.

### Canon EOS 500D

| Setting | Quantity | Measured | Published | Ratio |
|---|---|---|---|---|
| ISO 100 | Gain | 2.23 e⁻/ADU | 1.36 (Subaru page) | 1.64 |
| ISO 100 | Saturation | 27 700 e⁻ | 22 300 | 1.24 |
| ISO 3200 | Read noise | 2.39 e⁻ | 2.22 (extrapolated) | 1.08 |
| ISO 3200 | Dark current | 0.35 e⁻/s at 32 °C | — | — |

- **Gain at ISO 100.** The only published per-ISO data for the 500D (Subaru all-sky camera page)
  has gains exactly halving per stop, i.e. extrapolated from one measurement, and a saturation of
  16384 × gain that ignores the black level. DSA's 2.23 e⁻/ADU is cross-checked by the tile PTC
  (2.226) and gives a saturation of 27.7 ke⁻, in line with Canon's 4.7 µm sensors of that
  generation (photonstophotos: 450D 32.6 ke⁻, 550D 25.7 ke⁻). The scaled ISO 3200 gain
  (2.23 / 32 = 0.070) matches DSA's ISO 3200 tile PTC (0.0706).
- **Canon holds the dark level at the bias** (visible-area level is ~1024 ADU for 0, 40 and
  303 s; the masked border drifts instead). DSA detects this and measures dark current from its
  shot noise (EMVA 1288 §7.1): 0.34 and 0.35 e⁻/s at 31–32 °C, consistent between exposures.
- The ISO 3200 flats are at 1 % of full scale (~9 e⁻), so their gain (0.0745, pair method) is
  flagged as uncertain; PRNU and defects skip them.

## Known limitations

- Very faint, very large dust rings (filter dust, < 0.1 % deep, ~1000 px) are visible in the dust
  map but only the strongest are listed: the flat's own large-scale structure is of similar
  amplitude.
- Lossy raw (Sony cRAW) is corrected only for its decoded code spacing; block-coding artefacts
  near bright edges are not modelled. Uncompressed raw is recommended for characterisation.
- DSS master TIFFs are rescaled with the black point moved to 0: they are listed, not measured.
