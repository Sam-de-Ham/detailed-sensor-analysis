# Research reference: methods and published sensor data

Compiled 2026-11-01 from web research. **[vendor]** = manufacturer claim, **[indep]** = independent
measurement, **[derived]** = our arithmetic. ⚠ marks uncertain items. "e⁻/ADU" is the astro
convention for gain (EMVA 1288 uses the inverse, K in DN/e⁻).

## 1. Key takeaways for DSA

1. Measure everything; published numbers are only sanity checks. IMX585 vendors disagree
   (e.g. LCG read noise at minimum gain: ZWO 6.4 e⁻, ToupTek 2.74 e⁻).
2. ToupTek 12-bit modes store values ×16 in the FITS (multiples of 16, clip 65520). Vendor
   e⁻/ADU curves are in **native 12-bit ADU**; divide by 16 for file ADU. HDR mode is native 16-bit.
3. Quantisation can dominate read noise at low gain (ATR585M LCG gain 100: ~0.29 ADU₁₂).
   EMVA rule: if σ²_dark < 0.24 DN², report only an upper limit.
4. Sony A6400 switches conversion gain between ISO 320 and 400 (read noise 2.75 → 1.17 e⁻), and is
   ISO-invariant above that.
5. Canon 600D read noise falls steeply with ISO (14.7 e⁻ at ISO 100 → 3.1 e⁻ at 1600). The 500D is
   not in the photonstophotos database.
6. Sony bulb / long-exposure NR / continuous modes are 12-bit; timed exposures ≥ ~4 s may be
   spatially filtered ("star eater"); lossy cRAW corrupts variance statistics. Detect from data.

## 2. Published data

### 2.1 IMX585 (ToupTek ATR585M and siblings)

Sensor: Sony IMX585 (STARVIS 2, BSI), 2.9 µm pixels, 3840×2160 recorded (3856×2180 effective
array), 12-bit ADC (ToupTek HDR: 16-bit), QE peak ~91 % [vendor]. "Zero amp glow" claimed by ZWO
and ToupTek, no independent quantification found ⚠.

**ToupTek gain units:** percent, 100 = 1× = 0 dB (`dB = 20·log10(gain/100)`), range 100–15000.
ZWO uses 0.1 dB units: `ZWO = 200·log10(ToupTek/100)`. ToupTek has three explicit readout modes
(LCG, HCG, HDR) chosen in the driver (NINA `READOUTM`), each with its own curve; it does not switch
at a gain threshold.

ToupTek ATR585M vendor curves (read from the product page charts, −10 °C; ±5 %, FW ±10 %;
e⁻/ADU in native ADC units — 12-bit for LCG/HCG, 16-bit for HDR)
([product page](https://touptekastro.com/en-eu/products/atr585m)):

| Gain | dB | LCG e⁻/ADU | LCG RN e⁻ | LCG FW e⁻ | HCG e⁻/ADU | HCG RN e⁻ | HCG FW e⁻ | HDR e⁻/ADU₁₆ | HDR RN e⁻ | HDR FW e⁻ |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 0 | 9.5 | 2.74 | 38.9k | 1.0 | 0.79 | ~4.1k | 0.44 | 0.64 | ~28k |
| 177 | 5 | 5.45 | 3.04 | ~22k | 0.56 | 0.72 | ~2.3k | 0.25 | 0.55 | ~16.5k |
| 316 | 10 | 3.0 | 2.39 | ~12k | 0.31 | 0.59 | ~1.3k | 0.14 | 0.55 | ~9k |
| 562 | 15 | 1.72 | 2.40 | ~7k | 0.18 | 0.54 | ~740 | 0.08 | 0.54 | ~5.1k |
| 1000 | 20 | 0.95 | 2.21 | ~4k | 0.10 | 0.52 | ~420 | 0.04 | 0.51 | ~2.8k |
| 3162 | 30 | 0.30 | 1.91 | ~1.25k | 0.03 | 0.50 | ~135 | ~0.01 | 0.53 | ~0.9k |
| 10000 | 40 | 0.10 | 2.02 | ~400 | 0.01 | 0.54 | ~43 | ~0 | 0.55 | ~0.3k |

[derived] LCG FW at gain 100 = 9.5 × 4095 ≈ 38.9k: the quoted "full well" is the ADC-limited
saturation capacity. ToupTek's DR plot is capped at 12 stops (ADC bits), not FW/RN.

ZWO ASI585MM Pro [vendor] ([manual](https://i.zwoastro.com/wp-content/uploads/2025/09/04a912015d682f11887ca758deedb947.pdf)):
RN 0.7–6.4 e⁻, FW 38.7 ke⁻, unity gain 195, HCG automatically at gain ≥ 252.
Dark current vs temperature (labelled chart points, the best IMX585 dark current data available):

| T (°C) | −25 | −20 | −15 | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e⁻/s/px | 0.000386 | 0.000738 | 0.00126 | 0.00201 | 0.00435 | 0.00845 | 0.0162 | 0.0330 | 0.0656 | 0.1245 | 0.2495 | 0.512 |

[derived] doubling temperature ≈ 5 °C above −10 °C, ≈ 6–7 °C below.

Player One Uranus-M: RN 0.7–6.5 e⁻, FW ~47 ke⁻, HCG at gain 210 [vendor]. QHY5III585: spec sheets
conflict (FW 32–52 ke⁻) ⚠. Forum reports (AstroBin, SharpCap forum) show ATR585 HDR-mode
measurements varying with firmware ⚠.

**NINA FITS keywords** ([docs](https://nighttime-imaging.eu/docs/master/site/advanced/file_formats/fits/)):
`GAIN`/`OFFSET` are driver *settings*, not e⁻/ADU. `EGAIN` (driver `ElectronsPerADU`) is written
only when the driver reports it; units for ToupTek unclear ⚠ — treat as a hint. `READOUTM` carries
the readout mode. ToupTek 12-bit data are left-aligned in 16 bits (clip 65520)
([ToupTek.SDK](https://www.nuget.org/packages/ToupTek.SDK/)).

### 2.2 Canon EOS 600D / 500D (CR2, 14-bit)

600D, photonstophotos (Bill Claff) ([RN e⁻](https://www.photonstophotos.net/Charts/RN_e.htm),
[RN DN](https://www.photonstophotos.net/Charts/RN_ADU.htm), [PDR](https://www.photonstophotos.net/Charts/PDR.htm)),
FWC 26 796 e⁻, unity ISO ≈ 186:

| ISO | RN e⁻ | RN DN | Gain e⁻/DN [derived] | PDR EV |
|---|---|---|---|---|
| 100 | 14.72 | 7.84 | 1.88 | 8.46 |
| 200 | 7.84 | 8.40 | 0.93 | 8.21 |
| 400 | 4.96 | 10.63 | 0.47 | 7.66 |
| 800 | 3.63 | 15.56 | 0.23 | 6.97 |
| 1600 | 3.07 | 26.35 | 0.117 | 6.13 |
| 3200 | 2.81 | 48.17 | 0.058 | 5.21 |
| 6400 | 2.64 | 89.88 | 0.029 | 4.19 |

500D: **not** in photonstophotos. Only per-ISO data found: Subaru Telescope all-sky camera page
(O. Guyon) [indep] ([link](https://subarutelescope.org/staff/guyon/09allskysurvey.web/10detector.web/content.html)):

| ISO | RN ADU | Gain e⁻/ADU | RN e⁻ |
|---|---|---|---|
| 100 | 10.90 | 1.36 | 15.8 |
| 200 | 11.64 | 0.68 | 7.91 |
| 400 | 13.94 | 0.34 | 4.74 |
| 800 | 19.88 | 0.17 | 3.38 |
| 1600 | 32.27 | 0.085 | 2.74 |

⚠ Gains scale exactly 2× per stop (probably extrapolated from one measurement). Dark current upper
limit 0.0044 e⁻/s/px at 3 °C ("to be confirmed"). sensorgen.info is offline; no verified numbers.

Canon raw specifics: black ≈ 1024 (500D), ≈ 2048 (600D) ⚠ — read it per frame (LibRaw measures it
on the masked border). White level is ISO dependent. Canon keeps sub-black noise (no clipping).
Random horizontal (row) banding is characteristic of this generation. Expanded "H" ISOs are digital
pushes (comb/gap histograms) ([RawDigger](https://rawdigger.com/node/219)). CR2 is lossless.
MakerNote `CameraTemperature` is available.

### 2.3 Sony A6400 (ARW)

photonstophotos, FWC 36 443 e⁻, unity ISO ≈ 230:

| ISO | RN e⁻ | RN DN | Gain e⁻/DN [derived] | PDR EV |
|---|---|---|---|---|
| 100 | 3.25 | 1.41 | 2.30 | 10.51 |
| 200 | 2.91 | 2.53 | 1.15 | 9.79 |
| 320 | 2.75 | 3.81 | 0.72 | 9.16 |
| **400** | **1.17** | 2.04 | 0.57 | 9.25 |
| 800 | 1.09 | 3.78 | 0.29 | 8.17 |
| 1600 | 1.05 | 7.31 | 0.144 | 7.21 |
| 3200 | 1.03 | 14.32 | 0.072 | 6.17 |
| 6400 | 0.78 ⚠ | 21.71 | 0.036 | 5.03 |
| 12800 | 0.75 | 41.64 | 0.018 | 3.94 |

Black 512, white 16383 (14-bit), RGGB. 36443 / 2.297 ≈ 15 866 DN = 16383 − 512 + 5, consistent.
The ISO 6400 step may be extra processing ⚠.

Bit depth: 12-bit in bulb, long-exposure NR and continuous modes
([Sony support](https://www.sony.co.in/electronics/support/e-mount-body-ilce-6000-series/ilce-6400m/articles/00229990)).
**Compressed RAW is lossy**: 14→11-bit tone curve + 16-pixel blocks with 11-bit max/min and 7-bit
deltas ([lclevy/sony_raw](https://github.com/lclevy/sony_raw),
[Kasson](https://blog.kasson.com/?p=4823)). Variance-based statistics from cRAW are biased.
Star eater: spatial filtering from ~4 s in several bodies ([Kasson](https://blog.kasson.com/?p=16472));
no A6400-specific test found ⚠.

## 3. Methods

References: EMVA 1288 Release 4.0
([Linear](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf),
[General](https://emva.org/wp-content/uploads/EMVA1288General_4.0Release.pdf)); Janesick, *Photon
Transfer* (SPIE 2007); Mortara & Fowler (SPIE 290, 1981); Berry & Burnell, *HAIP*; Claff,
[Sensor Analysis Primer: Gain](https://photonstophotos.net/GeneralTopics/Sensors_&_Raw/Sensor_Analysis_Primer/Gain.htm).

**Gain (one flat pair + one bias pair):**
`g = ((μF1+μF2) − (μB1+μB2)) / (σ²(F1−F2) − σ²(B1−B2))`; read noise `= g·σ(B1−B2)/√2`.
Differencing removes all fixed patterns. EMVA eq. 18 corrects unequal means. Statistical error of
a variance: `σ(s²)/s² ≈ √(2/N)`. Systematics (nonlinearity, illumination drift, crosstalk, in-camera
filtering, lossy compression) usually dominate; smoothing *reduces* variance → gain too high. Flat
level ~30–60 % of clip. Without a bias pair, `g ≈ (μF − black)/(½σ²ΔF)` with relative bias
≈ RN²/(g·S). Per CFA plane, never demosaic (EMVA §6.6).

**Multi-level PTC (EMVA §6.6):** slope of σ²−σ²_dark vs μ−μ_dark over 0–70 % of saturation;
saturation where 0.1–0.2 % of pixels clip; linearity error from deviations of a linear fit.

**Read noise:** temporal only from ≥ 2 single frames. Quantisation correction √(σ² − 1/12) in native
units. A master bias cannot give read noise (its spatial σ² = s²_FPN + RN²/N).
EMVA spatial variance of an L-frame average: `s²_stack = s²_measured − σ²_temporal/L` (eq. 36).
Row/column/pixel decomposition (EMVA eqs. 37–42):
`s²_cav = var(col means) − σ²/(LM)`, `s²_rav = var(row means) − σ²/(LN)`,
`s²_col = s²_cav − s²_pix/M`, `s²_row = s²_rav − s²_pix/N`. Applied to difference frames it gives
temporal row/column noise (Canon banding).

**Dark current:** preferred = slope of mean dark vs exposure (EMVA §7.1); if the camera clamps
dark current, the mean is flat → use the variance slope `σ²_d = σ²_d0 + μ_I·t`. Single exposure:
`D ≈ g·(μ_dark − μ_bias)/t`. Temperature: `D = D_ref·2^((T−T_ref)/T_d)` or Arrhenius
`D ∝ T^{3/2}·exp(−E_a/kT)` (E_a ≈ 0.6 eV depletion, ≈ 1.1 eV diffusion). Hot pixels: EMVA gives no
fixed threshold; show log histograms and cumulative counts; default > median + 5σ_robust. Amp glow:
large-scale excess of the dark-current map; check its scaling with exposure. RTS: many frames,
per-pixel σ ≫ median, bimodal per-pixel histograms.

**PRNU (EMVA eq. 67):** `√(s²_y.50 − s²_y.dark)/(μ_y.50 − μ_y.dark)` on high-passed averaged flats,
with the temporal residual correction. **Vignetting:** radial model with free optical centre.
**Dust:** flat / smoothed flat, dips of 0.3–5 %; dust distance ≈ donut diameter × f-ratio.

**Pattern noise:** single bias = total, bias difference = temporal, master = fixed. 1-D FFTs of
row/column means and 2-D power spectra reveal periodic patterns (EMVA §8.6 spectrograms).

**Dynamic range:** `log2(μ_e.sat / RN_e)` with the ADC-limited saturation capacity
`g·(white − black)` when only one flat level exists.

**Optimal sub-exposure (Glover):** read noise adds a fraction x to sky-limited noise when
`t = C·R²/P`, `C = 1/((1+x)²−1)` (≈ 10 for 5 %, 25 for 2 %, 5 for 10 %); P = sky flux e⁻/px/s from
lights: `P = g·(sky − bias − dark)/t`.

**Data-integrity checks:** container scaling (values ≡ 0 mod 2^k), digital gain (histogram combs),
lossy compression (occupied-code spacing growing with signal), offset clipping (EMVA: < 0.5 % in
the zero bin), masters (float data, STACKCNT/NCOMBINE, noise ∝ 1/√N), in-camera spatial filtering
(neighbour correlation of difference frames, non-monotonic hot-pixel counts).

**Calibration quality:** dark scaling validity (predict D(t₂) from D(t₁)); header matching
(exposure, gain, offset, readout, binning, temperature, ISO); flat level 30–60 % of clip; bias
stability over time.
