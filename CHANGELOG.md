# Changelog

## 1.0.0 — 2026-11-01

First release.

- Library scan of any folder tree: FITS (NINA, SharpCap, Siril masters), camera raw through
  LibRaw (CR2, CR3, ARW, NEF, DNG, …) and DeepSkyStacker master TIFFs. Frames are classified
  from headers or folder and file names, grouped by camera and setting, and duplicate copies
  and processing output are skipped. Files are only read.
- Analyses, per colour plane: read noise and its distribution, pattern noise (EMVA 1288
  row/column/pixel split, spectra), gain and photon transfer, saturation, linearity and dynamic
  range, dark current (slope, level and shot-noise methods, temperature fit), amp glow,
  defective and RTS pixels, PRNU, vignetting, dust, optimal sub-exposure, data and calibration
  checks, comparison with published data.
- Handles the traps of real data: 12-bit data stored in 16 bits, black-level clamping, lossy
  compressed raw, in-camera filtering, offset clipping, masters vs subs, missing bias frames.
- Desktop app (PyQt6): one page per analysis with key numbers, interactive plots and sensor
  maps that zoom to single pixels; exports to HTML report, JSON, CSV, PNG and FITS.
- Command line: `dsa scan` and `dsa analyze`.
- Tests that recover known parameters from synthetic frames, plus the validation on real
  cameras in `docs/validation.md`.
