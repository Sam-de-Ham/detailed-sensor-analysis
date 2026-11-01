"""Read noise and pattern noise pages."""

from __future__ import annotations

import math

import numpy as np

from dsa.analysis.common import Value
from dsa.display import Stretch
from dsa.ui import theme
from dsa.ui.imageview import SensorMap
from dsa.ui.pages import Context, plane_label
from dsa.ui.plots import Plot
from dsa.ui.widgets import KeyFigure, Page, link_button, table


class ReadNoisePage(Page):
    title = "Read noise"

    def build(self, ctx: Context) -> None:
        rn = ctx.mode.read_noise
        doc = self.new_document()
        doc.heading(
            "Read noise",
            "The noise of the readout itself: what every sub pays once, "
            "whatever its exposure. Measured from the difference of bias frames, which "
            "cancels every fixed pattern.",
        )
        doc.status(rn)
        if not rn.adu:
            doc.notes(rn)
            doc.finish()
            return
        unit = ctx.unit
        main = rn.combined_e if rn.combined_e.ok else rn.combined_adu
        median = (
            (
                np.mean(rn.median_pixel_adu) * ctx.gain
                if ctx.electrons
                else np.mean(rn.median_pixel_adu)
            )
            if rn.median_pixel_adu
            else math.nan
        )
        keys = [
            (
                KeyFigure(
                    "Read noise", main, "rms over pixels" + (", upper limit" if main.bound else "")
                ),
                "pairs",
            ),
            (KeyFigure("In ADU", rn.combined_adu, "file units"), "pairs"),
            (
                KeyFigure(
                    "Median pixel",
                    Value(median, unit=unit),
                    "typical pixel (vendors often quote this)",
                ),
                "hist",
            ),
            (
                KeyFigure(
                    "Quantisation", Value(rn.quantisation_adu, unit="ADU"), "removed in quadrature"
                ),
                None,
            ),
        ]
        doc.keys(keys)
        source = link_button(f"{rn.set_label} →", lambda: ctx.open_set(rn.set_label))
        doc.add(source)

        if len(rn.planes) > 1:
            rows = []
            for i, name in enumerate(rn.planes):
                e = ctx.to_e(rn.adu[i])
                rows.append(
                    (
                        plane_label(name),
                        rn.adu[i].format(),
                        e.format() if e is not rn.adu[i] else "—",
                        f"{rn.median_pixel_adu[i]:.3g}" if rn.median_pixel_adu else "—",
                    )
                )
            doc.figure(
                "Per colour plane",
                "The four photosites of the Bayer cell are read through "
                "the same circuit, so they should agree.",
                table(
                    ["Plane", "ADU", unit if ctx.electrons else "e⁻", "Median pixel (ADU)"],
                    rows,
                    numeric=(1, 2, 3),
                ),
            )

        if rn.per_pair is not None and rn.per_pair.shape[0] > 1:
            plot = Plot("Bias pair", f"Read noise ({unit})", height=240)
            factor = ctx.gain if ctx.electrons else 1.0
            x = np.arange(1, rn.per_pair.shape[0] + 1)
            for i, name in enumerate(rn.planes):
                plot.points(
                    x,
                    rn.per_pair[:, i] * factor,
                    plane_label(name),
                    theme.PLANE_COLOURS.get(name, theme.ACCENT),
                    size=7,
                )
            doc.figure(
                "Stability across the set",
                "One estimate per independent pair of frames, "
                "in capture order. A trend means the bias drifted (temperature, warm-up).",
                plot,
                anchor="pairs",
            )

        if rn.sigma_hist:
            plot = Plot(f"Temporal noise of a pixel ({unit})", "Pixels", log_y=True, height=280)
            factor = ctx.gain if ctx.electrons else 1.0
            for (edges, counts), name in zip(rn.sigma_hist, rn.planes, strict=True):
                centres = (edges[:-1] + edges[1:]) / 2 * factor
                keep = counts > 0
                plot.line(
                    centres[keep],
                    counts[keep],
                    plane_label(name),
                    theme.PLANE_COLOURS.get(name, theme.ACCENT),
                    1.4,
                )
            if main.ok:
                plot.vline(main.value, "rms")
            doc.figure(
                "Distribution over pixels",
                "Each pixel's own noise from its scatter over the frames. The long tail to "
                "the right is random-telegraph and other noisy pixels: rare, but they set the "
                "look of short-exposure images.",
                plot,
                anchor="hist",
            )
        if rn.sigma_map is not None:
            data = rn.sigma_map * (ctx.gain if ctx.electrons else 1.0)
            doc.figure(
                "Noise map",
                "Per-pixel temporal noise. Bright dots are noisy (often RTS) "
                "pixels; structure would point to readout-circuit differences.",
                SensorMap(
                    data,
                    unit,
                    cfa=ctx.mode.mode.cfa,
                    stretch=Stretch.PERCENTILE,
                    title="Read noise map",
                ),
            )
        doc.notes(rn)
        doc.finish()


class PatternNoisePage(Page):
    title = "Pattern noise"

    def build(self, ctx: Context) -> None:
        pat = ctx.mode.pattern
        doc = self.new_document()
        doc.heading(
            "Pattern noise",
            "Noise shared by whole rows or columns. Temporal patterns "
            "change every frame (banding you can only average away); fixed patterns repeat "
            "and are removed by bias and dark subtraction.",
        )
        doc.status(pat)
        unit = ctx.unit
        conv = lambda v: ctx.to_e(v)  # noqa: E731

        def mean(values: list[Value]) -> Value:
            ok = [v for v in values if v.ok]
            if not ok:
                return Value(math.nan)
            return conv(Value(float(np.sqrt(np.mean([v.value**2 for v in ok]))), unit="ADU"))

        keys = []
        if pat.row_temporal:
            ratio = pat.banding_ratio
            keys += [
                (
                    KeyFigure(
                        "Row noise, temporal",
                        mean(pat.row_temporal),
                        f"{ratio:.0%} of pixel noise" if math.isfinite(ratio) else "",
                    ),
                    "profiles",
                ),
                (KeyFigure("Column noise, temporal", mean(pat.col_temporal)), "profiles"),
            ]
        if pat.row_fixed:
            keys += [
                (KeyFigure("Row pattern, fixed", mean(pat.row_fixed)), "fixed"),
                (KeyFigure("Column pattern, fixed", mean(pat.col_fixed)), "fixed"),
                (
                    KeyFigure("Pixel offset spread (DSNU)", mean(pat.pixel_fixed), "high-pass"),
                    "fixed",
                ),
            ]
        if keys:
            doc.keys(keys)

        def get(values: list[Value], i: int) -> str:
            return conv(values[i]).format() if i < len(values) and values[i].ok else "—"

        rows = []
        for i, name in enumerate(pat.planes):
            rows.append(
                (
                    plane_label(name),
                    get(pat.pixel_temporal, i),
                    get(pat.row_temporal, i),
                    get(pat.col_temporal, i),
                    get(pat.pixel_fixed, i),
                    get(pat.row_fixed, i),
                    get(pat.col_fixed, i),
                )
            )
        if rows:
            doc.figure(
                "Decomposition",
                f"EMVA 1288 split of the noise into pixel, row and "
                f"column parts, in {unit}. Temporal parts add to the read noise of every sub.",
                table(
                    [
                        "Plane",
                        "Pixel (temp.)",
                        "Row (temp.)",
                        "Column (temp.)",
                        "Pixel (fixed)",
                        "Row (fixed)",
                        "Column (fixed)",
                    ],
                    rows,
                    numeric=range(1, 7),
                ),
            )

        if pat.row_deviation is not None:
            data = pat.row_deviation * (ctx.gain if ctx.electrons else 1)
            doc.figure(
                "Row banding, frame by frame",
                "Each line of this image is one frame, each column one sensor row: the "
                "deviation of that row's mean from its average. Horizontal banding shows "
                "as vertical streaks.",
                SensorMap(
                    data,
                    unit,
                    stretch=Stretch.SYMMETRIC,
                    colour_map="Diverging",
                    height=300,
                    title="Row deviations",
                    square_pixels=False,
                ),
                anchor="profiles",
            )
        if pat.row_profile is not None:
            plot = Plot("Row", "Mean level (ADU)", height=220)
            plot.line(np.arange(pat.row_profile.size), pat.row_profile, "rows", theme.ACCENT, 1)
            plot2 = Plot("Column", "Mean level (ADU)", height=220)
            plot2.line(
                np.arange(pat.col_profile.size), pat.col_profile, "columns", theme.SERIES[1], 1
            )
            doc.figure(
                "Fixed row profile",
                f"Mean of each row of the averaged frame ({pat.fixed_label}), first colour plane.",
                plot,
                anchor="fixed",
            )
            doc.figure("Fixed column profile", "Mean of each column.", plot2)
        if pat.psd:
            plot = Plot(
                "Spatial frequency (cycles per plane pixel)", "Power (ADU²)", log_y=True, height=280
            )
            for k, (name, (freq, power)) in enumerate(pat.psd.items()):
                plot.line(freq, power, name, theme.SERIES[k % len(theme.SERIES)], 1.2)
            doc.figure(
                "Spectra of row and column means",
                "Periodic patterns show as peaks; a flat spectrum is random noise.",
                plot,
            )
        if pat.peaks:
            doc.figure(
                "Periodic patterns found",
                "Peaks ≥ 25× the local spectrum level.",
                table(
                    ["Axis", "Kind", "Period (px)", "Amplitude (ADU)"],
                    [(p.axis, p.kind, f"{p.period:.1f}", f"{p.amplitude:.3g}") for p in pat.peaks],
                    numeric=(2, 3),
                ),
            )
        if pat.spectrum2d is not None:
            doc.figure(
                "2-D power spectrum of the fixed pattern",
                "Centre = large scales. A bright horizontal or vertical line through the "
                "centre is row or column structure; isolated spots are periodic patterns.",
                SensorMap(
                    pat.spectrum2d,
                    "log₁₀ ADU²",
                    stretch=Stretch.PERCENTILE,
                    colour_map="Grey",
                    height=380,
                    title="2D spectrum",
                ),
            )
        doc.notes(pat)
        doc.finish()
