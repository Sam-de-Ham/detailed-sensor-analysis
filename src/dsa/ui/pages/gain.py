"""Gain, photon transfer, saturation and dynamic range."""

from __future__ import annotations

import math

import numpy as np

from dsa.ui import theme
from dsa.ui.pages import Context, plane_label
from dsa.ui.plots import Plot, set_ticks
from dsa.ui.widgets import KeyFigure, Page, table


class GainPage(Page):
    title = "Gain & photon transfer"

    def build(self, ctx: Context) -> None:
        g, dr = ctx.mode.gain, ctx.mode.dynamic_range
        doc = self.new_document()
        doc.heading(
            "Gain & photon transfer",
            "How many electrons one ADC step stands for, found from the fact that light "
            "is shot-noise limited: the variance of S electrons is S.",
        )
        doc.status(g)
        native_note = (
            f"{g.native.format(4)} per native ADC step"
            if g.container_scale > 1 and g.gain.ok
            else "per ADC step"
        )
        keys = [
            (KeyFigure("Gain", g.gain, native_note), "sets"),
            (
                KeyFigure("Tile PTC slope", g.tile_gain, "cross-check over the vignetting range"),
                "ptc",
            ),
            (KeyFigure("Saturation", dr.saturation_e, dr.saturation_kind), "noise-model"),
            (
                KeyFigure(
                    "Dynamic range", dr.stops, f"{dr.db:.1f} dB" if math.isfinite(dr.db) else ""
                ),
                "noise-model",
            ),
        ]
        doc.keys(keys)
        if g.estimated_from:
            doc.text(g.method)

        if g.ptc_signal is not None or g.set_points is not None:
            doc.figure(
                "Photon transfer curve",
                "Temporal variance against signal, both above the offset (log–log). Shot "
                "noise gives a slope of 1; its height sets the gain. Points are 32 px tiles "
                "of each flat pair — vignetting spreads them over a range of signal — and "
                "the large points are the frame centres.",
                self._ptc(ctx),
                anchor="ptc",
            )
        if g.gain.ok:
            doc.figure(
                "Noise model in electrons",
                "Your sensor's noise budget built from the measurements: read noise is a "
                "floor, shot noise grows as √signal, pixel response non-uniformity (PRNU) "
                "grows linearly — but only in uncalibrated frames; flat-fielding removes it.",
                self._noise_model(ctx),
                anchor="noise-model",
            )
        if g.per_set:
            plot = Plot("Flat set", "Gain (e⁻/ADU)", height=240)
            x = np.arange(len(g.per_set))
            vals = [s.combined.value for s in g.per_set]
            errs = [s.combined.error for s in g.per_set]
            if g.gain.ok and not g.multi_level:
                plot.hline(g.gain.value, "combined")
                plot.hline(g.gain.value + g.gain.error, "+1σ")
                plot.hline(g.gain.value - g.gain.error, "−1σ")
            plot.points(
                x,
                vals,
                "per set",
                theme.ACCENT,
                size=8,
                tips=[f"{s.label}\n{s.combined.format()}" for s in g.per_set],
            )
            plot.errorbars(x, vals, errs)
            set_ticks(
                plot, "bottom", [(i, s.filter or str(i + 1)) for i, s in enumerate(g.per_set)]
            )
            doc.figure(
                "Agreement between flat sets",
                "Each set (often one per filter) is an independent measurement. They should "
                "agree within their error bars; the band is the combined value ± 1σ.",
                plot,
                anchor="sets",
            )
            rows = [
                (
                    s.label,
                    s.combined.format(),
                    f"{s.level_fraction:.0%}",
                    str(s.n_pairs),
                    f"{s.stability:.2%}",
                    f"{s.correlation:.3f}",
                    s.offset_source,
                )
                for s in g.per_set
            ]
            doc.add(
                table(
                    ["Set", "Gain (e⁻/ADU)", "Level", "Pairs", "Level drift", "|ρ|", "Offset from"],
                    rows,
                    numeric=(1, 2, 3, 4, 5),
                )
            )
        if len(g.planes) > 1 and g.gain_planes:
            rows = [
                (plane_label(n), v.format()) for n, v in zip(g.planes, g.gain_planes, strict=True)
            ]
            doc.figure(
                "Per colour plane",
                "Gain is a property of the pixel and readout, not the "
                "filter colour, so the planes should agree. A mismatch means the camera "
                "scales channels digitally.",
                table(["Plane", "Gain (e⁻/ADU)"], rows, numeric=(1,)),
            )
        for lin in g.linearity:
            plot = Plot("Exposure (s)", "Deviation from linear (%)", height=240)
            plot.points(lin.exposure, lin.deviation, lin.filter or "flats", theme.ACCENT, size=8)
            plot.hline(0)
            doc.figure(
                f"Linearity{' — ' + lin.filter if lin.filter else ''}",
                f"Signal vs exposure against a straight-line fit. Linearity error (EMVA, "
                f"5–95 % of saturation): ±{lin.error_pct:.2f} %.",
                plot,
            )
        doc.notes(g)
        if dr.method:
            doc.text(dr.method)
        doc.finish()

    def _ptc(self, ctx: Context) -> Plot:
        """The classic photon transfer curve: total temporal variance against signal."""
        g = ctx.mode.gain
        read_var = float(np.mean(g.read_variance)) if g.read_variance else 0.0
        plot = Plot(
            "Signal above offset (ADU)",
            "Temporal variance (ADU²)",
            log_x=True,
            log_y=True,
            height=380,
            legend_left=True,
        )
        if g.ptc_signal is not None:
            for k in np.unique(g.ptc_set):
                sel = g.ptc_set == k
                label = g.per_set[int(k)].filter or g.per_set[int(k)].label
                plot.points(
                    g.ptc_signal[sel],
                    g.ptc_variance[sel] + read_var,
                    label,
                    theme.SERIES[int(k) % len(theme.SERIES)],
                    size=3,
                    alpha=110,
                )
        if g.set_points is not None and len(g.set_points):
            pts = g.set_points
            tips = [
                f"{s.label}\nsignal {p[0]:.0f} ADU, variance {p[1] + read_var:.0f} ADU²"
                for s, p in zip(g.per_set, pts, strict=True)
            ]
            plot.points(
                pts[:, 0], pts[:, 1] + read_var, "frame centres", theme.TEXT, size=9, tips=tips
            )
        if g.gain.ok:
            top = ctx.mode.dynamic_range.saturation_adu
            top = top if math.isfinite(top) else float(np.nanmax(g.set_points[:, 0])) * 2
            xs = np.geomspace(1, top, 120)
            plot.line(
                xs,
                read_var + xs / g.gain.value,
                f"model: read² + S/g, g = {g.gain.format()}",
                theme.FAINT,
                dashed=True,
            )
            if read_var > 0:
                plot.hline(read_var, "read noise²")
        return plot

    def _noise_model(self, ctx: Context) -> Plot:
        r = ctx.mode
        g = r.gain.gain.value
        rn = r.read_noise.combined_e.value if r.read_noise.combined_e.ok else math.nan
        prnu = r.flat.prnu.value / 100 if r.flat.prnu.ok else math.nan
        sat = r.dynamic_range.saturation_e.value if r.dynamic_range.saturation_e.ok else 1e5
        s = np.geomspace(0.1, sat, 200)
        plot = Plot(
            "Signal (e⁻)", "Noise (e⁻)", log_x=True, log_y=True, height=340, legend_left=True
        )
        parts = [("shot √S", np.sqrt(s), theme.SERIES[1])]
        total = s.copy()
        if math.isfinite(rn):
            parts.append(("read noise", np.full_like(s, rn), theme.SERIES[2]))
            total += rn**2
        if math.isfinite(prnu):
            parts.append(("PRNU × S", prnu * s, theme.SERIES[3]))
        for name, y, colour in parts:
            plot.line(s, y, name, colour, 1.2, dashed=True)
        plot.line(s, np.sqrt(total), "total (flat-fielded)", theme.ACCENT, 2)
        if math.isfinite(prnu):
            plot.line(s, np.sqrt(total + (prnu * s) ** 2), "total (uncalibrated)", theme.TEXT, 1.2)
        if r.gain.set_points is not None:
            pts = r.gain.set_points
            plot.points(
                pts[:, 0] * g,
                np.sqrt(pts[:, 1] + (r.gain.read_variance[0] if r.gain.read_variance else 0)) * g,
                "measured flats",
                theme.TEXT,
                size=8,
            )
        if math.isfinite(rn):
            plot.vline(rn**2, "read = shot")
        return plot
