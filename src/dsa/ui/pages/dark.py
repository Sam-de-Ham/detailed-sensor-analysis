"""Dark current: level, exposure and temperature dependence, per-pixel map and amp glow."""

from __future__ import annotations

import math

import numpy as np

from dsa.analysis.dark import summarise_glow
from dsa.analysis.specs import find_spec
from dsa.display import Stretch
from dsa.ui import theme
from dsa.ui.imageview import Overlay, SensorMap
from dsa.ui.pages import Context
from dsa.ui.plots import Plot
from dsa.ui.widgets import KeyFigure, Page, link_button, table


class DarkPage(Page):
    title = "Dark current"

    def build(self, ctx: Context) -> None:
        dk = ctx.mode.dark
        doc = self.new_document()
        doc.heading(
            "Dark current",
            "Electrons the sensor generates without light. It adds a "
            "signal (removed by darks) and shot noise (which no calibration removes), "
            "and roughly doubles every few degrees.",
        )
        doc.status(dk)
        unit = dk.unit
        temp = dk.reference_temperature
        keys = [
            (
                KeyFigure(
                    "Dark current",
                    dk.current,
                    (f"at {temp:.0f} °C, " if math.isfinite(temp) else "")
                    + f"{dk.estimator} method",
                ),
                "exposure",
            ),
            (
                KeyFigure(
                    "Doubling temperature",
                    dk.doubling,
                    "°C per doubling" if dk.doubling.ok else "needs darks at several temperatures",
                ),
                "temperature",
            ),
            (
                KeyFigure(
                    "Activation energy",
                    dk.activation,
                    "Arrhenius fit" if dk.activation.ok else "needs darks at several temperatures",
                ),
                "temperature",
            ),
            (
                KeyFigure(
                    "Amp glow",
                    dk.glow.location if dk.glow else "—",
                    f"+{dk.glow.peak:.2g} {unit}" if dk.glow and dk.glow.location != "none" else "",
                ),
                "glow",
            ),
        ]
        if dk.read_noise_bound.ok:
            keys.append(
                (KeyFigure("Read noise bound", dk.read_noise_bound, "from the same darks"), None)
            )
        doc.keys(keys)

        if dk.points:
            rows = [
                (
                    p.label,
                    f"{p.exposure:g} s",
                    f"{p.temperature:.1f} °C" if math.isfinite(p.temperature) else "—",
                    p.current.format() if p.current.ok else "—",
                    p.noise_current.format() if p.noise_current.ok else "—",
                )
                for p in dk.points
            ]
            doc.figure(
                "Dark sets",
                "Dark current from each set: from its mean level above the "
                "offset, and independently from the shot noise it adds.",
                table(
                    ["Set", "Exposure", "Temperature", "From level", "From noise"],
                    rows,
                    numeric=(1, 2, 3, 4),
                ),
            )
            for p in dk.points:
                doc.add(link_button(f"{p.label} →", lambda lbl=p.label: ctx.open_set(lbl)))
        if dk.slope_fits:
            plot = Plot("Exposure (s)", "Dark signal above offset (ADU)", height=280)
            for k, fit in enumerate(dk.slope_fits):
                colour = theme.SERIES[k % len(theme.SERIES)]
                name = (
                    f"{fit['temperature']:.0f} °C" if math.isfinite(fit["temperature"]) else "darks"
                )
                plot.points(fit["exposure"], fit["signal"], name, colour, size=8)
                plot.errorbars(fit["exposure"], fit["signal"], fit["error"], colour)
                xs = np.linspace(0, max(fit["exposure"]) * 1.05, 20)
                plot.line(xs, fit["slope"] * xs + fit["intercept"], None, colour, 1, dashed=True)
            doc.figure(
                "Signal vs exposure",
                "The slope is the dark current; the intercept shows "
                "whether a bias equals a zero-length dark.",
                plot,
                anchor="exposure",
            )
        if dk.temp_points is not None:
            doc.figure(
                "Temperature dependence",
                "Dark current against sensor temperature (log "
                "scale): a straight line means exponential growth. Dashed: published curve.",
                self._temperature(ctx),
                anchor="temperature",
            )
        if dk.map is not None:
            defects = ctx.mode.defects
            overlays = []
            hot = defects.get("hot")
            if hot is not None and hot.count:
                overlays.append(Overlay("hot", hot.xs[:5000], hot.ys[:5000], theme.PROBLEM, 8))
            caption = (
                "Dark current of every pixel"
                if dk.map_absolute
                else "Excess dark current over the median (no bias, so not absolute)"
            )
            doc.figure(
                "Dark current map",
                f"{caption}, from {dk.map_label}. Circles: hot pixels (up to 5000 shown).",
                SensorMap(
                    dk.map,
                    unit,
                    cfa=ctx.mode.mode.cfa,
                    stretch=Stretch.PERCENTILE,
                    title="Dark current map",
                    overlays=overlays,
                ),
                anchor="map",
            )
        if dk.hist is not None:
            edges, counts = dk.hist
            centres = (edges[:-1] + edges[1:]) / 2
            plot = Plot(f"Dark current ({unit})", "Pixels", log_y=True, height=260)
            keep = counts > 0
            plot.line(centres[keep], counts[keep], "pixels", theme.ACCENT, 1.4)
            if math.isfinite(dk.median_pixel):
                plot.vline(dk.median_pixel, "median")
            doc.figure(
                "Distribution over pixels",
                "Most pixels sit in the peak; the tail to the "
                "right is warm and hot pixels (log scale).",
                plot,
            )
        if dk.glow is not None:
            g = dk.glow
            doc.figure(
                "Amp glow",
                summarise_glow(g, unit) + " Large-scale excess dark current "
                "(defects removed), relative to the centre of the sensor.",
                SensorMap(
                    g.map,
                    unit,
                    scale=g.bin_factor,
                    stretch=Stretch.MINMAX,
                    colour_map="Inferno",
                    height=360,
                    title="Amp glow",
                ),
                anchor="glow",
            )
            if len(g.scaling) >= 2:
                rows = [(f"{t:g} s", f"{p:.3g}", f"{p / t:.3g}") for t, p in g.scaling]
                doc.add(
                    table(["Exposure", "Peak glow (ADU)", "Per second"], rows, numeric=(0, 1, 2))
                )
        doc.notes(dk)
        doc.finish()

    def _temperature(self, ctx: Context) -> Plot:
        dk = ctx.mode.dark
        pts = dk.temp_points
        plot = Plot(
            "Sensor temperature (°C)", f"Dark current ({dk.unit}/px)", log_y=True, height=300
        )
        plot.points(pts[:, 0], pts[:, 1], "measured", theme.ACCENT, size=9)
        plot.errorbars(pts[:, 0], pts[:, 1], pts[:, 2])
        if dk.doubling.ok:
            t = np.linspace(pts[:, 0].min() - 3, pts[:, 0].max() + 3, 40)
            ref = pts[np.argmax(pts[:, 1])]
            plot.line(
                t,
                ref[1] * 2 ** ((t - ref[0]) / dk.doubling.value),
                f"fit: doubles every {dk.doubling.value:.1f} °C",
                theme.ACCENT,
                1,
                dashed=True,
            )
        spec = find_spec(ctx.camera.camera)
        if spec and len(spec.dark_temps) > 1:
            plot.line(spec.dark_temps, spec.dark_current, "published", theme.FAINT, dashed=True)
        return plot
