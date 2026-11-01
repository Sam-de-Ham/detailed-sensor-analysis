"""Sub-exposure: how long subs must be for read noise not to matter under your sky."""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtWidgets import QDoubleSpinBox, QFormLayout, QLabel, QWidget

from dsa.analysis.exposure import PENALTIES, noise_penalty, optimal_exposure
from dsa.ui import theme
from dsa.ui.pages import Context
from dsa.ui.plots import Plot
from dsa.ui.widgets import KeyFigure, Page, table


class ExposurePage(Page):
    title = "Sub-exposure"

    def build(self, ctx: Context) -> None:
        ex = ctx.mode.exposure
        doc = self.new_document()
        doc.heading(
            "Sub-exposure",
            "Every sub adds one read noise. Once the sky's own shot noise "
            "dwarfs it, longer subs gain almost nothing — and risk more trailing, "
            "satellites and clipped stars.",
        )
        doc.status(ex)
        keys = []
        for p in ex.points:
            t5 = p.optimal.get(0.05, math.nan)
            note = f"read noise adds {p.penalty:.0%} at your {p.exposure:g} s"
            keys.append((KeyFigure(f"Sky, {p.filter or p.label}", p.flux, note), "curves"))
            text = f"{t5:.0f} s" if math.isfinite(t5) else "—"
            sub_note = "5 % read-noise penalty"
            if p.optimal_assumed:
                sub_note += f"; {p.optimal_assumed[0.05]:.0f} s with published read noise"
            keys.append(
                (
                    KeyFigure(
                        f"Shortest efficient sub, {p.filter or 'lights'}",
                        ("≤ " if ex.read_noise_bound else "") + text,
                        sub_note,
                    ),
                    "curves",
                )
            )
        if keys:
            doc.keys(keys)
        if ex.points:
            doc.figure(
                "Noise penalty vs sub length",
                "Extra noise compared with one infinitely long exposure of the same total "
                "time (Glover). Dots: the sub length you used.",
                self._curves(ctx),
                anchor="curves",
            )
            rows = []
            for p in ex.points:
                rows.append(
                    (
                        p.label,
                        p.flux.format(),
                        f"{p.exposure:g} s",
                        f"{p.penalty:.1%}",
                        f"{p.swamp:.1f}",
                        *(f"{p.optimal[x]:.0f} s" for x in PENALTIES),
                        f"{p.clipped:.3%}",
                    )
                )
            doc.add(
                table(
                    [
                        "Lights",
                        "Sky flux",
                        "Used",
                        "Penalty",
                        "Sky e⁻ / RN²",
                        "2 %",
                        "5 %",
                        "10 %",
                        "Clipped",
                    ],
                    rows,
                    numeric=range(1, 9),
                )
            )
        if math.isfinite(ex.read_noise_e):
            doc.figure(
                "Calculator",
                "Try another sky (e.g. a darker site, a narrowband filter) "
                "with your measured read noise.",
                self._calculator(ctx),
            )
        doc.notes(ex)
        doc.finish()

    def _curves(self, ctx: Context) -> Plot:
        ex = ctx.mode.exposure
        rn = ex.read_noise_e
        plot = Plot("Sub length (s)", "Extra noise (%)", log_x=True, log_y=True, height=320)
        top = max(max(p.optimal.get(0.02, 10) for p in ex.points) * 4, 600)
        t = np.geomspace(1, top, 200)
        for k, p in enumerate(ex.points):
            colour = theme.SERIES[k % len(theme.SERIES)]
            pen = [noise_penalty(rn, p.flux.value, x) * 100 for x in t]
            plot.line(t, pen, p.filter or p.label, colour, 1.6)
            plot.points(
                [p.exposure],
                [p.penalty * 100],
                None,
                colour,
                size=10,
                tips=[f"{p.label}\n{p.exposure:g} s → +{p.penalty:.1%}"],
            )
        for x in PENALTIES:
            plot.hline(x * 100, f"{x:.0%}")
        return plot

    def _calculator(self, ctx: Context) -> QWidget:
        ex = ctx.mode.exposure
        widget = QWidget()
        form = QFormLayout(widget)
        flux = QDoubleSpinBox(decimals=4, minimum=0.0001, maximum=10_000, singleStep=0.01)
        flux.setValue(ex.points[0].flux.value if ex.points and ex.points[0].flux.value > 0 else 0.5)
        flux.setSuffix(" e⁻/s/px")
        rn = QDoubleSpinBox(decimals=2, minimum=0.05, maximum=100, singleStep=0.05)
        rn.setValue(
            ex.assumed_read_noise if math.isfinite(ex.assumed_read_noise) else ex.read_noise_e
        )
        rn.setSuffix(" e⁻")
        result = QLabel()
        result.setObjectName("figureTitle")

        def update() -> None:
            parts = [
                f"{x:.0%}: {optimal_exposure(rn.value(), flux.value(), x):.0f} s" for x in PENALTIES
            ]
            result.setText("   ".join(parts))

        flux.valueChanged.connect(update)
        rn.valueChanged.connect(update)
        form.addRow("Sky flux", flux)
        form.addRow("Read noise", rn)
        form.addRow("Shortest sub for", result)
        update()
        return widget
