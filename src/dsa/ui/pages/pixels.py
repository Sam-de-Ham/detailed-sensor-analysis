"""Defective pixels: classes, map, distribution, RTS examples, and a defect list export."""

from __future__ import annotations

import csv
import math

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QPushButton

from dsa.display import Stretch
from dsa.ui import theme
from dsa.ui.imageview import Overlay, SensorMap
from dsa.ui.pages import Context
from dsa.ui.plots import Plot
from dsa.ui.widgets import KeyFigure, Page, table

CLASS_COLOURS = {
    "hot": theme.PROBLEM,
    "warm": theme.WARN,
    "bias": theme.SERIES[4],
    "dead": theme.SERIES[0],
    "weak": theme.SERIES[2],
    "strong": theme.SERIES[5],
    "rts": theme.SERIES[3],
}


class PixelsPage(Page):
    title = "Defective pixels"

    def build(self, ctx: Context) -> None:
        d = ctx.mode.defects
        doc = self.new_document()
        doc.heading(
            "Defective pixels",
            "Pixels that behave unlike their neighbours: too much dark "
            "current, too little response, an offset of their own, or noise that jumps "
            "between levels.",
        )
        doc.status(d)
        keys = []
        for c in d.classes:
            keys.append((KeyFigure(c.label, f"{c.count:,}", f"{c.fraction:.4%} of pixels"), "map"))
        if d.bad_columns:
            cols = sorted({x for x, _, _ in d.bad_columns})
            keys.append(
                (KeyFigure("Bad columns", str(len(cols)), "mean far from neighbours"), "columns")
            )
        if keys:
            doc.keys(keys)
        if d.classes:
            export = QPushButton("Export defect list…")
            export.clicked.connect(lambda: self._export(ctx))
            rows = [
                (c.label, f"{c.count:,}", f"{c.fraction:.4%}", c.description) for c in d.classes
            ]
            doc.figure(
                "Classes",
                "Counts per class. The list (with positions and values) can be "
                "exported, e.g. for cosmetic correction.",
                table(["Class", "Pixels", "Fraction", "Definition"], rows, numeric=(1, 2)),
                tools=[export],
            )
        background = ctx.mode.dark.map
        if background is None and ctx.mode.flat.sets:
            background = ctx.mode.flat.sets[0].prnu_map
        if background is not None and d.classes:
            overlays = [
                Overlay(
                    c.label, c.xs[:5000], c.ys[:5000], CLASS_COLOURS.get(c.key, theme.ACCENT), 8
                )
                for c in d.classes
                if c.count
            ]
            doc.figure(
                "Where they are",
                "Up to 5000 pixels per class, circled on the dark-current "
                "map. Zoom in to see single pixels.",
                SensorMap(
                    background,
                    "",
                    cfa=ctx.mode.mode.cfa,
                    stretch=Stretch.PERCENTILE,
                    colour_map="Grey",
                    overlays=overlays,
                    title="Defect map",
                ),
                anchor="map",
            )
        if d.cumulative is not None:
            thresholds, above = d.cumulative
            plot = Plot(
                f"Excess dark current ({d.cumulative_unit})",
                "Pixels above",
                log_x=True,
                log_y=True,
                height=280,
            )
            keep = above > 0
            plot.line(thresholds[keep], above[keep], "pixels", theme.ACCENT, 1.6)
            if math.isfinite(d.sub_noise):
                plot.vline(2 * d.sub_noise, "warm")
                plot.vline(5 * d.sub_noise, "hot")
            doc.figure(
                "How many pixels exceed a given dark current",
                "The full distribution (EMVA 1288 style), so you can apply your own "
                "threshold. Lines: warm (2×) and hot (5×) one sub's noise.",
                plot,
            )
        if d.rts is not None and d.rts.count:
            self._rts(ctx)
        if d.bad_columns:
            rows = [(str(x), src, f"{z:+.1f}") for x, src, z in d.bad_columns[:50]]
            doc.figure(
                "Bad columns",
                "Columns whose median deviates by more than 8σ from the others.",
                table(["Column", "Seen in", "z"], rows, numeric=(0, 2)),
                anchor="columns",
            )
        doc.notes(d)
        doc.finish()

    def _rts(self, ctx: Context) -> None:
        rts = ctx.mode.defects.rts
        doc = self.doc
        plot = Plot("Frame", "Value (ADU)", height=280)
        for k, trace in enumerate(rts.traces[:6]):
            x, y = rts.trace_xy[k]
            plot.line(
                np.arange(trace.size),
                trace - np.median(trace),
                f"({x}, {y})",
                theme.SERIES[k % len(theme.SERIES)],
                1.3,
            )
        doc.figure(
            "Random telegraph signal",
            f"{rts.count:,} pixels ({rts.fraction:.3%}) in "
            f"{rts.source} jump between discrete levels. Examples, each relative to its "
            "median; the largest steps first.",
            plot,
        )
        if rts.amplitudes.size:
            counts, edges = np.histogram(rts.amplitudes, bins=40)
            hist = Plot(f"Step size ({rts.unit})", "Pixels", height=220)
            hist.line((edges[:-1] + edges[1:]) / 2, counts, "RTS pixels", theme.SERIES[3], 1.5)
            doc.figure("RTS step sizes", "How far RTS pixels jump.", hist)

    def _export(self, ctx: Context) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export defect list", "defects.csv", "CSV (*.csv)"
        )
        if path:
            write_defects_csv(ctx.mode.defects, path)


def write_defects_csv(defects, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "x", "y", "value", "unit"])
        for c in defects.classes:
            for x, y, v in zip(c.xs, c.ys, c.values, strict=True):
                w.writerow([c.key, int(x), int(y), f"{float(v):.6g}", c.unit])
