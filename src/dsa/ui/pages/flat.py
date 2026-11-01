"""Flat field: PRNU, vignetting, dust and colour response, per flat set."""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from dsa.analysis.flat import FlatSetResult
from dsa.display import Stretch
from dsa.ui import theme
from dsa.ui.imageview import Overlay, SensorMap
from dsa.ui.pages import Context, plane_label
from dsa.ui.plots import Plot, set_ticks
from dsa.ui.widgets import KeyFigure, Page, link_button, table


class FlatPage(Page):
    title = "Flat field"

    def build(self, ctx: Context, index: int = 0) -> None:
        ff = ctx.mode.flat
        doc = self.new_document()
        doc.heading(
            "Flat field",
            "What the flats show about the sensor and the optics: "
            "pixel-to-pixel sensitivity (PRNU), light falloff towards the edges "
            "(vignetting) and shadows of dust.",
        )
        doc.status(ff)
        if not ff.sets:
            doc.notes(ff)
            doc.finish()
            return
        keys = [
            (KeyFigure("PRNU", ff.prnu, "pixel response non-uniformity, rms"), "prnu"),
            (KeyFigure("Dust motes", str(len(ff.dust)), "merged over all flats"), "dust"),
        ]
        doc.keys(keys, columns=4)
        if len(ff.sets) > 1:
            plot = Plot("Flat set", "PRNU (%)", height=220)
            x = np.arange(len(ff.sets))
            plot.points(
                x,
                [s.prnu_combined.value for s in ff.sets],
                "PRNU",
                theme.ACCENT,
                size=8,
                tips=[f"{s.label}\n{s.prnu_combined.format()}" for s in ff.sets],
            )
            set_ticks(plot, "bottom", [(i, s.filter or str(i + 1)) for i, s in enumerate(ff.sets)])
            doc.figure(
                "PRNU by flat set",
                "PRNU depends a little on wavelength; the sets should be close.",
                plot,
                anchor="prnu",
            )

        chooser = QWidget()
        row = QHBoxLayout(chooser)
        row.setContentsMargins(0, 16, 0, 0)
        row.addWidget(QLabel("Show flat set"))
        combo = QComboBox()
        combo.addItems([s.label for s in ff.sets])
        combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda i: self._reselect(ctx, i))
        row.addWidget(combo)
        row.addStretch(1)
        doc.add(chooser)
        self._set(ctx, ff.sets[index])
        doc.notes(ff)
        doc.finish()

    def _reselect(self, ctx: Context, index: int) -> None:
        value = self.verticalScrollBar().value()
        self.build(ctx, index)
        self.verticalScrollBar().setValue(value)

    def _set(self, ctx: Context, s: FlatSetResult) -> None:
        doc = self.doc
        doc.add(link_button(f"{s.label} →", lambda: ctx.open_set(s.label)))
        vig = s.vignetting
        keys = [
            (
                KeyFigure("PRNU", s.prnu_combined, f"level {s.level_fraction:.0%} of full scale"),
                None,
            )
        ]
        if vig is not None:
            worst = vig.corners[vig.worst_corner]
            keys += [
                (
                    KeyFigure(
                        "Darkest corner",
                        f"{worst:.0%}",
                        f"{vig.worst_corner}, −{vig.falloff_stops:.2f} stops",
                    ),
                    None,
                ),
                (
                    KeyFigure(
                        "Optical centre offset",
                        f"{math.hypot(*vig.centre_offset):.0f} px",
                        f"x {vig.centre_offset[0]:+.0f}, y {vig.centre_offset[1]:+.0f}",
                    ),
                    None,
                ),
                (
                    KeyFigure(
                        "Asymmetry",
                        f"{vig.asymmetry_x:+.1%}",
                        f"right vs left; bottom vs top {vig.asymmetry_y:+.1%}",
                    ),
                    None,
                ),
            ]
        doc.keys(keys)
        if vig is not None:
            cx, cy = vig.centre
            centre = Overlay(
                "optical centre", np.array([cx]), np.array([cy]), theme.TEXT, 14, symbol="+"
            )
            doc.figure(
                "Vignetting",
                "Light reaching each part of the sensor, relative to the "
                "brightest point (dust and defects median-filtered out). The cross marks "
                "the fitted optical centre; an off-centre cross or tilted contours point to "
                "tilt or a decentred optical train.",
                SensorMap(
                    vig.map,
                    "relative",
                    scale=vig.bin_factor,
                    stretch=Stretch.MINMAX,
                    colour_map="Viridis",
                    height=380,
                    overlays=[centre],
                    title="Vignetting",
                ),
            )
            plot = Plot("Distance from optical centre (px)", "Relative illumination", height=260)
            plot.points(vig.radius, vig.profile, "flat", theme.ACCENT, size=6)
            plot.line(vig.radius, vig.model, "radial model", theme.TEXT, 1.2, dashed=True)
            doc.figure(
                "Radial falloff",
                f"Median illumination in rings around the optical centre. "
                f"Model residual {vig.residual_rms:.2%} rms.",
                plot,
            )
            if vig.plane_corners:
                rows = [(plane_label(k), f"{v:.1%}") for k, v in vig.plane_corners.items()]
                doc.add(table(["Plane", "Corner / centre"], rows, numeric=(1,)))
        if s.prnu_map is not None:
            doc.figure(
                "Pixel response (PRNU) map",
                "Each pixel's sensitivity relative to its "
                "neighbourhood, in %. Should look like fine-grained noise; structure would "
                "be a sensor or calibration issue.",
                SensorMap(
                    s.prnu_map,
                    "%",
                    cfa=ctx.mode.mode.cfa,
                    stretch=Stretch.SYMMETRIC,
                    colour_map="Diverging",
                    title="PRNU map",
                ),
                anchor="prnu-map",
            )
            rows = [
                (plane_label(n), v.format())
                for n, v in zip(ctx.mode.flat.planes, s.prnu, strict=True)
            ]
            if len(rows) > 1:
                doc.add(table(["Plane", "PRNU"], rows, numeric=(1,)))
        if s.dust_map is not None:
            overlays = []
            if s.dust:
                overlays.append(
                    Overlay(
                        "dust",
                        np.array([d.x for d in s.dust]),
                        np.array([d.y for d in s.dust]),
                        theme.WARN,
                        sizes=np.array([d.diameter * 1.3 for d in s.dust]),
                    )
                )
            doc.figure(
                "Dust",
                "Light lost to dust shadows (relative to the smooth flat). "
                "Out-of-focus dust casts rings ('donuts'); the further from the sensor, the "
                "larger and fainter.",
                SensorMap(
                    s.dust_map * 100,
                    "% light loss",
                    scale=s.dust_scale,
                    stretch=Stretch.SYMMETRIC,
                    colour_map="Grey",
                    height=380,
                    overlays=overlays,
                    title="Dust",
                ),
                anchor="dust",
            )
        if s.dust:
            rows = [
                (
                    f"{d.x:.0f}",
                    f"{d.y:.0f}",
                    f"{d.diameter:.0f}",
                    f"{d.depth:.1%}",
                    "ring" if d.donut else "spot",
                    f"{d.distance_mm:.1f} mm" if math.isfinite(d.distance_mm) else "—",
                    ", ".join(_filters_for(ctx, d)),
                )
                for d in s.dust[:60]
            ]
            doc.add(
                table(
                    ["x", "y", "Diameter (px)", "Depth", "Shape", "Distance", "Seen in"],
                    rows,
                    numeric=(0, 1, 2, 3),
                )
            )
        if s.channel_ratios:
            rows = [(k, f"{v:.3f}") for k, v in s.channel_ratios.items()]
            doc.figure(
                "Colour response",
                "Flat level of each colour plane relative to green: the "
                "light source's colour times the filters' response. G1/G2 should be 1.",
                table(["Ratio", "Value"], rows, numeric=(1,)),
            )


def _filters_for(ctx: Context, mote) -> list[str]:
    for m in ctx.mode.flat.dust:
        if math.hypot(m.x - mote.x, m.y - mote.y) < max(12.0, 0.3 * m.diameter):
            return m.filters
    return mote.filters
