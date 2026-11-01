"""Data checks and the comparison with published specifications."""

from __future__ import annotations

import math

from dsa.analysis.common import Level, format_value
from dsa.ui import theme
from dsa.ui.pages import Context
from dsa.ui.plots import Plot
from dsa.ui.widgets import Page, label, note_widget, table

LEVEL_TITLE = {
    Level.PROBLEM: "Problems",
    Level.WARN: "Warnings",
    Level.INFO: "For information",
    Level.OK: "Passed",
}


class ChecksPage(Page):
    title = "Data checks"

    def build(self, ctx: Context) -> None:
        r = ctx.mode
        doc = self.new_document()
        doc.heading(
            "Data checks",
            "Tests of the frames themselves and of how well they serve "
            "calibration: the traps that make sensor numbers, or calibrated images, wrong.",
        )
        for level in (Level.PROBLEM, Level.WARN, Level.INFO, Level.OK):
            checks = [c for c in r.checks if c.level is level]
            if not checks:
                continue
            doc.subheading(f"{LEVEL_TITLE[level]} ({len(checks)})")
            for c in checks:
                text = f"<b>{c.title}</b> — {c.summary}"
                if c.detail:
                    text += f"<br><span style='color:{theme.MUTED}'>{c.detail}</span>"
                widget = note_widget(level, text)
                doc.add(widget)
        if r.failures:
            doc.subheading("Could not be read")
            for f in r.failures:
                doc.add(note_widget(Level.PROBLEM, f.text))
        doc.finish()


class SpecsPage(Page):
    title = "Published specs"

    def build(self, ctx: Context) -> None:
        spec = ctx.camera.spec
        doc = self.new_document()
        doc.heading(
            "Published specs",
            "Your measurements next to manufacturer data and independent "
            "measurements of the same sensor. Expect broad agreement: individual sensors, "
            "temperatures and measurement methods differ.",
        )
        if spec is None:
            doc.text("No published data is built in for this camera.")
            doc.finish()
            return
        doc.add(label(f"{spec.name} — {spec.sensor}", "figureTitle"))
        for n in spec.notes:
            doc.text(n)
        rows = []
        for m in ctx.camera.modes:
            for c in m.specs.rows:
                verdict = {True: "agrees", False: "differs", None: "—"}[c.agrees]
                if c.agrees is None and c.measured.bound == "upper" and c.ratio >= 1:
                    verdict = "consistent (limit)"
                rows.append(
                    (
                        m.mode.setting,
                        c.quantity,
                        c.measured.format(),
                        format_value(c.published, unit=c.unit, digits=4),
                        f"×{c.ratio:.2f}" if math.isfinite(c.ratio) else "—",
                        verdict,
                        c.source + (f" — {c.note}" if c.note else ""),
                    )
                )
        if rows:
            doc.figure(
                "Comparison",
                "Tolerances: gain ±10 %, read noise ±25 %, saturation ±20 %, dark current ×2.",
                table(
                    ["Setting", "Quantity", "Measured", "Published", "Ratio", "Verdict", "Source"],
                    rows,
                    numeric=(4,),
                ),
            )
        curve = spec.curve_for(ctx.mode.mode.readout)
        modes = [m for m in ctx.camera.modes if m.setting_value]
        if curve is not None:
            for quantity, values, ylabel, attr in (
                ("Gain", curve.gain, "Gain (e⁻/ADU, native)", "gain"),
                ("Read noise", curve.read_noise, "Read noise (e⁻)", "rn"),
                ("Saturation", curve.full_well, "Saturation (e⁻)", "sat"),
            ):
                if not values:
                    continue
                plot = Plot(spec.setting_name, ylabel, log_x=True, log_y=True, height=260)
                plot.line(curve.setting, values, "published", theme.FAINT, dashed=True)
                pts = []
                for m in modes:
                    v = {
                        "gain": m.gain.native,
                        "rn": m.read_noise.combined_e,
                        "sat": m.dynamic_range.saturation_e,
                    }[attr]
                    if v.ok:
                        pts.append((m.setting_value, v, m))
                if pts:
                    plot.points(
                        [p[0] for p in pts],
                        [p[1].value for p in pts],
                        "measured",
                        theme.ACCENT,
                        size=9,
                        tips=[f"{m.mode.setting}: {v.format()}" for _, v, m in pts],
                    )
                doc.figure(f"{quantity} against the published curve", curve.source, plot)
        if spec.dark_temps and len(spec.dark_temps) > 1:
            plot = Plot("Temperature (°C)", "Dark current (e⁻/s/px)", log_y=True, height=260)
            plot.line(spec.dark_temps, spec.dark_current, "published", theme.FAINT, dashed=True)
            for m in ctx.camera.modes:
                d = m.dark.current
                t = m.dark.reference_temperature
                if d.ok and math.isfinite(t) and m.dark.unit == "e⁻/s":
                    tip = f"{m.mode.setting}: {d.format()}" + (" (upper limit)" if d.bound else "")
                    plot.points(
                        [t],
                        [d.value],
                        "measured" + (" (limit)" if d.bound else ""),
                        theme.ACCENT,
                        size=9,
                        symbol="t" if d.bound == "upper" else "o",
                        tips=[tip],
                    )
            doc.figure("Dark current against the published curve", spec.dark_source, plot)
        doc.finish()
