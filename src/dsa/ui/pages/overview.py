"""Overview: the sensor at a glance, what the data supports, and how settings compare."""

from __future__ import annotations

import math

import numpy as np

from dsa.analysis.common import Status
from dsa.analysis.run import ModeReport
from dsa.analysis.specs import find_spec
from dsa.ui import theme
from dsa.ui.pages import Context
from dsa.ui.plots import Plot
from dsa.ui.widgets import STATUS_TEXT, KeyFigure, Page, table


def describe_sensor(ctx: Context) -> str:
    m, fmt = ctx.mode.mode, ctx.mode.format
    parts = []
    if ctx.camera.spec:
        parts.append(ctx.camera.spec.sensor)
    parts.append(f"{m.width} × {m.height}")
    if ctx.camera.pixel_size:
        parts.append(f"{ctx.camera.pixel_size:g} µm pixels")
    parts.append(f"Bayer {m.cfa}" if m.cfa else "mono")
    if fmt.integer and math.isfinite(fmt.adc_bits):
        bits = f"{round(fmt.adc_bits)}-bit"
        if fmt.container_scale > 1:
            bits += f" stored ×{fmt.container_scale}"
        parts.append(bits)
    if fmt.lossy:
        parts.append("lossy compressed raw")
    return " · ".join(parts)


def _rows(report: ModeReport) -> list[tuple[str, str, Status, str, str]]:
    """(page key, analysis, status, basis, how to improve)."""
    out = []
    sections = [
        ("noise", "Read noise", report.read_noise, report.read_noise.set_label),
        (
            "pattern",
            "Pattern noise",
            report.pattern,
            report.pattern.source_label or report.pattern.fixed_label,
        ),
        (
            "gain",
            "Gain & photon transfer",
            report.gain,
            ", ".join(s.label for s in report.gain.per_set[:3])
            + (" …" if len(report.gain.per_set) > 3 else "")
            or report.gain.estimated_from,
        ),
        ("gain", "Dynamic range", report.dynamic_range, report.dynamic_range.saturation_kind),
        ("dark", "Dark current", report.dark, report.dark.map_label),
        ("pixels", "Defective pixels", report.defects, ""),
        (
            "flat",
            "Flat field",
            report.flat,
            ", ".join(s.filter or s.label for s in report.flat.sets),
        ),
        (
            "exposure",
            "Sub-exposure",
            report.exposure,
            ", ".join(p.label for p in report.exposure.points),
        ),
    ]
    for key, name, section, basis in sections:
        out.append((key, name, section.status, basis, section.unlock[0] if section.unlock else ""))
    return out


class OverviewPage(Page):
    title = "Overview"

    def build(self, ctx: Context) -> None:
        r = ctx.mode
        doc = self.new_document()
        doc.heading(f"{ctx.camera.camera} — {r.mode.setting}", describe_sensor(ctx))

        dr, dark, defects = r.dynamic_range, r.dark, r.defects
        hot = defects.get("hot")
        hot_text = f"{hot.count:,}" if hot else "—"
        hot_note = f"{hot.fraction:.3%} of pixels" if hot else "needs darks"
        temp = dark.reference_temperature
        glow = dark.glow
        glow_text = "none" if glow is None or glow.location == "none" else glow.location
        rn = r.read_noise.combined_e if r.read_noise.combined_e.ok else r.read_noise.combined_adu
        figures = [
            (
                KeyFigure(
                    "Gain",
                    r.gain.gain,
                    "e⁻ per file ADU"
                    + (
                        f" · {r.gain.native.format(4)} native"
                        if r.format.container_scale > 1 and r.gain.gain.ok
                        else ""
                    ),
                ),
                "gain",
            ),
            (
                KeyFigure(
                    "Read noise", rn, "upper limit (no bias)" if rn.bound else "temporal, rms"
                ),
                "noise",
            ),
            (KeyFigure("Saturation", dr.saturation_e, dr.saturation_kind), "gain"),
            (
                KeyFigure(
                    "Dynamic range", dr.stops, f"{dr.db:.1f} dB" if math.isfinite(dr.db) else ""
                ),
                "gain",
            ),
            (
                KeyFigure(
                    "Dark current",
                    dark.current,
                    f"at {temp:.0f} °C" if math.isfinite(temp) else dark.estimator,
                ),
                "dark",
            ),
            (KeyFigure("PRNU", r.flat.prnu, "pixel response, rms"), "flat"),
            (KeyFigure("Hot pixels", hot_text, hot_note), "pixels"),
            (
                KeyFigure(
                    "Amp glow",
                    glow_text,
                    f"+{glow.peak:.2g} {dark.unit}" if glow and glow.location != "none" else "",
                ),
                "dark",
            ),
        ]
        doc.keys([(f, None) for f, _ in figures])
        for figure, page in figures:
            figure.clicked.connect(lambda p=page: ctx.goto(p))

        doc.subheading("What your data supports")
        rows = _rows(r)
        doc.add(
            table(
                ["Analysis", "Status", "Based on", "To improve"],
                [(n, STATUS_TEXT[s].capitalize(), b, u) for _, n, s, b, u in rows],
            )
        )

        if len(ctx.camera.modes) > 1:
            self._across_settings(ctx)
        doc.finish()

    def _across_settings(self, ctx: Context) -> None:
        doc = self.doc
        modes = [m for m in ctx.camera.modes if m.setting_value]
        if len(modes) < 2:
            return
        name = "ISO" if modes[0].mode.iso is not None else "Gain setting"
        xs = np.array([m.setting_value for m in modes], float)
        spec = find_spec(ctx.camera.camera)
        curve = spec.curve_for(modes[0].mode.readout) if spec else None

        rn_plot = Plot(name, "Read noise (e⁻)", log_x=True, height=260)
        rn = [m.read_noise.combined_e for m in modes]
        ok = np.array([v.ok for v in rn])
        if curve:
            rn_plot.line(curve.setting, curve.read_noise, "published", theme.FAINT, dashed=True)
        if ok.any():
            tips = [f"{m.mode.setting}: {v.format()}" for m, v in zip(modes, rn, strict=True)]
            rn_plot.points(
                xs[ok],
                [v.value for v in np.array(rn)[ok]],
                "measured",
                theme.ACCENT,
                size=8,
                tips=[t for t, o in zip(tips, ok, strict=True) if o],
            )
            rn_plot.errorbars(
                xs[ok], [v.value for v in np.array(rn)[ok]], [v.error for v in np.array(rn)[ok]]
            )
        doc.figure(
            "Read noise across settings",
            "Input-referred read noise at each setting found in your data. Where it stops "
            "falling, higher settings only cost dynamic range.",
            rn_plot,
        )

        g_plot = Plot(name, "Gain (e⁻/ADU, native)", log_x=True, log_y=True, height=260)
        gains = [m.gain.native for m in modes]
        good = [(x, g, m) for x, g, m in zip(xs, gains, modes, strict=True) if g.ok]
        if curve:
            g_plot.line(curve.setting, curve.gain, "published", theme.FAINT, dashed=True)
        measured = [(x, g, m) for x, g, m in good if not m.gain.estimated_from]
        scaled = [(x, g, m) for x, g, m in good if m.gain.estimated_from]
        for group, label, colour in (
            (measured, "measured", theme.ACCENT),
            (scaled, "scaled from another setting", theme.WARN),
        ):
            if group:
                g_plot.points(
                    [x for x, _, _ in group],
                    [g.value for _, g, _ in group],
                    label,
                    colour,
                    size=8,
                    tips=[f"{m.mode.setting}: {g.format()}" for _, g, m in group],
                )
        g_plot.hline(1.0, "unity gain")
        doc.figure(
            "Gain across settings",
            "Electrons per ADC step. Unity gain (1 e⁻/ADU) is where one electron is one "
            "step; above it the ADC resolves single electrons.",
            g_plot,
        )

        dr_values = [m.dynamic_range.stops for m in modes]
        if any(v.ok for v in dr_values):
            dr_plot = Plot(name, "Dynamic range (stops)", log_x=True, height=240)
            pts = [(x, v, m) for x, v, m in zip(xs, dr_values, modes, strict=True) if v.ok]
            dr_plot.points(
                [x for x, _, _ in pts],
                [v.value for _, v, _ in pts],
                "measured",
                theme.ACCENT,
                size=8,
                tips=[f"{m.mode.setting}: {v.format()}" for _, v, m in pts],
            )
            dr_plot.line(
                [x for x, _, _ in pts], [v.value for _, v, _ in pts], None, theme.ACCENT, 1
            )
            doc.figure(
                "Dynamic range across settings",
                "log₂(saturation / read noise). Each doubling of analogue gain costs up to "
                "one stop once read noise stops improving.",
                dr_plot,
            )
