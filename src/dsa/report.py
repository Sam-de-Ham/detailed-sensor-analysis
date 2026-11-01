"""A self-contained, shareable HTML report (figures embedded as PNG, light theme for reading)."""

from __future__ import annotations

import base64
import html
import io
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsa import __version__
from dsa.analysis.common import Level, Section, Status, Value, format_value
from dsa.analysis.exposure import PENALTIES, noise_penalty
from dsa.analysis.run import CameraReport, ModeReport, headline
from dsa.display import Stretch, apply
from dsa.stats import bin_image

INK = "#1d2430"
MUTED = "#6b7480"
ACCENT = "#2f6fb3"
SERIES = ["#2f6fb3", "#c98a1e", "#2e9c8c", "#c24f78", "#7a5cc7", "#8a9a2c"]
PLANES = {"L": "#3c4450", "R": "#c44a4a", "G1": "#3f9a5c", "G2": "#7bb35a", "B": "#3f6fc4"}

CSS = f"""
body {{ font-family: 'Segoe UI', Inter, Helvetica, Arial, sans-serif; color: {INK};
       max-width: 1040px; margin: 40px auto; padding: 0 24px; line-height: 1.5; }}
h1 {{ font-weight: 600; margin-bottom: 4px; }} h2 {{ margin-top: 48px; font-weight: 600;
     border-bottom: 1px solid #dde2e8; padding-bottom: 6px; }}
h3 {{ margin-top: 28px; font-weight: 600; font-size: 1.05em; }}
.lead, .caption, .muted {{ color: {MUTED}; }} .caption {{ font-size: 0.92em; margin-top: 2px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.93em; margin: 8px 0 16px; }}
th {{ text-align: left; color: {MUTED}; font-weight: 600; border-bottom: 1px solid #dde2e8;
      padding: 4px 8px; }}
td {{ padding: 4px 8px; border-bottom: 1px solid #f0f2f5; vertical-align: top; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.keys {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px 22px; margin: 18px 0; }}
.key {{ border-top: 1px solid #dde2e8; padding-top: 6px; }}
.key .l {{ color: {MUTED}; font-size: 0.85em; }} .key .v {{ font-size: 1.35em; }}
.key .n {{ color: {MUTED}; font-size: 0.8em; }}
.note {{ border-left: 3px solid #b8c0ca; padding: 2px 10px; margin: 6px 0; }}
.note.warn {{ border-color: #d39a2c; }} .note.problem {{ border-color: #d05050; }}
.note.ok {{ border-color: #4aa36b; }}
.status {{ font-size: 0.75em; font-weight: 600; letter-spacing: 0.5px; color: {MUTED}; }}
.status.measured {{ color: #3a8a5a; }} .status.limited {{ color: #b07a18; }}
img {{ max-width: 100%; }} figure {{ margin: 18px 0; }}
footer {{ margin-top: 60px; color: {MUTED}; font-size: 0.85em; }}
"""


def _e(text: object) -> str:
    return html.escape(str(text))


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _figure(title: str, caption: str, image: str) -> str:
    return f"<figure><h3>{_e(title)}</h3><div class='caption'>{_e(caption)}</div>{image}</figure>"


def _axes(xlabel: str, ylabel: str, logx: bool = False, logy: bool = False, size=(8.5, 3.6)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.25)
    return fig, ax


def _map(
    data: np.ndarray,
    unit: str,
    stretch: Stretch = Stretch.PERCENTILE,
    cmap: str = "inferno",
    circles: list[tuple[float, float, float]] | None = None,
    scale: int = 1,
) -> str:
    factor = max(1, max(data.shape) // 900)
    small = bin_image(np.asarray(data, np.float32), factor) if factor > 1 else data
    shown, (lo, hi) = apply(small, stretch)
    h, w = small.shape
    full = factor * scale
    fig, ax = plt.subplots(figsize=(8.5, 8.5 * h / w + 0.4))
    im = ax.imshow(
        shown,
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        extent=(0, w * full, h * full, 0),
        interpolation="nearest",
    )
    if stretch in (Stretch.PERCENTILE, Stretch.SYMMETRIC, Stretch.MINMAX):
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=unit)
    for x, y, d in circles or []:
        ax.add_patch(plt.Circle((x, y), d / 2 * 1.3, fill=False, color="#e0a030", lw=1.2))
    ax.tick_params(colors=MUTED, labelsize=8)
    return _png(fig)


def _keys(items: list[tuple[str, Value | str, str]]) -> str:
    cells = []
    for title, value, note in items:
        text = (
            value.format()
            if isinstance(value, Value) and value.ok
            else ("—" if isinstance(value, Value) else value)
        )
        cells.append(
            f"<div class='key'><div class='l'>{_e(title)}</div><div class='v'>"
            f"{_e(text)}</div><div class='n'>{_e(note)}</div></div>"
        )
    return f"<div class='keys'>{''.join(cells)}</div>"


def _table(headers: list[str], rows: list[list[object]], numeric: tuple[int, ...] = ()) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td class='{'num' if i in numeric else ''}'>{_e(c)}</td>" for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _notes(section: Section) -> str:
    out = []
    out += [f"<div class='note {n.level.value}'>{_e(n.text)}</div>" for n in section.notes]
    if section.method:
        out.append(f"<p class='muted'><b>Method.</b> {_e(section.method)}</p>")
    out += [f"<div class='note'>To improve: {_e(u)}</div>" for u in dict.fromkeys(section.unlock)]
    return "".join(out)


def _status(section: Section) -> str:
    text = {
        Status.MEASURED: "MEASURED",
        Status.LIMITED: "LIMITED BY THE DATA",
        Status.UNAVAILABLE: "NOT AVAILABLE FROM THIS DATA",
    }[section.status]
    return f"<div class='status {section.status.value}'>{text}</div>"


# --------------------------------------------------------------------------- sections --


def _mode(r: ModeReport, camera: CameraReport) -> str:
    parts = [f"<h2>{_e(r.mode.setting)}</h2>"]
    parts.append(_keys([(name, v, s.value) for name, v, s in headline(r)]))
    g = r.gain.gain
    unit = "e⁻" if g.ok else "ADU"
    factor = g.value if g.ok else 1.0

    rn = r.read_noise
    parts.append(f"<h3>Read noise</h3>{_status(rn)}")
    if rn.sigma_hist:
        fig, ax = _axes(f"Temporal noise of a pixel ({unit})", "Pixels", logy=True)
        for (edges, counts), name in zip(rn.sigma_hist, rn.planes, strict=True):
            c = (edges[:-1] + edges[1:]) / 2 * factor
            ax.plot(
                c[counts > 0],
                counts[counts > 0],
                color=PLANES.get(name, ACCENT),
                lw=1.2,
                label=name,
            )
        ax.legend(frameon=False)
        parts.append(
            _figure(
                "Per-pixel read noise",
                "Distribution of each pixel's temporal noise; "
                "the tail is RTS and other noisy pixels.",
                _png(fig),
            )
        )
    parts.append(_notes(rn))

    pat = r.pattern
    if pat.row_temporal or pat.row_fixed:
        rows = []
        for i, name in enumerate(pat.planes):

            def cell(values, i=i):
                return (
                    values[i].scaled(factor, unit).format()
                    if i < len(values) and values[i].ok
                    else "—"
                )

            rows.append(
                [
                    name,
                    cell(pat.pixel_temporal),
                    cell(pat.row_temporal),
                    cell(pat.col_temporal),
                    cell(pat.pixel_fixed),
                    cell(pat.row_fixed),
                    cell(pat.col_fixed),
                ]
            )
        parts.append(
            "<h3>Pattern noise</h3>"
            + _status(pat)
            + _table(
                [
                    "Plane",
                    "Pixel (temporal)",
                    "Row (temporal)",
                    "Column (temporal)",
                    "Pixel (fixed)",
                    "Row (fixed)",
                    "Column (fixed)",
                ],
                rows,
                numeric=tuple(range(1, 7)),
            )
        )

    parts.append(f"<h3>Gain &amp; photon transfer</h3>{_status(r.gain)}")
    if r.gain.ptc_signal is not None and g.ok:
        read_var = float(np.mean(r.gain.read_variance)) if r.gain.read_variance else 0.0
        fig, ax = _axes("Signal above offset (ADU)", "Temporal variance (ADU²)", True, True)
        for k in np.unique(r.gain.ptc_set):
            sel = r.gain.ptc_set == k
            ax.scatter(
                r.gain.ptc_signal[sel],
                r.gain.ptc_variance[sel] + read_var,
                s=2,
                alpha=0.35,
                color=SERIES[int(k) % len(SERIES)],
                label=r.gain.per_set[int(k)].filter or r.gain.per_set[int(k)].label,
            )
        xs = np.geomspace(
            1,
            max(r.dynamic_range.saturation_adu, 10)
            if math.isfinite(r.dynamic_range.saturation_adu)
            else float(np.max(r.gain.ptc_signal)) * 2,
            100,
        )
        ax.plot(xs, read_var + xs / g.value, "--", color=MUTED, lw=1, label="read² + S/g")
        ax.legend(frameon=False, fontsize=8, markerscale=4)
        parts.append(
            _figure(
                "Photon transfer curve",
                f"Gain {g.format()}; tiles of each flat pair spread over the vignetting range.",
                _png(fig),
            )
        )
    if r.gain.per_set:
        parts.append(
            _table(
                ["Flat set", "Gain", "Level", "Pairs"],
                [
                    [s.label, s.combined.format(), f"{s.level_fraction:.0%}", s.n_pairs]
                    for s in r.gain.per_set
                ],
                numeric=(1, 2, 3),
            )
        )
    parts.append(_notes(r.gain))
    if r.dynamic_range.method:
        parts.append(f"<p class='muted'>{_e(r.dynamic_range.method)}</p>")

    dk = r.dark
    parts.append(f"<h3>Dark current</h3>{_status(dk)}")
    if dk.points:
        parts.append(
            _table(
                ["Set", "Exposure", "Temperature", "From level", "From noise"],
                [
                    [
                        p.label,
                        f"{p.exposure:g} s",
                        f"{p.temperature:.1f} °C" if math.isfinite(p.temperature) else "—",
                        p.current.format(),
                        p.noise_current.format(),
                    ]
                    for p in dk.points
                ],
                numeric=(1, 2, 3, 4),
            )
        )
    if dk.map is not None:
        parts.append(
            _figure("Dark current map", f"From {dk.map_label} ({dk.unit}).", _map(dk.map, dk.unit))
        )
    if dk.glow is not None and dk.glow.location != "none":
        parts.append(
            _figure(
                "Amp glow",
                f"Peak +{dk.glow.peak:.3g} {dk.unit} at the {dk.glow.location}.",
                _map(dk.glow.map, dk.unit, Stretch.MINMAX, scale=dk.glow.bin_factor),
            )
        )
    parts.append(_notes(dk))

    d = r.defects
    parts.append(f"<h3>Defective pixels</h3>{_status(d)}")
    if d.classes:
        parts.append(
            _table(
                ["Class", "Pixels", "Fraction", "Definition"],
                [[c.label, f"{c.count:,}", f"{c.fraction:.4%}", c.description] for c in d.classes],
                numeric=(1, 2),
            )
        )
    if d.cumulative is not None:
        t, above = d.cumulative
        fig, ax = _axes(f"Excess dark current ({d.cumulative_unit})", "Pixels above", True, True)
        ax.plot(t[above > 0], above[above > 0], color=ACCENT)
        parts.append(
            _figure(
                "Cumulative distribution",
                "Pixels whose dark current exceeds the local level by more than the x value.",
                _png(fig),
            )
        )

    ff = r.flat
    parts.append(f"<h3>Flat field</h3>{_status(ff)}")
    if ff.sets:
        parts.append(
            _table(
                ["Flat set", "PRNU", "Darkest corner", "Centre offset", "Dust"],
                [
                    [
                        s.label,
                        s.prnu_combined.format(),
                        f"{s.vignetting.corners[s.vignetting.worst_corner]:.0%}"
                        if s.vignetting
                        else "—",
                        f"{math.hypot(*s.vignetting.centre_offset):.0f} px"
                        if s.vignetting
                        else "—",
                        len(s.dust),
                    ]
                    for s in ff.sets
                ],
                numeric=(1, 2, 3, 4),
            )
        )
        s = ff.sets[0]
        if s.vignetting is not None:
            parts.append(
                _figure(
                    f"Vignetting — {s.label}",
                    "Relative illumination.",
                    _map(
                        s.vignetting.map,
                        "relative",
                        Stretch.MINMAX,
                        "viridis",
                        scale=s.vignetting.bin_factor,
                    ),
                )
            )
        if s.dust_map is not None:
            circles = [(m.x, m.y, m.diameter) for m in s.dust]
            parts.append(
                _figure(
                    f"Dust — {s.label}",
                    "Light lost to dust shadows (%).",
                    _map(
                        s.dust_map * 100,
                        "%",
                        Stretch.SYMMETRIC,
                        "gray",
                        circles,
                        scale=s.dust_scale,
                    ),
                )
            )
    parts.append(_notes(ff))

    ex = r.exposure
    parts.append(f"<h3>Sub-exposure</h3>{_status(ex)}")
    if ex.points and math.isfinite(ex.read_noise_e):
        fig, ax = _axes("Sub length (s)", "Extra noise (%)", True, True)
        t = np.geomspace(1, 3600, 200)
        for k, p in enumerate(ex.points):
            ax.plot(
                t,
                [noise_penalty(ex.read_noise_e, p.flux.value, x) * 100 for x in t],
                color=SERIES[k % len(SERIES)],
                label=p.filter or p.label,
            )
        for x in PENALTIES:
            ax.axhline(x * 100, color=MUTED, lw=0.6, ls="--")
        ax.legend(frameon=False)
        parts.append(_figure("Read-noise penalty", "Extra noise vs one long exposure.", _png(fig)))
        parts.append(
            _table(
                ["Lights", "Sky flux", "2 %", "5 %", "10 %"],
                [
                    [p.label, p.flux.format(), *(f"{p.optimal[x]:.0f} s" for x in PENALTIES)]
                    for p in ex.points
                ],
                numeric=(1, 2, 3, 4),
            )
        )
    parts.append(_notes(ex))

    parts.append("<h3>Data checks</h3>")
    for c in r.checks:
        level = c.level.value if c.level is not Level.INFO else ""
        parts.append(
            f"<div class='note {level}'><b>{_e(c.title)}</b> — {_e(c.summary)}"
            f"{'<br><span class=muted>' + _e(c.detail) + '</span>' if c.detail else ''}</div>"
        )
    if r.specs.rows:
        parts.append("<h3>Published specs</h3>")
        parts.append(
            _table(
                ["Quantity", "Measured", "Published", "Ratio", "Verdict", "Source"],
                [
                    [
                        c.quantity,
                        c.measured.format(),
                        format_value(c.published, unit=c.unit, digits=4),
                        f"×{c.ratio:.2f}",
                        {True: "agrees", False: "differs", None: "—"}[c.agrees],
                        c.source,
                    ]
                    for c in r.specs.rows
                ],
                numeric=(1, 2, 3),
            )
        )
    return "".join(parts)


def render(report: CameraReport) -> str:
    title = f"{report.camera} — sensor analysis"
    lead = report.spec.sensor if report.spec else ""
    body = [f"<h1>{_e(title)}</h1><div class='lead'>{_e(lead)}</div>"]
    body.append(
        _table(
            ["Setting", "Sets", "Frames"],
            [[m.mode.setting, len(m.sets), sum(s.count for s in m.sets)] for m in report.modes],
            numeric=(1, 2),
        )
    )
    for mode in report.modes:
        body.append(_mode(mode, report))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    body.append(
        f"<footer>Detailed Sensor Analysis {__version__} · {stamp}. Uncertainties are "
        "1σ statistical; systematic effects are discussed in the notes.</footer>"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{_e(title)}</title>"
        f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>"
    )


def write_report(report: CameraReport, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", report.camera).strip("-").lower()
    path = folder / f"dsa-{slug}.html"
    path.write_text(render(report), encoding="utf-8")
    return path
