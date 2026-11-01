"""The frames behind the numbers: every set of the setting, its per-frame statistics, any frame."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTableWidget, QWidget

from dsa.catalog import FrameSet, format_exposure
from dsa.display import Stretch
from dsa.frames import read_pixels
from dsa.measure import SetStats
from dsa.ui import theme
from dsa.ui.imageview import SensorMap
from dsa.ui.pages import Context, plane_label
from dsa.ui.plots import Plot
from dsa.ui.widgets import Page, table

StatsLoader = Callable[[FrameSet], SetStats | None]


class FramesPage(Page):
    title = "Frames"

    def __init__(self, load_stats: StatsLoader) -> None:
        super().__init__()
        self._load_stats = load_stats
        self._ctx: Context | None = None

    def build(self, ctx: Context, select: str | None = None) -> None:
        self._ctx = ctx
        sets = list(ctx.mode.sets)
        doc = self.new_document()
        doc.heading(
            "Frames",
            "Every frame set found at this setting, and the frames behind "
            "each result. Pick a set to see its frames; pick a frame to see it.",
        )
        rows = [
            (
                fs.label,
                fs.kind.label,
                f"{fs.count}",
                str(fs.folder),
                fs.frames[0].kind_source,
                "" if fs.quantitative else "not used (rescaled master)",
            )
            for fs in sets
        ]
        overview = table(
            ["Set", "Type", "Files", "Folder", "Classified by", ""], rows, numeric=(2,)
        )
        doc.figure("Sets", "", overview)
        chooser = QWidget()
        row = QHBoxLayout(chooser)
        row.setContentsMargins(0, 12, 0, 0)
        row.addWidget(QLabel("Set"))
        combo = QComboBox()
        combo.addItems([fs.label for fs in sets])
        row.addWidget(combo, 1)
        doc.add(chooser)
        index = next((i for i, fs in enumerate(sets) if fs.label == select), 0)
        combo.setCurrentIndex(index)
        overview.selectRow(index)
        combo.currentIndexChanged.connect(lambda i: self._rebuild_with(sets[i].label))
        overview.cellClicked.connect(lambda r, _c: self._rebuild_with(sets[r].label))
        if sets:
            self._show_set(sets[index])
        doc.finish()

    def _rebuild_with(self, label: str) -> None:
        if self._ctx is not None:
            self.build(self._ctx, label)

    def _show_set(self, fs: FrameSet) -> None:
        doc = self.doc
        st = self._load_stats(fs) if fs.quantitative else None
        frames = fs.frames
        used = set(st.paths) if st is not None else set()
        rows = []
        for f in frames:
            mean = "—"
            if st is not None and str(f.path) in used:
                k = st.paths.index(str(f.path))
                mean = " / ".join(f"{v:.1f}" for v in st.frame_mean[k])
            rows.append(
                (
                    f.path.name,
                    f.timestamp.strftime("%Y-%m-%d %H:%M:%S") if f.timestamp else "—",
                    format_exposure(f.exposure) if f.exposure is not None else "—",
                    f"{f.temperature:.1f}" if f.temperature is not None else "—",
                    mean,
                    "✓" if str(f.path) in used else "",
                )
            )
        planes = " / ".join(st.planes) if st is not None else ""
        frame_table = table(
            ["File", "Time", "Exposure", "°C", f"Mean ({planes or 'ADU'})", "Used"],
            rows,
            numeric=(2, 3, 4),
            height=260,
        )
        frame_table.setToolTip("Double-click a frame to view it; right-click to copy its path")
        doc.figure(
            f"{fs.label}",
            f"{len(frames)} files in {fs.folder}. Statistics use up to the "
            "set limit of frames, spread evenly over the set (✓).",
            frame_table,
        )
        if st is not None and st.n_frames > 1:
            plot = Plot(
                "Time" if np.isfinite(st.times).any() else "Frame", "Mean level (ADU)", height=220
            )
            x = (
                st.times - np.nanmin(st.times)
                if np.isfinite(st.times).any()
                else np.arange(st.n_frames)
            )
            if np.isfinite(st.times).any():
                plot.getPlotItem().setLabel("bottom", "Seconds since the first frame")
            for i, name in enumerate(st.planes):
                plot.points(
                    x,
                    st.frame_mean[:, i],
                    plane_label(name),
                    theme.PLANE_COLOURS.get(name, theme.ACCENT),
                    size=6,
                    tips=[f"{name}: {v:.2f} ADU" for v in st.frame_mean[:, i]],
                )
            doc.figure(
                "Level through the set",
                "Drifts show warm-up, temperature changes or a changing light source.",
                plot,
            )
        holder = QWidget()
        box = QHBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        self._viewer_box = box
        self._frame_label = QLabel("Double-click a frame above to view it.")
        self._frame_label.setObjectName("muted")
        box.addWidget(self._frame_label)
        doc.figure(
            "Frame",
            "Raw sensor values, auto-stretched for display (readout shows the real values).",
            holder,
        )
        frame_table.cellDoubleClicked.connect(lambda r, _c: self._view(frames[r]))
        frame_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        frame_table.customContextMenuRequested.connect(
            lambda pos: self._copy_path(frame_table, frames, pos)
        )

    def _view(self, info) -> None:
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            pixels = read_pixels(info)
        finally:
            QGuiApplication.restoreOverrideCursor()
        while self._viewer_box.count():
            item = self._viewer_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._viewer_box.addWidget(
            SensorMap(
                pixels.data.astype(np.float32),
                "ADU",
                cfa=info.cfa,
                stretch=Stretch.AUTO,
                colour_map="Grey",
                height=520,
                title=info.path.stem,
            )
        )

    def _copy_path(self, widget: QTableWidget, frames, pos) -> None:
        row = widget.rowAt(pos.y())
        if row >= 0:
            QGuiApplication.clipboard().setText(str(frames[row].path))
