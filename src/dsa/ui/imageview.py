"""Zoomable sensor maps: pixel-exact zoom, stretches, colour maps, overlays and a value readout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dsa.cfa import CfaLayout
from dsa.display import Stretch, apply
from dsa.ui import theme

MARKERS_SHOWN = 300  # overlays with more markers start hidden

COLOUR_MAPS = {
    "Inferno": "inferno",
    "Grey": "gray",
    "Viridis": "viridis",
    "Cividis": "cividis",
    "Diverging": "RdBu_r",
}


@dataclass
class Overlay:
    name: str
    xs: np.ndarray  # full-resolution pixel coordinates
    ys: np.ndarray
    colour: str
    size: float = 7  # screen pixels, or data pixels (diameters) when ``sizes`` is given
    sizes: np.ndarray | None = None  # per-marker diameter in pixels (circles)
    symbol: str = "o"


class SensorMap(QWidget):
    """An image of the sensor (or a map derived from it) that zooms down to single pixels."""

    def __init__(
        self,
        data: np.ndarray,
        unit: str = "",
        *,
        cfa: str | None = None,
        scale: int = 1,
        stretch: Stretch = Stretch.PERCENTILE,
        colour_map: str = "Inferno",
        height: int = 420,
        title: str = "",
        overlays: list[Overlay] | None = None,
        square_pixels: bool = True,
    ) -> None:
        super().__init__()
        self._full = np.asarray(data, dtype=np.float32)
        self._unit = unit
        self._layout_cfa = CfaLayout(cfa) if cfa and scale == 1 else CfaLayout(None)
        self._scale = scale
        self._title = title
        self._overlays = overlays or []
        self._data = self._full
        self._plane_step = 1
        self._plane_offset = (0, 0)
        self.setMinimumHeight(height)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.plane = QComboBox()
        self.plane.addItem("Mosaic")
        if not self._layout_cfa.is_mono:
            self.plane.addItems(self._layout_cfa.names)
        self.plane.setVisible(not self._layout_cfa.is_mono)
        self.plane.setToolTip("Show all pixels, or one colour plane of the Bayer pattern")
        self.stretch = QComboBox()
        self.stretch.addItems([s.value for s in Stretch])
        self.stretch.setCurrentText(stretch.value)
        self.stretch.setToolTip("Display stretch (doesn't change the data)")
        self.cmap = QComboBox()
        self.cmap.addItems(list(COLOUR_MAPS))
        self.cmap.setCurrentText(colour_map)
        for w in (self.plane, self.stretch, self.cmap):
            bar.addWidget(w)
        bar.addStretch(1)
        self.readout = QLabel(" ")
        self.readout.setFont(theme.mono_font(9))
        self.readout.setObjectName("muted")
        bar.addWidget(self.readout)
        count = sum(ov.xs.size for ov in self._overlays)
        self.show_markers = QToolButton()
        self.show_markers.setText(f"Markers ({count:,})")
        self.show_markers.setCheckable(True)
        # Thousands of circles hide the image at full view; they're for zooming in.
        self.show_markers.setChecked(count <= MARKERS_SHOWN)
        self.show_markers.setVisible(count > 0)
        self.show_markers.setToolTip("Show or hide the circled pixels")
        self.show_markers.toggled.connect(self._draw_overlays)
        bar.addWidget(self.show_markers)
        for text, tip, slot in (
            ("Fit", "Show the whole sensor", self.fit),
            ("1:1", "One screen pixel per sensor pixel", self.one_to_one),
            ("Expand", "Open in a large window", self.expand),
            ("Save", "Save as PNG (as displayed) or FITS (the data)", self.save),
        ):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        layout.addLayout(bar)

        self.view = pg.GraphicsLayoutWidget()
        self.view.setBackground(theme.BACKGROUND)
        self.plot = self.view.addPlot()
        self.plot.setAspectLocked(square_pixels)  # False for data tables shown as images
        self._square = square_pixels
        self.plot.invertY(True)
        self.plot.showGrid(False, False)
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setPen(pg.mkPen(theme.BORDER))
            self.plot.getAxis(axis).setTextPen(pg.mkPen(theme.FAINT))
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setAutoDownsample(True)
        self.plot.addItem(self.image)
        self.colorbar = pg.ColorBarItem(interactive=False, width=12, label=unit)
        self.colorbar.setImageItem(self.image, insert_in=self.plot)
        self.colorbar.getAxis("right").setTextPen(pg.mkPen(theme.MUTED))
        layout.addWidget(self.view, 1)
        self._markers: list[pg.ScatterPlotItem] = []

        self.plane.currentTextChanged.connect(self._select_plane)
        self.stretch.currentTextChanged.connect(self._render)
        self.cmap.currentTextChanged.connect(self._render)
        self.view.scene().sigMouseMoved.connect(self._hover)
        self._render()
        self._draw_overlays()
        self.fit()

    # --------------------------------------------------------------------- rendering --

    def _select_plane(self, name: str) -> None:
        if name == "Mosaic":
            self._data, self._plane_step, self._plane_offset = self._full, 1, (0, 0)
        else:
            self._data = self._layout_cfa.plane(self._full, name)
            self._plane_step = self._layout_cfa.step
            self._plane_offset = self._layout_cfa.offsets[name]
        self._render()
        self._draw_overlays()

    def _render(self) -> None:
        stretch = Stretch(self.stretch.currentText())
        shown, (lo, hi) = apply(self._data, stretch)
        cmap = pg.colormap.getFromMatplotlib(COLOUR_MAPS[self.cmap.currentText()])
        self.image.setImage(np.nan_to_num(shown, nan=lo), levels=(lo, hi))
        self.image.setColorMap(cmap)
        step = self._plane_step * self._scale
        h, w = self._data.shape
        self.image.setRect(self._plane_offset[1], self._plane_offset[0], w * step, h * step)
        linear = stretch in (Stretch.PERCENTILE, Stretch.SYMMETRIC, Stretch.MINMAX)
        self.colorbar.setColorMap(cmap)
        self.colorbar.setLevels((lo, hi))
        self.colorbar.setVisible(linear)

    def _draw_overlays(self) -> None:
        for m in self._markers:
            self.plot.removeItem(m)
        self._markers = []
        if not self.show_markers.isChecked():
            return
        for ov in self._overlays:
            xs, ys = ov.xs.astype(float), ov.ys.astype(float)
            if self._plane_step > 1:  # keep only markers on the shown colour plane
                dy, dx = self._plane_offset
                keep = (ys.astype(int) % 2 == dy) & (xs.astype(int) % 2 == dx)
                xs, ys = xs[keep], ys[keep]
            pen = pg.mkPen(ov.colour, width=1.3)
            if ov.sizes is not None:
                item = pg.ScatterPlotItem(
                    x=xs,
                    y=ys,
                    size=ov.sizes,
                    pxMode=False,
                    pen=pen,
                    brush=None,
                    symbol=ov.symbol,
                    name=ov.name,
                )
            else:
                item = pg.ScatterPlotItem(
                    x=xs + 0.5,
                    y=ys + 0.5,
                    size=ov.size,
                    pen=pen,
                    brush=None,
                    symbol=ov.symbol,
                    name=ov.name,
                )
            item.setZValue(10)
            self.plot.addItem(item)
            self._markers.append(item)

    # ---------------------------------------------------------------------- actions --

    def fit(self) -> None:
        self.plot.autoRange(padding=0.01)

    def one_to_one(self) -> None:
        vb = self.plot.getViewBox()
        (x0, x1), (y0, y1) = vb.viewRange()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = vb.width() / 2, vb.height() / 2
        vb.setRange(xRange=(cx - w, cx + w), yRange=(cy - h, cy + h), padding=0)

    def expand(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._title or "Sensor map")
        dialog.resize(1300, 900)
        box = QVBoxLayout(dialog)
        big = SensorMap(
            self._full,
            self._unit,
            cfa=self._layout_cfa.pattern,
            scale=self._scale,
            stretch=Stretch(self.stretch.currentText()),
            colour_map=self.cmap.currentText(),
            height=700,
            title=self._title,
            overlays=self._overlays,
            square_pixels=self._square,
        )
        box.addWidget(big)
        dialog.show()

    def save(self) -> None:
        base = (self._title or "map").replace(" ", "_").replace("/", "-")
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save map", base, "PNG image (*.png);;FITS data (*.fits)"
        )
        if not path:
            return
        if path.lower().endswith((".fits", ".fit")) or "FITS" in chosen:
            from astropy.io import fits

            fits.PrimaryHDU(self._full, header=fits.Header({"BUNIT": self._unit})).writeto(
                path, overwrite=True
            )
            return
        render_png(
            self._data, Stretch(self.stretch.currentText()), self.cmap.currentText(), Path(path)
        )

    def _hover(self, pos) -> None:
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        p = self.plot.getViewBox().mapSceneToView(pos)
        x, y = int(np.floor(p.x())), int(np.floor(p.y()))
        step = self._plane_step * self._scale
        dy, dx = self._plane_offset
        iy, ix = (y - dy) // step, (x - dx) // step
        h, w = self._data.shape
        if 0 <= iy < h and 0 <= ix < w:
            value = float(self._data[iy, ix])
            self.readout.setText(f"x {x:>5}  y {y:>5}   {value:.4g} {self._unit}")
        else:
            self.readout.setText(" ")


def render_png(data: np.ndarray, stretch: Stretch, colour_map: str, path: Path) -> None:
    """Save a map at native resolution with the given stretch and colour map."""
    rgba = to_rgba(data, stretch, colour_map)
    h, w, _ = rgba.shape
    image = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    image.save(str(path))


def to_rgba(data: np.ndarray, stretch: Stretch, colour_map: str) -> np.ndarray:
    shown, (lo, hi) = apply(data, stretch)
    norm = np.clip((np.nan_to_num(shown, nan=lo) - lo) / max(hi - lo, 1e-12), 0, 1)
    cmap = pg.colormap.getFromMatplotlib(COLOUR_MAPS.get(colour_map, "inferno"))
    lut = cmap.getLookupTable(0.0, 1.0, 256, alpha=True)
    return np.ascontiguousarray(lut[(norm * 255).astype(np.uint8)])


def swatch(colour: str) -> QLabel:
    label = QLabel("●")
    label.setStyleSheet(f"color: {QColor(colour).name()};")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label
