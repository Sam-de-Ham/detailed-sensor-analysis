"""Plots: a styled pyqtgraph PlotWidget with a hover readout of the nearest data point."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor

from dsa.ui import theme


class CleanAxis(pg.AxisItem):
    """Log axes labelled only at 1, 2 and 5 per decade, in plain numbers."""

    def logTickValues(self, minVal, maxVal, size, stdTicks):
        lo, hi = math.floor(minVal), math.ceil(maxVal)
        majors = [float(v) for v in range(lo, hi + 1) if minVal <= v <= maxVal]
        minors = [
            d + math.log10(m)
            for d in range(lo - 1, hi + 1)
            for m in (2, 5)
            if minVal <= d + math.log10(m) <= maxVal
        ]
        if hi - lo > 4:  # many decades: decades only
            minors = []
        return [(None, majors), (None, minors)]

    def logTickStrings(self, values, scale, spacing):
        return [_tick(10**v) for v in values]


@dataclass
class _Series:
    name: str
    x: np.ndarray
    y: np.ndarray
    tips: Sequence[str] | None


def _tick(v: float) -> str:
    if 1 <= v < 1e6:
        return f"{v:,.0f}"
    return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e") if v < 1e-3 else f"{v:g}"


def _fmt(v: float) -> str:
    if not math.isfinite(v):
        return "—"
    a = abs(v)
    if a != 0 and (a >= 1e5 or a < 1e-3):
        return f"{v:.3e}"
    return f"{v:.4g}"


class Plot(pg.PlotWidget):
    """A plot with consistent styling. Hovering shows the nearest point of any series."""

    def __init__(
        self,
        x_label: str = "",
        y_label: str = "",
        *,
        log_x: bool = False,
        log_y: bool = False,
        height: int = 300,
        legend: bool = True,
        legend_left: bool = False,
    ) -> None:
        super().__init__(axisItems={"left": CleanAxis("left"), "bottom": CleanAxis("bottom")})
        self.setMinimumHeight(height)
        self.setMaximumHeight(int(height * 1.6))
        item = self.getPlotItem()
        item.showGrid(x=True, y=True, alpha=0.08)
        item.setLogMode(log_x, log_y)
        self._log = (log_x, log_y)
        for name in ("left", "bottom"):
            axis = item.getAxis(name)
            axis.setPen(pg.mkPen(theme.BORDER))
            axis.setTextPen(pg.mkPen(theme.MUTED))
            axis.enableAutoSIPrefix(False)  # "(×0.001)" on an axis is easy to misread
        item.setLabel("bottom", x_label, color=theme.MUTED)
        item.setLabel("left", y_label, color=theme.MUTED)
        item.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        if legend:
            # Top right by default; top left for plots whose data rise to the right.
            offset = (70, 10) if legend_left else (-10, 10)
            self.legend = item.addLegend(offset=offset, labelTextColor=theme.MUTED)
            self.legend.setBrush(pg.mkBrush(QColor(theme.BACKGROUND)))
        self._series: list[_Series] = []
        self._readout = pg.TextItem(
            "", color=theme.TEXT, anchor=(0, 1), fill=pg.mkBrush(theme.RAISED)
        )
        self._readout.setZValue(100)
        self._marker = pg.ScatterPlotItem(
            size=9, pen=pg.mkPen(theme.TEXT, width=1.2), brush=pg.mkBrush(None)
        )
        self._marker.setZValue(99)
        item.addItem(self._readout, ignoreBounds=True)
        item.addItem(self._marker, ignoreBounds=True)
        self._readout.hide()
        self.scene().sigMouseMoved.connect(self._hover)

    # ----------------------------------------------------------------------- content --

    def line(
        self,
        x,
        y,
        name: str | None = None,
        colour: str = theme.ACCENT,
        width: float = 1.6,
        dashed: bool = False,
        tips: Sequence[str] | None = None,
    ) -> pg.PlotDataItem:
        pen = pg.mkPen(
            colour, width=width, style=Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        )
        item = self.plot(np.asarray(x, float), np.asarray(y, float), pen=pen, name=name)
        self._track(name, x, y, tips)
        return item

    def points(
        self,
        x,
        y,
        name: str | None = None,
        colour: str = theme.ACCENT,
        size: float = 6,
        symbol: str = "o",
        tips: Sequence[str] | None = None,
        alpha: int = 230,
    ) -> pg.PlotDataItem:
        c = QColor(colour)
        c.setAlpha(alpha)
        item = self.plot(
            np.asarray(x, float),
            np.asarray(y, float),
            pen=None,
            symbol=symbol,
            symbolSize=size,
            symbolBrush=pg.mkBrush(c),
            symbolPen=None,
            name=name,
        )
        self._track(name, x, y, tips)
        return item

    def errorbars(self, x, y, err, colour: str = theme.ACCENT) -> None:
        """Vertical ±err bars, drawn as line segments so they follow log axes like the data."""
        x, y, err = (np.asarray(v, float) for v in (x, y, err))
        good = np.isfinite(err) & np.isfinite(x) & np.isfinite(y)
        if not good.any():
            return
        lo = y[good] - err[good]
        if self._log[1]:
            lo = np.maximum(lo, y[good] * 1e-3)
        xs = np.repeat(x[good], 2)
        ys = np.column_stack([lo, y[good] + err[good]]).ravel()
        self.plot(xs, ys, pen=pg.mkPen(colour, width=1), connect="pairs")

    def band(self, x, lo, hi, colour: str = theme.ACCENT, alpha: int = 40) -> None:
        c = QColor(colour)
        c.setAlpha(alpha)
        top = self.plot(np.asarray(x, float), np.asarray(hi, float), pen=None)
        bottom = self.plot(np.asarray(x, float), np.asarray(lo, float), pen=None)
        self.addItem(pg.FillBetweenItem(top, bottom, brush=pg.mkBrush(c)))

    def hline(self, value: float, label: str = "", colour: str = theme.FAINT) -> None:
        y = math.log10(value) if self._log[1] and value > 0 else value
        line = pg.InfiniteLine(
            pos=y,
            angle=0,
            pen=pg.mkPen(colour, width=1, style=Qt.PenStyle.DashLine),
            label=label,
            labelOpts={"color": colour, "position": 0.95, "anchors": [(1, 1), (1, 1)]},
        )
        self.addItem(line)

    def vline(self, value: float, label: str = "", colour: str = theme.FAINT) -> None:
        x = math.log10(value) if self._log[0] and value > 0 else value
        line = pg.InfiniteLine(
            pos=x,
            angle=90,
            pen=pg.mkPen(colour, width=1, style=Qt.PenStyle.DashLine),
            label=label,
            labelOpts={"color": colour, "position": 0.92, "anchors": [(0, 0), (0, 0)]},
        )
        self.addItem(line)

    def _track(self, name, x, y, tips) -> None:
        self._series.append(
            _Series(name or "", np.asarray(x, float).ravel(), np.asarray(y, float).ravel(), tips)
        )

    # -------------------------------------------------------------------------- hover --

    def _to_view(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(divide="ignore", invalid="ignore"):
            vx = np.log10(x) if self._log[0] else x
            vy = np.log10(y) if self._log[1] else y
        return vx, vy

    def _hover(self, pos: QPointF) -> None:
        vb = self.getPlotItem().getViewBox()
        if not self.sceneBoundingRect().contains(pos) or not self._series:
            self._readout.hide()
            self._marker.clear()
            return
        # Distances in screen pixels, computed in view coordinates scaled by the pixel size.
        mouse = vb.mapSceneToView(pos)
        (x0, x1), (y0, y1) = vb.viewRange()
        px = (x1 - x0) / max(vb.width(), 1)
        py = (y1 - y0) / max(vb.height(), 1)
        best = None
        for s in self._series:
            vx, vy = self._to_view(s.x, s.y)
            d = ((vx - mouse.x()) / px) ** 2 + ((vy - mouse.y()) / py) ** 2
            d = np.where(np.isfinite(d), d, np.inf)
            if d.size == 0:
                continue
            k = int(np.argmin(d))
            if best is None or d[k] < best[0]:
                best = (d[k], s, k)
        if best is None or best[0] > 30**2:
            self._readout.hide()
            self._marker.clear()
            return
        _, s, i = best
        text = s.tips[i] if s.tips is not None else f"{_fmt(s.x[i])},  {_fmt(s.y[i])}"
        if s.name and s.tips is None:
            text = f"{s.name}\n{text}"
        vx, vy = self._to_view(np.array([s.x[i]]), np.array([s.y[i]]))
        self._marker.setData([vx[0]], [vy[0]])
        self._readout.setText(text)
        self._readout.setPos(vx[0], vy[0])
        self._readout.show()


def set_ticks(plot: Plot, axis: str, ticks: list[tuple[float, str]]) -> None:
    plot.getPlotItem().getAxis(axis).setTicks([ticks])
