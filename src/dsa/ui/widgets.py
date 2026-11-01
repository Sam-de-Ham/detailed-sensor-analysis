"""Building blocks of the result pages: key figures, captioned figures, notes, tables.

Pages read like a short document: a title and one-line lead, the key numbers, then figures with
captions, then what limits the result and what data would improve it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dsa.analysis.common import Level, Section, Status, Value

STATUS_TEXT = {
    Status.MEASURED: "MEASURED",
    Status.LIMITED: "LIMITED BY THE DATA",
    Status.UNAVAILABLE: "NOT AVAILABLE FROM THIS DATA",
}


def label(text: str, name: str | None = None, wrap: bool = True) -> QLabel:
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


def rule() -> QFrame:
    line = QFrame()
    line.setObjectName("rule")
    line.setFixedHeight(1)
    return line


def _restyle(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class KeyFigure(QFrame):
    """One headline number: label, value ± error and unit, and a short note. Clickable."""

    clicked = pyqtSignal()

    def __init__(self, title: str, value: Value | str, note: str = "") -> None:
        super().__init__()
        self.setObjectName("keyFigure")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 8, 12, 6)
        box.setSpacing(2)
        box.addWidget(label(title, "keyLabel"))
        row = QHBoxLayout()
        row.setSpacing(6)
        if isinstance(value, Value):
            text = value.format() if value.ok else "—"
            unit = value.unit
            if unit and text.endswith(" " + unit):
                text = text[: -len(unit) - 1]
        else:
            text, unit = value, ""
        val = label(text, "keyValue", wrap=False)
        row.addWidget(val)
        if unit:
            u = label(unit, "keyUnit", wrap=False)
            row.addWidget(u, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        box.addLayout(row)
        if note:
            box.addWidget(label(note, "keyNote"))

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)


class FigureBlock(QWidget):
    """A titled figure with a caption."""

    def __init__(
        self, title: str, caption: str, content: QWidget, tools: Sequence[QWidget] = ()
    ) -> None:
        super().__init__()
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 18, 0, 6)
        box.setSpacing(4)
        head = QHBoxLayout()
        head.addWidget(label(title, "figureTitle"), 1)
        for tool in tools:
            head.addWidget(tool)
        box.addLayout(head)
        if caption:
            box.addWidget(label(caption, "caption"))
        box.addSpacing(4)
        box.addWidget(content)


class Collapsible(QWidget):
    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton(f"{title} ›")
        self.button.setObjectName("link")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.body = label(text, "muted")
        self.body.setVisible(False)
        self._title = title
        self.button.clicked.connect(self._toggle)
        box.addWidget(self.button, 0, Qt.AlignmentFlag.AlignLeft)
        box.addWidget(self.body)

    def _toggle(self) -> None:
        visible = not self.body.isVisible()
        self.body.setVisible(visible)
        self.button.setText(f"{self._title} {'‹' if visible else '›'}")


def note_widget(level: Level, text: str) -> QWidget:
    widget = label(text)
    widget.setObjectName("note")
    widget.setProperty("level", level.value)
    _restyle(widget)
    return widget


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    numeric: Sequence[int] = (),
    height: int | None = None,
) -> QTableWidget:
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(list(headers))
    t.verticalHeader().setVisible(False)
    t.setShowGrid(False)
    t.setAlternatingRowColors(True)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            item = QTableWidgetItem(str(text))
            if c in numeric:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t.setItem(r, c, item)
    header = t.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setStretchLastSection(True)
    rows_h = sum(t.rowHeight(i) for i in range(len(rows)))
    t.setFixedHeight(height or min(rows_h + header.height() + 6, 420))
    return t


class Document(QWidget):
    """A vertical page of content with a comfortable reading width."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("page")
        outer = QHBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 40)
        column = QWidget()
        column.setMaximumWidth(1240)
        self.box = QVBoxLayout(column)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.setSpacing(6)
        outer.addWidget(column)
        self.anchors: dict[str, QWidget] = {}

    def add(self, widget: QWidget) -> QWidget:
        self.box.addWidget(widget)
        return widget

    def heading(self, title: str, lead: str = "") -> None:
        self.add(label(title, "pageTitle"))
        if lead:
            self.add(label(lead, "pageLead"))
        self.box.addSpacing(6)

    def status(self, section: Section) -> None:
        badge = label(STATUS_TEXT[section.status], "statusBadge", wrap=False)
        badge.setProperty("status", section.status.value)
        _restyle(badge)
        self.add(badge)

    def keys(self, figures: Sequence[tuple[KeyFigure, str | None]], columns: int = 4) -> None:
        """Key figures in a grid; each scrolls to the anchor it names when clicked."""
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        for i, (figure, anchor) in enumerate(figures):
            grid.addWidget(figure, i // columns, i % columns)
            if anchor:
                figure.clicked.connect(lambda a=anchor: self.scroll_to(a))
        holder = QWidget()
        holder.setLayout(grid)
        self.box.addSpacing(8)
        self.add(holder)
        self.box.addSpacing(8)

    def figure(
        self,
        title: str,
        caption: str,
        content: QWidget,
        anchor: str | None = None,
        tools: Sequence[QWidget] = (),
    ) -> FigureBlock:
        block = FigureBlock(title, caption, content, tools)
        self.add(block)
        if anchor:
            self.anchors[anchor] = block
        return block

    def subheading(self, text: str) -> None:
        self.box.addSpacing(14)
        self.add(label(text.upper(), "sectionLabel"))
        self.add(rule())

    def text(self, text: str, name: str | None = "muted") -> None:
        self.add(label(text, name))

    def notes(self, section: Section) -> None:
        if section.notes:
            self.subheading("Notes")
            for note in section.notes:
                self.add(note_widget(note.level, note.text))
        if section.method:
            self.box.addSpacing(10)
            self.add(Collapsible("How this was measured", section.method))
        if section.unlock:
            self.subheading("To get more from your data")
            for item in dict.fromkeys(section.unlock):
                self.add(note_widget(Level.INFO, item))

    def finish(self) -> None:
        self.box.addStretch(1)

    def scroll_to(self, anchor: str) -> None:
        target = self.anchors.get(anchor)
        scroll = self._scroll_area()
        if target is not None and scroll is not None:
            QTimer.singleShot(0, lambda: scroll.verticalScrollBar().setValue(target.y()))

    def _scroll_area(self) -> QScrollArea | None:
        parent = self.parent()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parent()
        return parent


class Page(QScrollArea):
    """A scrollable result page, rebuilt whenever the shown camera or setting changes."""

    title = ""

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.doc = Document()
        self.setWidget(self.doc)

    def new_document(self) -> Document:
        self.doc = Document()
        self.setWidget(self.doc)
        return self.doc

    def message(self, title: str, text: str) -> None:
        doc = self.new_document()
        doc.heading(title, text)
        doc.finish()


def link_button(text: str, slot: Callable[[], None]) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("link")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.clicked.connect(slot)
    return button
