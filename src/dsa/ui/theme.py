"""A calm dark theme: one accent, muted series colours, typography doing the work."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

BACKGROUND = "#121417"
SURFACE = "#171a1e"
RAISED = "#1f2328"
BORDER = "#2a2f36"
TEXT = "#e4e6e9"
MUTED = "#8b929b"
FAINT = "#5c636c"
ACCENT = "#6aa5e0"

OK = "#6fbf8f"
INFO = "#8b929b"
WARN = "#d9a54a"
PROBLEM = "#e06c6c"

# Series colours: distinguishable, desaturated, readable on the dark background.
SERIES = ["#6aa5e0", "#d9a54a", "#5fbfb0", "#d97a9b", "#a68fe0", "#b9c26a", "#e0896a"]
PLANE_COLOURS = {"L": "#c9ced6", "R": "#e07a7a", "G1": "#79c28f", "G2": "#a9d48a", "B": "#7aa6e0"}

FONTS = ["Segoe UI", "Inter", "Helvetica Neue", "Noto Sans", "DejaVu Sans"]
MONO_FONTS = ["Cascadia Mono", "Consolas", "JetBrains Mono", "Menlo", "DejaVu Sans Mono"]

STYLESHEET = f"""
QWidget {{ color: {TEXT}; font-size: 10pt; }}
QMainWindow, QDialog, QScrollArea, #page {{ background: {BACKGROUND}; }}
QScrollArea {{ border: none; }}
#sidebar {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
#brand {{ font-size: 13pt; font-weight: 600; letter-spacing: 0.5px; }}
#brandSub, #muted, #caption {{ color: {MUTED}; }}
#faint {{ color: {FAINT}; }}
#sectionLabel {{ color: {FAINT}; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }}
#nav {{ background: transparent; border: none; outline: 0; }}
#nav::item {{ padding: 6px 12px; border-radius: 4px; margin: 1px 6px; color: {MUTED}; }}
#nav::item:selected {{ background: {RAISED}; color: {TEXT}; }}
#nav::item:hover:!selected {{ color: {TEXT}; }}
#nav::item:disabled {{ color: {FAINT}; }}
#pageTitle {{ font-size: 18pt; font-weight: 600; }}
#pageLead {{ color: {MUTED}; font-size: 10.5pt; }}
#figureTitle {{ font-size: 11pt; font-weight: 600; }}
#keyLabel {{ color: {MUTED}; font-size: 8.5pt; }}
#keyValue {{ font-size: 17pt; font-weight: 500; }}
#keyUnit {{ color: {MUTED}; font-size: 10pt; }}
#keyNote {{ color: {FAINT}; font-size: 8.5pt; }}
#keyFigure {{ border-top: 1px solid {BORDER}; }}
#keyFigure:hover {{ border-top: 1px solid {ACCENT}; }}
#rule {{ background: {BORDER}; }}
#note {{ border-left: 2px solid {INFO}; padding: 2px 0 2px 10px; }}
#note[level="warn"] {{ border-left-color: {WARN}; }}
#note[level="problem"] {{ border-left-color: {PROBLEM}; }}
#note[level="ok"] {{ border-left-color: {OK}; }}
#statusBadge {{ color: {MUTED}; font-size: 8.5pt; font-weight: 600; letter-spacing: 0.5px; }}
#statusBadge[status="measured"] {{ color: {OK}; }}
#statusBadge[status="limited"] {{ color: {WARN}; }}
#statusBadge[status="unavailable"] {{ color: {FAINT}; }}
QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 4px;
               padding: 5px 12px; }}
QPushButton:hover {{ border-color: {MUTED}; }}
QPushButton:disabled {{ color: {FAINT}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: #0d1117;
                       font-weight: 600; }}
QPushButton#primary:disabled {{ background: {RAISED}; border-color: {BORDER}; color: {FAINT}; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT}; padding: 0; }}
QPushButton#link:hover {{ text-decoration: underline; }}
QToolButton {{ background: transparent; border: none; color: {MUTED}; padding: 2px 6px; }}
QToolButton:hover {{ color: {TEXT}; }}
QToolButton:checked {{ color: {TEXT}; border-bottom: 1px solid {ACCENT}; }}
QComboBox, QSpinBox, QLineEdit {{ background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 4px; padding: 3px 6px; }}
QComboBox:hover, QSpinBox:hover {{ border-color: {MUTED}; }}
QComboBox QAbstractItemView {{ background: {RAISED}; border: 1px solid {BORDER};
    selection-background-color: {BORDER}; }}
QTreeWidget, QTableWidget, QTableView, QListWidget {{ background: {BACKGROUND};
    alternate-background-color: {SURFACE}; border: none; gridline-color: {BORDER}; }}
QTreeWidget::item, QTableWidget::item {{ padding: 3px 4px; }}
QTreeWidget::item:selected, QTableWidget::item:selected {{ background: {RAISED}; color: {TEXT}; }}
QHeaderView::section {{ background: {BACKGROUND}; border: none; border-bottom: 1px solid {BORDER};
    padding: 4px 6px; color: {MUTED}; font-weight: 600; }}
QProgressBar {{ background: {SURFACE}; border: none; border-radius: 2px; max-height: 4px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {FAINT}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}
QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 4px 18px; border-radius: 3px; }}
QMenu::item:selected {{ background: {BORDER}; }}
QSplitter::handle {{ background: {BORDER}; }}
"""


def mono_font(size: float = 9.5) -> QFont:
    font = QFont()
    font.setFamilies(MONO_FONTS)
    font.setPointSizeF(size)
    return font


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    font = QFont()
    font.setFamilies(FONTS)
    font.setPointSize(10)
    app.setFont(font)
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BACKGROUND,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: BACKGROUND,
        QPalette.ColorRole.AlternateBase: SURFACE,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: RAISED,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.Highlight: BORDER,
        QPalette.ColorRole.HighlightedText: TEXT,
        QPalette.ColorRole.ToolTipBase: RAISED,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.PlaceholderText: FAINT,
        QPalette.ColorRole.Link: ACCENT,
    }
    for role, colour in roles.items():
        palette.setColor(role, QColor(colour))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
    pg.setConfigOptions(
        antialias=True, background=BACKGROUND, foreground=MUTED, imageAxisOrder="row-major"
    )
