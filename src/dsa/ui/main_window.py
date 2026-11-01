"""The main window: sidebar navigation, background scanning and analysis, exports."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from dsa import __version__
from dsa.analysis.run import CameraReport, Settings, analyse_camera
from dsa.catalog import Catalog, FrameSet, scan
from dsa.measure import StatsCache
from dsa.paths import default_cache_dir
from dsa.ui.pages import Context
from dsa.ui.pages.checks import ChecksPage, SpecsPage
from dsa.ui.pages.dark import DarkPage
from dsa.ui.pages.exposure import ExposurePage
from dsa.ui.pages.flat import FlatPage
from dsa.ui.pages.frames import FramesPage
from dsa.ui.pages.gain import GainPage
from dsa.ui.pages.library import LibraryPage
from dsa.ui.pages.noise import PatternNoisePage, ReadNoisePage
from dsa.ui.pages.overview import OverviewPage
from dsa.ui.pages.pixels import PixelsPage
from dsa.ui.tasks import Task, shutdown_tasks
from dsa.ui.widgets import label

PAGE_ORDER = [
    ("overview", "Overview"),
    ("noise", "Read noise"),
    ("pattern", "Pattern noise"),
    ("gain", "Gain & photon transfer"),
    ("dark", "Dark current"),
    ("pixels", "Defective pixels"),
    ("flat", "Flat field"),
    ("exposure", "Sub-exposure"),
    ("checks", "Data checks"),
    ("specs", "Published specs"),
    ("frames", "Frames"),
]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Detailed Sensor Analysis")
        self.resize(1480, 960)
        self.settings = QSettings("DetailedSensorAnalysis", "DSA")
        self.cache_dir = default_cache_dir()
        self.cache = StatsCache(self.cache_dir)
        self.catalog: Catalog | None = None
        self.reports: dict[str, CameraReport] = {}
        self.task: Task | None = None
        self._max_frames_used = 50

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._sidebar())
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        self.library = LibraryPage()
        self.library.scan_requested.connect(self.start_scan)
        self.library.analyse_requested.connect(self.start_analysis)
        self.stack.addWidget(self.library)
        self.pages = {
            "overview": OverviewPage(),
            "noise": ReadNoisePage(),
            "pattern": PatternNoisePage(),
            "gain": GainPage(),
            "dark": DarkPage(),
            "pixels": PixelsPage(),
            "flat": FlatPage(),
            "exposure": ExposurePage(),
            "checks": ChecksPage(),
            "specs": SpecsPage(),
            "frames": FramesPage(self._load_stats),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)
        self._built: set[str] = set()

        folders = [Path(p) for p in (self.settings.value("folders", [], list) or [])]
        self.library.set_folders(folders)
        self._update_navigation()

    # ----------------------------------------------------------------------- sidebar --

    def _sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(240)
        box = QVBoxLayout(side)
        box.setContentsMargins(14, 18, 14, 14)
        box.setSpacing(6)
        box.addWidget(label("DSA", "brand", wrap=False))
        box.addWidget(label("Detailed Sensor Analysis", "brandSub", wrap=False))
        box.addSpacing(14)
        self.library_button = QPushButton("Library")
        self.library_button.clicked.connect(self.show_library)
        box.addWidget(self.library_button)
        box.addSpacing(12)
        box.addWidget(label("CAMERA", "sectionLabel", wrap=False))
        self.camera_combo = QComboBox()
        self.camera_combo.currentTextChanged.connect(self._camera_changed)
        box.addWidget(self.camera_combo)
        box.addSpacing(8)
        box.addWidget(label("SETTING", "sectionLabel", wrap=False))
        self.mode_list = QListWidget()
        self.mode_list.setObjectName("nav")
        self.mode_list.setMaximumHeight(140)
        self.mode_list.setWordWrap(True)
        self.mode_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.mode_list.currentRowChanged.connect(self._mode_changed)
        box.addWidget(self.mode_list)
        box.addSpacing(8)
        box.addWidget(label("RESULTS", "sectionLabel", wrap=False))
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        for key, title in PAGE_ORDER:
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.nav.addItem(item)
        self.nav.currentItemChanged.connect(self._nav_changed)
        box.addWidget(self.nav, 1)
        self.export_button = QPushButton("Export")
        menu = QMenu(self.export_button)
        menu.addAction("HTML report…", self.export_report)
        menu.addAction("Results as JSON…", self.export_json)
        self.export_button.setMenu(menu)
        box.addWidget(self.export_button)
        self.status = label("", "faint")
        box.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()
        box.addWidget(self.progress)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self._cancel)
        box.addWidget(self.cancel_button)
        box.addWidget(label(f"v{__version__}", "faint", wrap=False))
        return side

    def _update_navigation(self) -> None:
        has = bool(self.reports)
        self.nav.setEnabled(has)
        self.mode_list.setEnabled(has)
        self.camera_combo.setEnabled(has)
        self.export_button.setEnabled(has)

    # ----------------------------------------------------------------------- tasks --

    def _run(self, fn, done, text: str) -> None:
        if self.task is not None and self.task.isRunning():
            return
        self.task = Task(fn, self)
        self.task.progress.connect(self._on_progress)
        self.task.succeeded.connect(done)
        self.task.failed.connect(self._on_failed)
        self.task.cancelled.connect(lambda: self._finish_task("Cancelled"))
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.show()
        self.status.setText(text)
        self.task.start()

    def _on_progress(self, fraction: float, text: str) -> None:
        self.progress.setValue(int(fraction * 1000))
        self.status.setText(text[:60])

    def _on_failed(self, message: str) -> None:
        self._finish_task("Failed")
        QMessageBox.warning(self, "Detailed Sensor Analysis", message)

    def _finish_task(self, text: str = "") -> None:
        self.progress.hide()
        self.cancel_button.hide()
        self.status.setText(text)

    def _cancel(self) -> None:
        if self.task is not None:
            self.task.cancel()

    def start_scan(self) -> None:
        folders = list(self.library.folders)
        self.settings.setValue("folders", [str(f) for f in folders])
        cache = self.cache_dir / "metadata.json"
        self._run(
            lambda progress, cancel: scan(folders, cache, progress, cancel),
            self._scanned,
            "Scanning…",
        )

    def _scanned(self, catalog: Catalog) -> None:
        self._finish_task(f"{len(catalog.frames):,} frames found")
        self.catalog = catalog
        self.library.catalog = catalog
        self.library.render()

    def start_analysis(self, cameras: list[str]) -> None:
        if self.catalog is None:
            return
        catalog = self.catalog
        settings = Settings(max_frames=self.library.max_frames, rts=self.library.rts_enabled)
        self._max_frames_used = settings.max_frames

        def work(progress, cancel):
            out = {}
            for i, camera in enumerate(cameras):

                def sub(f, t, i=i, camera=camera):
                    progress((i + f) / len(cameras), f"{camera}: {t}")

                out[camera] = analyse_camera(catalog, camera, self.cache, settings, sub, cancel)
            return out

        self._run(work, self._analysed, "Analysing…")

    def _analysed(self, reports: dict[str, CameraReport]) -> None:
        self._finish_task("Analysis complete")
        self.reports.update(reports)
        self._update_navigation()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItems(list(self.reports))
        self.camera_combo.blockSignals(False)
        first = next(iter(reports))
        self.camera_combo.setCurrentText(first)
        self._camera_changed(first)
        self.goto("overview")

    # ------------------------------------------------------------------ navigation --

    @property
    def report(self) -> CameraReport | None:
        return self.reports.get(self.camera_combo.currentText())

    def _camera_changed(self, camera: str) -> None:
        report = self.reports.get(camera)
        self.mode_list.blockSignals(True)
        self.mode_list.clear()
        if report is not None:
            for m in report.modes:
                self.mode_list.addItem(m.mode.setting)
        self.mode_list.blockSignals(False)
        if report is not None and report.modes:
            # Start on the setting with the most complete data.
            best = max(range(len(report.modes)), key=lambda i: _completeness(report.modes[i]))
            self.mode_list.setCurrentRow(best)
        self._invalidate()

    def _mode_changed(self, _row: int) -> None:
        self._invalidate()

    def _invalidate(self) -> None:
        self._built.clear()
        current = self.nav.currentItem()
        if current is not None and self.stack.currentWidget() is not self.library:
            self._show(current.data(Qt.ItemDataRole.UserRole))

    def _nav_changed(self, item: QListWidgetItem | None, _previous) -> None:
        if item is not None:
            self._show(item.data(Qt.ItemDataRole.UserRole))

    def _context(self) -> Context | None:
        report = self.report
        row = self.mode_list.currentRow()
        if report is None or not 0 <= row < len(report.modes):
            return None
        return Context(report, report.modes[row], self.goto, self.open_set)

    def _show(self, key: str, **kwargs) -> None:
        ctx = self._context()
        page = self.pages[key]
        if ctx is None:
            page.message(page.title, "Scan a library and analyse a camera first.")
        elif key not in self._built or kwargs:
            page.build(ctx, **kwargs)
            self._built.add(key)
        self.stack.setCurrentWidget(page)

    def goto(self, key: str) -> None:
        for i in range(self.nav.count()):
            if self.nav.item(i).data(Qt.ItemDataRole.UserRole) == key:
                self.nav.setCurrentRow(i)
                return

    def open_set(self, label_text: str) -> None:
        self.goto("frames")
        self._show("frames", select=label_text)

    def show_library(self) -> None:
        self.nav.clearSelection()
        self.nav.setCurrentItem(None)
        self.stack.setCurrentWidget(self.library)

    def _load_stats(self, fs: FrameSet):
        return self.cache.get(fs, self._max_frames_used)

    # ----------------------------------------------------------------------- export --

    def export_report(self) -> None:
        report = self.report
        if report is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Write the report into this folder")
        if not folder:
            return
        from dsa.report import write_report

        path = write_report(report, Path(folder))
        self.status.setText(f"Report: {path.name}")

    def export_json(self) -> None:
        report = self.report
        if report is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", "dsa-results.json", "JSON (*.json)"
        )
        if path:
            from dsa.export import write_json

            write_json(report, Path(path))

    def closeEvent(self, event) -> None:
        self.settings.setValue("folders", [str(f) for f in self.library.folders])
        shutdown_tasks()
        super().closeEvent(event)


def _completeness(report) -> int:
    from dsa.analysis.common import Status

    sections = (report.read_noise, report.gain, report.dark, report.flat, report.exposure)
    return sum({Status.MEASURED: 2, Status.LIMITED: 1}.get(s.status, 0) for s in sections)
