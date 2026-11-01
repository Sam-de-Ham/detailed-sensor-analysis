"""Library: choose folders, see what was found, start the analysis."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dsa.catalog import Catalog
from dsa.ui.widgets import Collapsible, Page, label


class LibraryPage(Page):
    title = "Library"
    scan_requested = pyqtSignal()
    analyse_requested = pyqtSignal(list)  # camera names

    def __init__(self) -> None:
        super().__init__()
        self.folders: list[Path] = []
        self.catalog: Catalog | None = None
        # Plain values: the widgets are rebuilt with the page.
        self.max_frames = 50
        self.rts_enabled = True
        self.setAcceptDrops(True)
        self.render()

    # ------------------------------------------------------------------------ layout --

    def render(self) -> None:
        doc = self.new_document()
        doc.heading(
            "Library",
            "Point DSA at folders of calibration frames — raw subs, masters, "
            "or both, from any number of nights and cameras. Files are only read, never "
            "changed. Drop folders here or add them below.",
        )
        folders = QListWidget()
        folders.addItems([str(f) for f in self.folders])
        folders.setFixedHeight(max(60, 24 * len(self.folders) + 12))
        self._folder_list = folders
        buttons = QHBoxLayout()
        add = QPushButton("Add folder…")
        add.clicked.connect(self._add_folder)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_folder)
        scan = QPushButton("Scan")
        scan.setObjectName("primary")
        scan.setEnabled(bool(self.folders))
        scan.clicked.connect(self.scan_requested.emit)
        for b in (add, remove):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(scan)
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(folders)
        box.addLayout(buttons)
        doc.figure("Folders", "", holder)

        if self.catalog is not None:
            self._render_catalog()
        doc.finish()

    def _render_catalog(self) -> None:
        doc = self.doc
        cat = self.catalog
        summary = (
            f"{len(cat.frames):,} frames from {len(cat.cameras)} camera"
            f"{'s' if len(cat.cameras) != 1 else ''}. {len(cat.duplicates):,} duplicate "
            f"copies and {len(cat.skipped):,} other files skipped."
        )
        tree = QTreeWidget()
        tree.setHeaderLabels(["Camera / setting / set", "Files", "Folder", "Classified by"])
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(True)
        for camera in cat.cameras:
            cam_item = QTreeWidgetItem([camera, "", "", ""])
            tree.addTopLevelItem(cam_item)
            for mode in cat.modes(camera):
                sets = cat.sets_for(mode)
                mode_item = QTreeWidgetItem([mode.setting, str(sum(s.count for s in sets)), "", ""])
                cam_item.addChild(mode_item)
                for fs in sets:
                    note = fs.frames[0].kind_source
                    if not fs.quantitative:
                        note += " · rescaled master, listed only"
                    mode_item.addChild(
                        QTreeWidgetItem([fs.label, str(fs.count), str(fs.folder), note])
                    )
                mode_item.setExpanded(len(cat.sets) <= 40)  # small libraries: show every set
            cam_item.setExpanded(True)
        tree.setColumnWidth(0, 360)
        tree.setColumnWidth(1, 60)
        tree.setColumnWidth(2, 360)
        tree.setMinimumHeight(420)
        doc.figure("Found", summary, tree)

        options = QWidget()
        row = QHBoxLayout(options)
        row.setContentsMargins(0, 8, 0, 0)
        frames = QSpinBox(minimum=4, maximum=500, value=self.max_frames)
        frames.setToolTip(
            "Frames per set used for statistics (spread evenly over the set). "
            "More frames: smaller uncertainties, longer first analysis."
        )
        frames.valueChanged.connect(lambda v: setattr(self, "max_frames", v))
        rts = QCheckBox("Find RTS pixels (re-reads bias or dark frames)")
        rts.setChecked(self.rts_enabled)
        rts.toggled.connect(lambda v: setattr(self, "rts_enabled", v))
        row.addWidget(QLabel("Frames per set"))
        row.addWidget(frames)
        row.addSpacing(16)
        row.addWidget(rts)
        row.addStretch(1)
        for camera in cat.cameras:
            b = QPushButton(f"Analyse {camera}")
            b.clicked.connect(lambda _=False, c=camera: self.analyse_requested.emit([c]))
            row.addWidget(b)
        if len(cat.cameras) > 1:
            all_button = QPushButton("Analyse all")
            all_button.setObjectName("primary")
            all_button.clicked.connect(lambda: self.analyse_requested.emit(list(cat.cameras)))
            row.addWidget(all_button)
        doc.add(options)
        if cat.skipped:
            lines = "<br>".join(f"{p.name} — {why}" for p, why in cat.skipped[:300])
            doc.add(Collapsible(f"Skipped files ({len(cat.skipped)})", lines))
        if cat.duplicates:
            lines = "<br>".join(f"{a} = {b}" for a, b in cat.duplicates[:300])
            doc.add(Collapsible(f"Duplicate copies ({len(cat.duplicates)})", lines))
        doc.add(
            label(
                "Duplicates are detected by content, so the same frames copied into several "
                "projects are counted once.",
                "faint",
            )
        )

    # ----------------------------------------------------------------------- folders --

    def set_folders(self, folders: list[Path]) -> None:
        self.folders = [f for f in folders if f.is_dir()]
        self.render()

    def _add_folder(self) -> None:
        start = str(self.folders[-1].parent) if self.folders else ""
        path = QFileDialog.getExistingDirectory(self, "Add a folder of frames", start)
        if path:
            self._add(Path(path))

    def _add(self, path: Path) -> None:
        if path not in self.folders:
            self.folders.append(path)
            self.render()

    def _remove_folder(self) -> None:
        rows = sorted({i.row() for i in self._folder_list.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.folders[r]
        self.render()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                self._add(path)
        event.acceptProposedAction()
