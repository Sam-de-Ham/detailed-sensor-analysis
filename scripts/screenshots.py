"""Render every page of the app for one camera to PNG files (for the README and for review).

    uv run python scripts/screenshots.py <data folder> <out dir> [--camera NAME] [--setting TEXT]
        [--full] [--figures]

Uses the normal analysis cache, so after a first analysis it takes seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--camera", default="")
    parser.add_argument("--setting", default="")
    parser.add_argument("--width", type=int, default=1480)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--full", action="store_true", help="capture whole pages, not the window")
    parser.add_argument("--figures", action="store_true", help="also save every figure on its own")
    args = parser.parse_args()
    # A real window (it flashes briefly): Qt's offscreen platform has no fonts on Windows.

    from PyQt6.QtWidgets import QApplication

    from dsa.analysis.run import Settings, analyse_camera
    from dsa.catalog import scan
    from dsa.ui.main_window import PAGE_ORDER, MainWindow
    from dsa.ui.theme import apply_theme
    from dsa.ui.widgets import FigureBlock

    app = QApplication([])
    apply_theme(app)
    window = MainWindow()
    window.resize(args.width, args.height)
    window.show()
    window.library.set_folders([args.data.absolute()])
    catalog = scan([args.data], window.cache_dir / "metadata.json")
    window.catalog = catalog
    window.library.catalog = catalog
    window.library.render()
    args.out.mkdir(parents=True, exist_ok=True)
    window.show_library()
    app.processEvents()
    window.grab().save(str(args.out / "00-library.png"))
    cameras = [c for c in catalog.cameras if args.camera.lower() in c.lower()]
    reports = {c: analyse_camera(catalog, c, window.cache, Settings()) for c in cameras[:1]}
    window._analysed(reports)
    if args.setting:
        for i in range(window.mode_list.count()):
            if args.setting.lower() in window.mode_list.item(i).text().lower():
                window.mode_list.setCurrentRow(i)
    for n, (key, _) in enumerate(PAGE_ORDER, start=1):
        window.goto(key)
        app.processEvents()
        page = window.pages[key]
        if args.full:
            page.widget().grab().save(str(args.out / f"{n:02d}-{key}.png"))
        else:
            window.grab().save(str(args.out / f"{n:02d}-{key}.png"))
        if args.figures:
            for k, block in enumerate(page.widget().findChildren(FigureBlock)):
                block.grab().save(str(args.out / f"fig-{key}-{k:02d}.png"))
    window.close()


if __name__ == "__main__":
    main()
