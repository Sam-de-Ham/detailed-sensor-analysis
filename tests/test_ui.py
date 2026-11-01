"""Offscreen UI smoke test: every page builds from a real analysis without errors."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from dsa.ui.main_window import PAGE_ORDER, MainWindow
from dsa.ui.theme import apply_theme


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    apply_theme(instance)
    return instance


@pytest.mark.parametrize("session_name", ["mono", "colour"])
def test_every_page_builds(app, request, session_name, tmp_path):
    session = request.getfixturevalue(session_name)
    window = MainWindow()
    window.cache_dir = tmp_path
    window.catalog = session.catalog
    window.library.catalog = session.catalog
    window.library.render()
    window._analysed({session.report.camera: session.report})
    for key, _ in PAGE_ORDER:
        window.goto(key)
        app.processEvents()
        assert window.stack.currentWidget() is window.pages[key]
    window.open_set(session.mode.sets[0].label)
    app.processEvents()
    window.show_library()
    window.close()


def _wait(app, window, timeout_s: float = 120) -> None:
    import time

    start = time.time()
    while window.task is not None and not window.task.isFinished():
        app.processEvents()
        assert time.time() - start < timeout_s, "task did not finish"
        time.sleep(0.02)
    app.processEvents()


def test_scan_and_analyse_through_the_ui(app, mono, tmp_path):
    from dsa.measure import StatsCache

    window = MainWindow()
    window.cache_dir = tmp_path
    window.cache = StatsCache(tmp_path)
    window.library.set_folders([mono.root])
    window.start_scan()
    _wait(app, window)
    assert window.catalog is not None and window.catalog.cameras == ["SIMCAM"]
    window.start_analysis(["SIMCAM"])
    _wait(app, window)
    assert "SIMCAM" in window.reports
    assert window.stack.currentWidget() is window.pages["overview"]
    window.close()
