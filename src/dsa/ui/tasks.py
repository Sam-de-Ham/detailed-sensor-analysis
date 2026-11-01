"""Run long work (scanning, analysis) off the UI thread.

A :class:`Task` wraps ``fn(progress, cancel)``. Progress arrives as a Qt signal, which Qt
delivers on the UI thread, so widgets are only ever touched from there. (Pattern from
siril-autostack.)
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from dsa.measure import Cancelled


class Task(QThread):
    progress = pyqtSignal(float, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        fn: Callable[[Callable[[float, str], None], threading.Event], Any],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._fn = fn
        self.cancel_event = threading.Event()
        _live.add(self)
        self.finished.connect(lambda: _live.discard(self))

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            result = self._fn(lambda f, t: self.progress.emit(f, t), self.cancel_event)
        except Cancelled:
            self.cancelled.emit()
        except Exception as err:  # surfaced in the UI instead of killing the thread silently
            traceback.print_exc()
            self.failed.emit(str(err) or type(err).__name__)
        else:
            self.succeeded.emit(result)


# Qt aborts the process if a running QThread is destroyed: every task is joined on exit.
_live: set[Task] = set()


def shutdown_tasks(timeout_ms: int = 10_000) -> None:
    for task in list(_live):
        task.cancel()
    for task in list(_live):
        task.wait(timeout_ms)
