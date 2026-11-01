"""Start the desktop application: ``uv run dsa-gui``."""

from __future__ import annotations

import sys
import traceback

from PyQt6.QtWidgets import QApplication, QMessageBox


def install_exception_hook() -> None:
    """Show unexpected errors instead of letting PyQt abort the process.

    PyQt6 calls qFatal for an exception escaping a slot unless sys.excepthook is customised.
    """

    def hook(kind, value, tb) -> None:
        text = "".join(traceback.format_exception(kind, value, tb))
        print(text, file=sys.stderr)
        app = QApplication.instance()
        if app is not None:
            QMessageBox.critical(
                None, "Unexpected error", f"{value}\n\nDetails were printed to the console."
            )

    sys.excepthook = hook


def main() -> int:
    from dsa.ui.main_window import MainWindow
    from dsa.ui.theme import apply_theme

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Detailed Sensor Analysis")
    apply_theme(app)
    install_exception_hook()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
