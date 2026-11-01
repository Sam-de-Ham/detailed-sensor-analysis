"""Where DSA keeps its own files (never next to the user's data)."""

from __future__ import annotations

import os
from pathlib import Path


def app_dir() -> Path:
    base = (
        os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    )
    return Path(base) / "DetailedSensorAnalysis"


def default_cache_dir() -> Path:
    return app_dir() / "cache"
