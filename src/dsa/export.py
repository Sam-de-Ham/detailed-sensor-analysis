"""Results as JSON: every number with its uncertainty, small arrays inline, large maps omitted."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from dsa import __version__
from dsa.analysis.common import Value
from dsa.analysis.levels import Offset
from dsa.analysis.run import CameraReport
from dsa.analysis.specs import SensorSpec
from dsa.catalog import FrameSet

MAX_INLINE = 4096  # array elements; bigger arrays (sensor maps) are summarised by their shape


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float | np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Value):
        return {
            "value": to_jsonable(obj.value),
            "error": to_jsonable(obj.error),
            "unit": obj.unit,
            "bound": obj.bound,
        }
    if isinstance(obj, np.ndarray):
        if obj.size > MAX_INLINE:
            return {"omitted_array": list(obj.shape)}
        return to_jsonable(obj.tolist())
    if isinstance(obj, FrameSet):
        return {
            "label": obj.label,
            "kind": obj.kind.value,
            "files": obj.count,
            "folder": str(obj.folder),
            "exposure": to_jsonable(obj.exposure),
            "filter": obj.filter,
            "master": obj.is_master,
            "temperature": to_jsonable(obj.temperature),
        }
    if isinstance(obj, SensorSpec):
        return {"name": obj.name, "sensor": obj.sensor}
    if isinstance(obj, Offset):
        return {
            "source": obj.source,
            "level": to_jsonable(obj.level),
            "error": to_jsonable(obj.error),
            "label": obj.label,
        }
    if dataclasses.is_dataclass(obj):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def write_json(report: CameraReport, path: Path) -> Path:
    data = {
        "generator": f"Detailed Sensor Analysis {__version__}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "report": to_jsonable(report),
    }
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path
