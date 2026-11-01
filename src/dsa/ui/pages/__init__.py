"""Result pages. Each turns one part of a :class:`ModeReport` into a document of figures."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from dsa.analysis.common import Value
from dsa.analysis.run import CameraReport, ModeReport


@dataclass
class Context:
    camera: CameraReport
    mode: ModeReport
    goto: Callable[[str], None]  # navigate to a page by key
    open_set: Callable[[str], None]  # show the frames of a set (by label)

    @property
    def gain(self) -> float:
        g = self.mode.gain.gain
        return g.value if g.ok else math.nan

    @property
    def electrons(self) -> bool:
        return math.isfinite(self.gain)

    def to_e(self, adu: Value) -> Value:
        """ADU → e⁻ when the gain is known (else unchanged)."""
        if not self.electrons or not adu.ok:
            return adu
        return adu.scaled(self.gain, "e⁻")

    @property
    def unit(self) -> str:
        return "e⁻" if self.electrons else "ADU"


def plane_label(name: str) -> str:
    return {"L": "Mono", "G1": "G1 (red row)", "G2": "G2 (blue row)"}.get(name, name)
