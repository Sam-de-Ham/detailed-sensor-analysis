"""The measured frame sets of one sensor mode, with helpers to pick the right ones."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from dsa.catalog import FrameSet, Mode
from dsa.cfa import CfaLayout
from dsa.frames import FrameType
from dsa.measure import SetStats


@dataclass
class ModeData:
    mode: Mode
    sets: list[FrameSet]
    stats: dict[str, SetStats]  # by FrameSet.key; sets that failed to measure are absent

    @property
    def layout(self) -> CfaLayout:
        return CfaLayout(self.mode.cfa)

    @property
    def planes(self) -> list[str]:
        return list(self.layout.names)

    def all(self, subs: bool | None = None) -> Iterator[tuple[FrameSet, SetStats]]:
        """Measured sets; ``subs=True`` only sub-frame sets, ``False`` only masters."""
        for fs in self.sets:
            st = self.stats.get(fs.key)
            if st is None:
                continue
            if subs is None or subs != fs.is_master:
                yield fs, st

    def of(self, kind: FrameType, subs: bool | None = True) -> list[tuple[FrameSet, SetStats]]:
        return [(fs, st) for fs, st in self.all(subs) if fs.kind is kind]

    def best(self, kind: FrameType, min_frames: int = 2) -> tuple[FrameSet, SetStats] | None:
        """The sub set with the most frames (at least ``min_frames``), else the largest master."""
        subs = [p for p in self.of(kind, subs=True) if p[1].n_frames >= min_frames]
        if subs:
            return max(subs, key=lambda p: p[1].n_frames)
        masters = self.of(kind, subs=False)
        if masters:
            return max(masters, key=lambda p: p[1].n_combined)
        return None
