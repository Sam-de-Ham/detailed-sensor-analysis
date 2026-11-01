"""Colour filter array (Bayer) layout: split a mosaic into colour planes and back.

Every statistic in DSA is computed per colour plane. On a Bayer sensor the four photosites of a
2x2 cell see different light and can have different black levels, so pooling them would mix
populations (a flat's red and blue planes differ by a factor of two or more).

Planes are named ``R``, ``G1``, ``G2``, ``B``: ``G1`` is the green on the red row and ``G2`` the
green on the blue row. A mono sensor has the single plane ``L``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

MONO = "L"
_VALID = frozenset({"RGGB", "BGGR", "GRBG", "GBRG"})


@dataclass(frozen=True)
class CfaLayout:
    """Where each colour plane sits within the repeating cell. ``pattern=None`` is mono."""

    pattern: str | None = None

    def __post_init__(self) -> None:
        if self.pattern is not None and self.pattern not in _VALID:
            raise ValueError(f"unsupported CFA pattern {self.pattern!r}")

    @property
    def is_mono(self) -> bool:
        return self.pattern is None

    @property
    def step(self) -> int:
        return 1 if self.pattern is None else 2

    @property
    def offsets(self) -> dict[str, tuple[int, int]]:
        """``(row, col)`` of each plane within the cell, in R, G1, G2, B order."""
        if self.pattern is None:
            return {MONO: (0, 0)}
        cells = {
            (0, 0): self.pattern[0],
            (0, 1): self.pattern[1],
            (1, 0): self.pattern[2],
            (1, 1): self.pattern[3],
        }
        red_row = next(pos for pos, colour in cells.items() if colour == "R")[0]
        found: dict[str, tuple[int, int]] = {}
        for pos, colour in cells.items():
            if colour == "G":
                found["G1" if pos[0] == red_row else "G2"] = pos
            else:
                found[colour] = pos
        return {name: found[name] for name in ("R", "G1", "G2", "B")}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.offsets)

    def plane(self, image: np.ndarray, name: str) -> np.ndarray:
        """A strided view of one colour plane (no copy)."""
        dy, dx = self.offsets[name]
        step = self.step
        return image[dy::step, dx::step]

    def planes(self, image: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
        for name in self.names:
            yield name, self.plane(image, name)

    def merge(self, planes: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        """Re-interleave per-plane arrays (e.g. maps) into a full-resolution mosaic."""
        first = next(iter(planes.values()))
        out = np.zeros(shape, dtype=first.dtype)
        for name, data in planes.items():
            dy, dx = self.offsets[name]
            target = out[dy :: self.step, dx :: self.step]
            target[...] = data[: target.shape[0], : target.shape[1]]
        return out

    def flipped_vertically(self, height: int) -> CfaLayout:
        """The layout seen after flipping an image of ``height`` rows upside down."""
        if self.pattern is None or height % 2:  # odd height keeps the row parity
            return self
        return CfaLayout(self.pattern[2:] + self.pattern[:2])


def colour_of(name: str) -> str:
    """The filter colour of a plane name (``G1`` -> ``G``)."""
    return name[0]
