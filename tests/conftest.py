"""Shared fixtures: synthetic sessions, generated once per test run."""

from __future__ import annotations

import pytest

from dsa.synthetic import colour_model
from tests.sessions import MONO, Session, make_session


@pytest.fixture(scope="session")
def mono(tmp_path_factory: pytest.TempPathFactory) -> Session:
    """A cooled 12-bit mono camera stored ×16, with every calibration frame type."""
    return make_session(
        tmp_path_factory.mktemp("mono"),
        MONO,
        bias=20,
        darks={60: 10, 300: 10},
        dark_temps=(0.0, 10.0),
        flats={"L": (1500, 12), "Ha": (1600, 12)},
        lights={"L": (60, 4, 0.8)},
    )


@pytest.fixture(scope="session")
def colour(tmp_path_factory: pytest.TempPathFactory) -> Session:
    """A DSLR-like 14-bit Bayer camera."""
    model = colour_model(width=480, height=360, prnu=0.006, vignetting=0.4)
    return make_session(
        tmp_path_factory.mktemp("colour"),
        model,
        bias=16,
        darks={120: 8},
        flats={"": (6000, 12)},
        temperature=20.0,
    )
