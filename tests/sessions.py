"""Synthetic capture sessions with known sensor parameters, analysed end to end."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dsa.analysis.run import CameraReport, ModeReport, Settings, analyse_camera
from dsa.catalog import Catalog, scan
from dsa.measure import StatsCache
from dsa.synthetic import Dust, SensorModel, SyntheticSensor, write_session


@dataclass
class Session:
    sensor: SyntheticSensor
    root: Path
    catalog: Catalog
    report: CameraReport

    @property
    def model(self) -> SensorModel:
        return self.sensor.model

    @property
    def mode(self) -> ModeReport:
        assert len(self.report.modes) == 1
        return self.report.modes[0]


def make_session(root: Path, model: SensorModel, **session) -> Session:
    sensor = SyntheticSensor(model)
    write_session(sensor, root, **session)
    catalog = scan([root])
    report = analyse_camera(catalog, catalog.cameras[0], StatsCache(None), Settings())
    return Session(sensor, root, catalog, report)


MONO = SensorModel(
    width=480,
    height=360,
    gain=1.0,
    offset=60,
    read_noise=1.2,
    row_noise=0.15,
    dark_current=0.02,
    dark_ref_temp=-10,
    hot_fraction=3e-4,
    amp_glow=0.5,
    prnu=0.008,
    vignetting=0.35,
    dust=(Dust(150, 120, 28, 0.03),),
    rts_fraction=1e-3,
    rts_amplitude=8.0,
)
