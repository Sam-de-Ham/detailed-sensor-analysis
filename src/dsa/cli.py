"""Command line: ``dsa scan`` lists what a folder contains, ``dsa analyze`` runs the analyses."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from dsa.analysis.common import Level
from dsa.analysis.run import CameraReport, ModeReport, Settings, analyse_camera, headline
from dsa.catalog import Catalog, scan
from dsa.measure import StatsCache
from dsa.paths import default_cache_dir


def _progress(fraction: float, text: str) -> None:
    sys.stderr.write(f"\r{fraction:6.1%}  {text[:90]:<90}")
    sys.stderr.flush()


def print_catalog(catalog: Catalog) -> None:
    print(
        f"{len(catalog.frames)} frames, {len(catalog.duplicates)} duplicates skipped, "
        f"{len(catalog.skipped)} other files skipped"
    )
    for camera in catalog.cameras:
        print(f"\n{camera}")
        for mode in catalog.modes(camera):
            print(
                f"  {mode.setting}  ({mode.width}×{mode.height}"
                f"{', ' + mode.cfa if mode.cfa else ''})"
            )
            for fs in catalog.sets_for(mode):
                print(f"    {fs.label:<48} {fs.folder}")


def print_mode(report: ModeReport) -> None:
    print(f"\n  {report.mode.setting}")
    for name, value, status in headline(report):
        print(f"    {name:<15} {value.format():<34} {status.value}")
    if report.read_noise.adu:
        planes = ", ".join(
            f"{p} {v.format()}"
            for p, v in zip(report.read_noise.planes, report.read_noise.adu, strict=True)
        )
        print(f"    {'RN per plane':<15} {planes}")
    g = report.gain
    if g.per_set:
        print(f"    {'Gain native':<15} {g.native.format()}   tile PTC: {g.tile_gain.format()}")
        for s in g.per_set:
            print(f"      {s.label:<44} {s.combined.format():<26} level {s.level_fraction:.0%}")
    for p in report.dark.points:
        print(
            f"    dark {p.label:<40} level {p.current.format():<28} "
            f"noise {p.noise_current.format()}"
        )
    if report.dark.glow is not None:
        print(
            f"    glow: {report.dark.glow.location}, "
            f"peak {report.dark.glow.peak:.3g} {report.dark.unit}"
        )
    for s in report.flat.sets:
        vig = s.vignetting
        corner = f"worst corner {vig.corners[vig.worst_corner]:.0%}" if vig else ""
        print(
            f"    flat {s.label:<40} PRNU {s.prnu_combined.format():<20} "
            f"{corner}, dust {len(s.dust)}"
        )
    print("    defects: " + ", ".join(f"{c.label} {c.count}" for c in report.defects.classes))
    for p in report.exposure.points:
        opt = p.optimal.get(0.05, math.nan)
        print(f"    sky {p.label:<40} {p.flux.format()}  5 %-sub {opt:.0f} s")
    for row in report.specs.rows:
        verdict = {True: "agrees", False: "DIFFERS", None: "n/a"}[row.agrees]
        print(
            f"    spec {row.quantity:<12} {row.measured.format():<26} "
            f"published {row.published:.4g} "
            f"(×{row.ratio:.2f}) {verdict}"
        )
    for check in report.checks:
        if check.level in (Level.WARN, Level.PROBLEM):
            print(f"    [{check.level.value}] {check.title}: {check.summary}")
    for section in (report.read_noise, report.gain, report.dark, report.flat, report.exposure):
        for note in section.notes:
            if note.level in (Level.WARN, Level.PROBLEM):
                print(f"    note: {note.text}")
    for failure in report.failures:
        print(f"    FAILED: {failure.text}")


def print_report(report: CameraReport) -> None:
    print(f"\n{report.camera}" + (f"  —  {report.spec.sensor}" if report.spec else ""))
    for mode in report.modes:
        print_mode(mode)


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("folders", nargs="+", type=Path, help="folders of frames (read only)")
    common.add_argument(
        "--cache", type=Path, default=None, help=f"cache folder (default {default_cache_dir()})"
    )
    common.add_argument("--no-cache", action="store_true", help="don't read or write a cache")
    parser = argparse.ArgumentParser(prog="dsa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan", parents=[common], help="list the cameras, settings and frame sets")
    p_an = sub.add_parser("analyze", parents=[common], help="analyse the cameras found")
    p_an.add_argument("--camera", help="only this camera (substring match)")
    p_an.add_argument("--max-frames", type=int, default=50, help="frames per set (default 50)")
    p_an.add_argument("--report", type=Path, help="write an HTML report per camera here")
    p_an.add_argument("--json", type=Path, help="write results as JSON per camera here")
    args = parser.parse_args(argv)

    cache_dir = None if args.no_cache else (args.cache or default_cache_dir())
    catalog = scan(
        args.folders,
        cache_path=cache_dir / "metadata.json" if cache_dir else None,
        progress=_progress,
    )
    sys.stderr.write("\n")
    if args.command == "scan":
        print_catalog(catalog)
        return 0
    cameras = [c for c in catalog.cameras if not args.camera or args.camera.lower() in c.lower()]
    settings = Settings(max_frames=args.max_frames)
    for camera in cameras:
        start = time.time()
        report = analyse_camera(catalog, camera, StatsCache(cache_dir), settings, _progress)
        sys.stderr.write(f"\n{camera}: {time.time() - start:.0f} s\n")
        print_report(report)
        if args.json:
            from dsa.export import write_json

            args.json.mkdir(parents=True, exist_ok=True)
            slug = "".join(ch if ch.isalnum() else "-" for ch in camera).lower()
            print(f"\n  json: {write_json(report, args.json / f'dsa-{slug}.json')}")
        if args.report:
            from dsa.report import write_report

            path = write_report(report, args.report)
            print(f"\n  report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
