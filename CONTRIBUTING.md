# Contributing

Bug reports, sample frames and ideas are welcome — please use the issue templates. For anything
about reading or classifying files, the output of `dsa scan` on the folder (or one affected
file's FITS header) usually makes the problem reproducible.

## Development setup

```bash
git clone https://github.com/Sam-de-Ham/detailed-sensor-analysis
cd detailed-sensor-analysis
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Dependencies are managed with [uv](https://docs.astral.sh/uv/): add them with `uv add`.

## How the code is organised

See [docs/architecture.md](docs/architecture.md). In short: `frames.py` and `catalog.py` find
and describe files, `measure.py` reads each frame set once into summary statistics, the modules
in `analysis/` turn those into results, and `ui/` and `report.py` present them. The analyses
don't import Qt.

## Accuracy first

Every analysis needs a test that recovers a known value from synthetic frames
(`dsa.synthetic` generates them from a sensor model with known gain, noise, dark current,
defects and flat-field structure; `tests/sessions.py` runs them through the whole pipeline).
When you change a method, check it against real frames too and note anything surprising.

## Screenshots

```bash
uv run python scripts/screenshots.py <folder of frames> <output folder> --camera <name> --figures
```

renders every page (and with `--figures`, every figure) of the app to PNG.

## Pull requests

- Keep them focused, with tests, and run the checks above.
- Note user-visible changes in `CHANGELOG.md` under an `## Unreleased` heading.

## Releases

Set the version in `src/dsa/__init__.py`, move the changelog entries under it, then tag and
push (`git tag v1.2.3 && git push origin v1.2.3`). The release workflow runs the tests, builds
the wheel and source package, and publishes the GitHub release with the changelog notes.
