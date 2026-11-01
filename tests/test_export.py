"""Exports: the HTML report and JSON are complete and self-contained."""

from __future__ import annotations

import json

from dsa.cli import main
from dsa.export import write_json
from dsa.report import write_report


def test_html_report(mono, tmp_path):
    path = write_report(mono.report, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.count("<img") >= 6  # figures are embedded, not linked
    assert "Photon transfer curve" in text
    assert mono.mode.gain.gain.format() in text


def test_json_round_trip(mono, tmp_path):
    data = json.loads(write_json(mono.report, tmp_path / "r.json").read_text(encoding="utf-8"))
    mode = data["report"]["modes"][0]
    assert abs(mode["gain"]["gain"]["value"] - mono.mode.gain.gain.value) < 1e-12
    assert mode["dark"]["map"] == {"omitted_array": [mono.model.height, mono.model.width]}


def test_cli_scan_and_analyze(mono, tmp_path, capsys):
    assert main(["scan", str(mono.root), "--cache", str(tmp_path)]) == 0
    assert "SIMCAM" in capsys.readouterr().out
    assert main(["analyze", str(mono.root), "--no-cache", "--report", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Gain" in out and "report:" in out
