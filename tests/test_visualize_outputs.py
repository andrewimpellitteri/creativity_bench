"""Offline checks for chart output files: PNG plus its SVG vector copy."""

from __future__ import annotations

import json

import pytest

from creativity_bench.visualize import (
    acceptance_curves,
    load_runs,
    plot_acceptance_curves,
    plot_comparison,
)


@pytest.fixture(scope="session")
def agg():
    """Skip chart-rendering tests where matplotlib (Agg) is unavailable."""
    matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
    matplotlib.use("Agg")
    return matplotlib


def run_payload(model="a", seed=0, score=0.5):
    return {
        "schema_version": 2,
        "model": model,
        "provider": "fixture",
        "seed": seed,
        "composite": score,
        "scores": {"same_but_different": score},
        "weights": {"same_but_different": 1},
        "metadata": {
            "protocol_version": "0.4-validity",
            "protocol_fingerprint": "fixture-fingerprint",
            "selected_tasks": ["same_but_different"],
            "task_sizes": {"n": 2},
            "fast": True,
            "judge_model": "fixture-judge",
            "judge_provider": "fixture",
            "embed_model": None,
            "embed_provider": None,
            "generation_provider": "fixture",
            "generation_base_url": None,
            "judge_base_url": None,
            "embed_base_url": None,
            "generation_settings": {"policy": "test"},
            "judge_settings": {"policy": "test"},
            "evaluation_complete": True,
        },
    }


@pytest.fixture()
def fixture_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "a.json").write_text(json.dumps(run_payload()))
    (runs_dir / "b.json").write_text(json.dumps(run_payload("b", score=0.75)))
    return runs_dir


def test_png_output_also_writes_svg_copy(tmp_path, capsys, agg, fixture_runs):
    out = tmp_path / "chart.png"
    assert plot_comparison(fixture_runs, out) == 0
    svg = tmp_path / "chart.svg"
    assert out.exists()
    assert svg.exists()
    assert out.read_bytes().startswith(b"\x89PNG")
    assert "<svg" in svg.read_text()
    printed = capsys.readouterr().out
    assert f"Wrote {out}" in printed
    assert f"Wrote {svg}" in printed and "vector copy" in printed


def test_non_png_output_keeps_single_file(tmp_path, capsys, agg, fixture_runs):
    out = tmp_path / "chart.svg"
    assert plot_comparison(fixture_runs, out) == 0
    assert out.exists()
    assert "<svg" in out.read_text()
    assert sorted(p.name for p in tmp_path.glob("chart*")) == ["chart.svg"]
    printed = capsys.readouterr().out
    assert f"Wrote {out}" in printed


def curve_payload(model="a", curves=((1, 2, 3),), attempts=3):
    payload = run_payload(model)
    payload["tasks"] = {
        "same_but_different": {
            "name": "same_but_different",
            "score": 0.5,
            "metrics": {},
            "details": {
                "attempts_per_premise": attempts,
                "premises": [
                    {"premise": f"p{i}", "acceptance_curve": list(curve)}
                    for i, curve in enumerate(curves)
                ],
            },
        }
    }
    return payload


def test_acceptance_curves_collects_one_curve_per_premise(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "a.json").write_text(json.dumps(curve_payload(curves=((1, 2, 3), (0, 1, 2)))))
    (runs_dir / "b.json").write_text(json.dumps(curve_payload("b", curves=((1, 1, 1),))))
    curves = acceptance_curves(load_runs(runs_dir))
    assert curves == {"a": [[1, 2, 3], [0, 1, 2]], "b": [[1, 1, 1]]}


def test_acceptance_curves_skip_ragged_and_missing_data(tmp_path):
    """A curve shorter than the budget would invent acceptances if padded."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "ragged.json").write_text(
        json.dumps(curve_payload(curves=((1, 2), (1, 2, 3)), attempts=3))
    )
    (runs_dir / "no_task.json").write_text(json.dumps(run_payload("b")))
    assert acceptance_curves(load_runs(runs_dir)) == {"a": [[1, 2, 3]]}


def test_acceptance_curve_chart_writes_png_and_svg(tmp_path, capsys, agg):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "a.json").write_text(json.dumps(curve_payload(curves=((1, 2, 3), (0, 1, 2)))))
    out = tmp_path / "curves.png"
    assert plot_acceptance_curves(runs_dir, out) == 0
    assert out.read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (tmp_path / "curves.svg").read_text()
    assert "2 curves" in capsys.readouterr().out


def test_acceptance_curve_chart_reports_when_there_is_nothing_to_plot(tmp_path, capsys, agg):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "a.json").write_text(json.dumps(run_payload()))
    assert plot_acceptance_curves(runs_dir, tmp_path / "curves.png") == 1
    assert "No Same But Different acceptance curves" in capsys.readouterr().out
