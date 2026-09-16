"""Offline checks for chart output files: PNG plus its SVG vector copy."""

from __future__ import annotations

import json

import pytest

from creativity_bench.visualize import plot_comparison


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
