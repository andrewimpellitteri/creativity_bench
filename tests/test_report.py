import json

import pytest

from creativity_bench.report import build_leaderboard, collect_rows, write_leaderboard
from creativity_bench.visualize import TASK_ORDER

FULL_SCORES = {t: 0.5 for t in TASK_ORDER}


def write_run(
    runs_dir,
    name,
    *,
    model,
    composite,
    scores,
    seed=0,
    judge="judge-x",
    provider="openai",
    timestamp="2026-09-15T12:00:00",
    schema=2,
):
    payload = {
        "schema_version": schema,
        "model": model,
        "provider": provider,
        "composite": composite,
        "scores": scores,
        "weights": {t: 1 / len(scores) for t in scores},
        "seed": seed,
        "duration_seconds": 60.0,
        "metadata": {"timestamp": timestamp, "judge_model": judge},
        "tasks": [],
    }
    path = runs_dir / name
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def runs_dir(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d


def test_orders_by_composite_and_averages_repeats(runs_dir):
    write_run(runs_dir, "a1.json", model="alpha", composite=0.8, scores=FULL_SCORES)
    write_run(runs_dir, "a2.json", model="alpha", composite=0.6, scores=FULL_SCORES, seed=1)
    write_run(runs_dir, "b1.json", model="beta", composite=0.9, scores=FULL_SCORES)

    rows = collect_rows(
        {
            "alpha": [
                json.loads((runs_dir / "a1.json").read_text()),
                json.loads((runs_dir / "a2.json").read_text()),
            ],
            "beta": [json.loads((runs_dir / "b1.json").read_text())],
        }
    )

    assert [r["model"] for r in rows] == ["beta", "alpha"]
    alpha = rows[1]
    assert alpha["composite"] == pytest.approx(0.7)
    assert alpha["std"] == pytest.approx(0.1)
    assert alpha["n"] == 2
    assert alpha["seeds"] == "0, 1"


def test_missing_task_renders_em_dash(runs_dir):
    partial = {t: 0.5 for t in TASK_ORDER if t in ("free_association", "diversity")}
    write_run(runs_dir, "old.json", model="alpha", composite=0.5, scores=partial)

    text = build_leaderboard(runs_dir, generated="2026-09-15")

    assert "— |" in text
    assert "| `alpha` |" in text


def test_bold_best_composite_and_task(runs_dir):
    better = {t: 0.9 for t in TASK_ORDER}
    worse = {t: 0.1 for t in TASK_ORDER}
    write_run(runs_dir, "g.json", model="gold", composite=0.9, scores=better)
    write_run(runs_dir, "s.json", model="silver", composite=0.7, scores=worse)

    text = build_leaderboard(runs_dir, generated="2026-09-15")

    assert "| 1 | `gold` | **0.900 ± 0.000**" in text
    assert "**0.900**" in text
    assert "0.100" in text


def test_notes_flag_self_judging(runs_dir):
    write_run(
        runs_dir, "s.json", model="judge-x", composite=0.5, scores=FULL_SCORES, judge="judge-x"
    )
    text = build_leaderboard(runs_dir, generated="2026-09-15")
    assert "graded its own outputs" in text


def test_no_self_judge_note_when_external(runs_dir):
    write_run(runs_dir, "s.json", model="alpha", composite=0.5, scores=FULL_SCORES, judge="judge-x")
    text = build_leaderboard(runs_dir, generated="2026-09-15")
    assert "graded its own outputs" not in text


def test_empty_runs_dir_raises(runs_dir):
    with pytest.raises(ValueError, match="No usable run files"):
        build_leaderboard(runs_dir)


def test_stale_or_invalid_schema_files_skipped(runs_dir, capsys):
    write_run(runs_dir, "good.json", model="alpha", composite=0.5, scores=FULL_SCORES)
    write_run(runs_dir, "old.json", model="ancient", composite=0.9, scores=FULL_SCORES, schema=1)
    (runs_dir / "junk.json").write_text("{not json")

    text = build_leaderboard(runs_dir, generated="2026-09-15")

    assert "ancient" not in text
    assert "alpha" in text
    assert "not valid JSON" in capsys.readouterr().out


def test_write_leaderboard_creates_parents_and_returns_path(tmp_path, runs_dir):
    write_run(runs_dir, "s.json", model="alpha", composite=0.5, scores=FULL_SCORES)
    out = tmp_path / "results" / "deep" / "leaderboard.md"

    returned = write_leaderboard(runs_dir, out)

    assert returned == out
    assert out.exists()
    assert out.read_text().startswith("# Creativity Bench — Leaderboard")


def test_fast_run_flagged_and_noted(runs_dir):
    write_run(runs_dir, "f.json", model="alpha", composite=0.5, scores=FULL_SCORES)
    payload = json.loads((runs_dir / "f.json").read_text())
    payload["metadata"]["fast"] = True
    (runs_dir / "f.json").write_text(json.dumps(payload))

    text = build_leaderboard(runs_dir, generated="2026-09-15")

    assert "`alpha` ⚡" in text
    assert "not directly comparable" in text


def test_self_judge_detection_ignores_fast_marker(runs_dir):
    write_run(
        runs_dir, "s.json", model="judge-x", composite=0.5, scores=FULL_SCORES, judge="judge-x"
    )
    payload = json.loads((runs_dir / "s.json").read_text())
    payload["metadata"]["fast"] = True
    (runs_dir / "s.json").write_text(json.dumps(payload))

    text = build_leaderboard(runs_dir, generated="2026-09-15")

    assert "graded its own outputs" in text
