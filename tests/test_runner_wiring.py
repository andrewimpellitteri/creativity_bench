"""End-to-end runner wiring for the upgraded task capabilities.

The runner must feed the upgraded tasks their new inputs (sampled premises,
both Telephone conditions, the judge panel parameter) and the saved runs must
carry the new metrics (survival curves, sensitivity/specificity, gate stats)
into metadata and reports. Offline only: FakeClient/FakeEmbedder throughout.
"""

from __future__ import annotations

import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench import data
from creativity_bench.runner import run_benchmark


def collapse_responder(messages):
    """Every telephone chain converges: expansions repeat verbatim."""
    prompt = messages[-1]["content"]
    if "Summarize" in prompt:
        return "fixed summary"
    return "fixed story"


def test_telephone_wiring_runs_both_conditions_over_sampled_premises():
    client = FakeClient(collapse_responder)
    result = run_benchmark(client, client, FakeEmbedder(), tasks=["telephone"], seed=5, fast=True)

    task = result.task_results["telephone"]
    assert task.metrics["conditions"] == ["deterministic", "stochastic"]
    assert task.metrics["n_premises"] == result.metadata["task_sizes"]["telephone_premises"]

    curves = task.metrics["survival_curves"]
    assert set(curves) == {"deterministic", "stochastic"}
    assert all(len(curve) == task.metrics["max_iterations"] for curve in curves.values())
    # With the fake client every chain's second expansion is identical to the
    # first, so each chain collapses at round 1 in both conditions.
    assert all(value == 0.0 for curve in curves.values() for value in curve)
    assert task.metrics["observed_collapses"] == 2 * len(task.details["premises"])
    assert task.score == pytest.approx(1 / task.metrics["max_iterations"])

    premises = task.details["premises"]
    assert len(premises) == 2
    assert all(p in data.STORY_PROMPTS for p in premises)
    for premise in premises:
        # One expansion prompt per condition carries the sampled premise.
        assert sum(premise in call[-1]["content"] for call in client.calls) >= 2

    assert result.metadata["telephone_conditions"] == ["deterministic", "stochastic"]
    assert result.metadata["evaluation_complete"] is True


def test_shaggy_dog_wiring_passes_judge_panel_and_gate_metrics_appear():
    gate_prompts = []
    explanations = iter(
        [
            "Kindness towards animals, clearly.",
            "A submarine race, obviously.",
            "Soup for dinner, naturally.",
        ]
    )

    def judge_responder(messages):
        prompt = messages[-1]["content"]
        if "shaggy dog storytelling contest" in prompt:
            gate_prompts.append(prompt)
            return '{"comprehensible": true}'
        return next(explanations)

    writer = FakeClient(lambda _: "A man walks into a bar. Nothing comes of it. The end.")
    judge = FakeClient(judge_responder, model="judge-model")
    result = run_benchmark(writer, judge, None, tasks=["shaggy_dog"], seed=3, fast=True)

    task = result.task_results["shaggy_dog"]
    # The runner supplies the multi-judge parameter; the single production
    # judge forms a one-element panel, and the gate runs before scoring.
    assert task.metrics["n_judges"] == 1
    assert task.metrics["judge_models"] == ["judge-model"]
    assert len(gate_prompts) == 1
    assert task.metrics["comprehensible"] is True
    assert task.metrics["judge_unresolved"] == 0
    assert task.metrics["degenerate"] is False
    assert len(task.details["explanations_per_judge"]) == 1
    assert len(task.details["explanations"]) == result.metadata["task_sizes"]["judges"]
    assert task.score == pytest.approx(1.0)
    assert result.metadata["evaluation_complete"] is True


def _write_run(runs_dir, name, model, *, seed=0, tasks=None):
    import json

    from creativity_bench.comparison import PROVENANCE_FIELDS

    scores = {"telephone": 0.5, "subversion": 0.5, "shaggy_dog": 0.25}
    payload = {
        "schema_version": 2,
        "model": model,
        "provider": "openai",
        "composite": 0.5,
        "scores": scores,
        "weights": {t: 1 / len(scores) for t in scores},
        "seed": seed,
        "duration_seconds": 1.0,
        "metadata": {
            **{f: "test" for f in PROVENANCE_FIELDS},
            "timestamp": "2026-09-15T12:00:00",
            "judge_model": "judge-x",
            "fast": False,
            "selected_tasks": sorted(scores),
            "generation_provider": "openai",
            "task_sizes": {"telephone_premises": 2},
            "generation_settings": {"policy": "test"},
            "judge_settings": {"policy": "test"},
            "evaluation_complete": True,
        },
        "tasks": tasks or [],
    }
    (runs_dir / name).write_text(json.dumps(payload))


def test_report_renders_new_task_metrics(tmp_path):
    from creativity_bench.report import build_leaderboard

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    upgraded_tasks = {
        "telephone": {
            "name": "telephone",
            "score": 0.5,
            "metrics": {
                "survival_curves": {
                    "deterministic": [1.0, 0.25],
                    "stochastic": [1.0, 0.75],
                }
            },
            "details": {},
        },
        "subversion": {
            "name": "subversion",
            "score": 0.5,
            "metrics": {"sensitivity": 1.0, "specificity": 0.5},
            "details": {},
        },
        "shaggy_dog": {
            "name": "shaggy_dog",
            "score": 0.25,
            "metrics": {"comprehensible": True},
            "details": {},
        },
    }
    _write_run(runs_dir, "a.json", "alpha", tasks=upgraded_tasks)
    _write_run(runs_dir, "b.json", "legacy", seed=1)

    text = build_leaderboard(runs_dir, generated="2026-09-15")

    assert "### Task diagnostics" in text
    assert (
        "| `alpha` | 0.250 | 0.750 | 1.000 | 0.500 | 1.000 |" in text
    ), text
    # Legacy runs without task payloads render as em dashes, not crashes.
    assert "| `legacy` | — | — | — | — | — |" in text
