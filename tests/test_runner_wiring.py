"""End-to-end runner wiring for the upgraded task capabilities.

The runner must feed the upgraded tasks their new inputs (sampled premises,
both Telephone conditions, the judge panel parameter) and the saved runs must
carry the new metrics (survival curves, sensitivity/specificity, gate stats)
into metadata and reports. Offline only: FakeClient/FakeEmbedder throughout.
"""

from __future__ import annotations

import itertools
import json
import random

import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench import data
from creativity_bench.comparison import PROVENANCE_FIELDS
from creativity_bench.report import build_leaderboard
from creativity_bench.runner import run_benchmark
from creativity_bench.tasks.camels_back import _edits_conflict, build_edit_schedule
from creativity_bench.tasks.subversion import INVERSION_DIMENSIONS


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


PASS_VERDICT = '{"coherent": true, "edits_applied": true, "quality_maintained": true}'


CAMELS_NEXT = {
    "the original story text": "the once modified story text",
    "the once modified story text": "the twice modified story text",
}


def camels_responder(messages):
    prompt = messages[-1]["content"]
    if "coherent" in prompt:  # edit judge
        return PASS_VERDICT
    if "INSTRUCTIONS:" in prompt:  # edit round: transform the current story
        for current, modified in CAMELS_NEXT.items():
            if f"STORY:\n{current}" in prompt:
                return modified
        return "a freshly modified story text"
    if "based on this premise" in prompt:  # starting story
        return "the original story text"
    return "unused"


def test_camels_back_wiring_produces_schedule_and_survival_metrics():
    def run():
        return run_benchmark(
            FakeClient(camels_responder),
            FakeClient(camels_responder),
            None,
            tasks=["camels_back"],
            seed=9,
            fast=True,
        )

    result = run()
    task = result.task_results["camels_back"]
    # The runner hands the task its own rng stream: the precomputed schedule
    # matches building it directly from Random("9:camels_back").
    expected = build_edit_schedule(
        data.EDIT_REQUESTS,
        result.metadata["task_sizes"]["max_edits"],
        random.Random("9:camels_back"),
    )
    assert task.details["schedule"] == expected
    assert len(expected) == 3
    for bundle in expected:
        assert bundle and all(edit in data.EDIT_REQUESTS for edit in bundle)
        assert all(not _edits_conflict(a, b) for a, b in itertools.combinations(bundle, 2))
    # Every round passes and the story keeps changing: right-censored at budget.
    assert task.metrics["rounds_survived"] == 3
    assert task.metrics["right_censored"] is True
    assert task.metrics["stopped_changing"] is False
    assert task.metrics["constraints"] == "replaced_each_round"
    assert task.score == pytest.approx(1.0)
    # Identical seed, identical schedule.
    assert run().task_results["camels_back"].details["schedule"] == expected


def subversion_responder(messages):
    prompt = messages[-1]["content"]
    if "opposite" in prompt and "JSON object with this boolean field" in prompt:
        return '{"opposite": true}'
    if "based on this premise" in prompt:  # seed story generation
        return "the original story text"
    return "an unrelated generated story"  # subversion writer


def test_subversion_wiring_records_dimensions_and_discrimination_metrics():
    client = FakeClient(subversion_responder)
    judge = FakeClient(subversion_responder)
    result = run_benchmark(client, judge, None, tasks=["subversion"], seed=4, fast=True)

    task = result.task_results["subversion"]
    assert task.metrics["inversion_dimensions"] == list(INVERSION_DIMENSIONS)
    # Fast budget: one premise, two runs per premise -> 4 judged pairs.
    assert task.metrics["within_pairs"] == 2
    assert task.metrics["cross_pairs"] == 2
    # All verdicts are "opposite": perfect sensitivity, no specificity.
    assert task.metrics["sensitivity"] == pytest.approx(1.0)
    assert task.metrics["specificity"] == pytest.approx(0.0)
    assert task.score == pytest.approx(0.0)
    # Each pair records the dimension its subversion inverted, cycling in the
    # shared fixed order, and the writer prompts name that dimension.
    dimensions = {pair["j"]: pair["dimension"] for pair in task.details["pairs"]}
    assert dimensions == {0: "outcome", 1: "tone"}
    assert all(pair["dimension"] in INVERSION_DIMENSIONS for pair in task.details["pairs"])
    for dimension in dimensions.values():
        assert any(f"dimension: {dimension}" in call[-1]["content"] for call in client.calls)


def subset_responder(messages):
    """Covers telephone plus Same But Different writer and judge calls."""
    system = messages[0].get("content") or ""
    prompt = messages[-1]["content"]
    if "premise_adherent" in system:  # same_but_different judge
        return (
            '{"premise_adherent": true, "comprehensible": true, "plot_distinct": true, '
            '"evidence": "ok", "summary": "a causal plot summary"}'
        )
    if "Write a complete short story" in system:  # same_but_different writer
        return f"a distinct story about: {prompt[:80]}"
    if "Summarize" in prompt:
        return "fixed summary"
    return "fixed story"


def test_same_seed_samples_identical_inputs_across_task_subsets():
    def run(tasks, seed):
        return run_benchmark(
            FakeClient(subset_responder),
            FakeClient(subset_responder),
            FakeEmbedder(),
            tasks=tasks,
            seed=seed,
            fast=True,
        )

    both = run(["telephone", "same_but_different"], 11)
    telephone_alone = run(["telephone"], 11)
    premises_alone = run(["same_but_different"], 11)
    other_seed = run(["telephone"], 12)

    def telephone(result):
        return result.task_results["telephone"].details["premises"]

    sampled = telephone(both)
    # The telephone rng stream is independent of which tasks run alongside it.
    assert telephone(both) == telephone(telephone_alone) == sampled
    assert len(sampled) == both.metadata["task_sizes"]["telephone_premises"]
    assert telephone(other_seed) != sampled

    def sbd_premises(result):
        records = result.task_results["same_but_different"].details["premises"]
        return [record["premise"] for record in records]

    assert sbd_premises(both) == sbd_premises(premises_alone)
    assert len(sbd_premises(both)) == both.metadata["task_sizes"]["distinct_premises"]
    assert all(p in data.CREATIVE_PREMISES for p in sbd_premises(both))
