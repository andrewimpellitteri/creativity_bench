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
