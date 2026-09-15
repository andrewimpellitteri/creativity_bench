"""Offline tests for Telephone conditions and censored survival curves.

Design audit: "Stochastic wording can prevent convergence indefinitely; eight
rounds is a ceiling, not an observed collapse time. Report censored survival
curves across premises, and separate deterministic and stochastic generation
conditions."
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.tasks.telephone import telephone_game


def test_telephone_single_stochastic_chain_matches_legacy_contract():
    def responder(messages):
        prompt = messages[-1]["content"]
        return "same summary." if "Summarize" in prompt else "story text"

    client = FakeClient(responder)
    embedder = FakeEmbedder(fixed={"cat": np.array([1.0, 0.0])})
    result = telephone_game(client, embedder, seed_text="A cat chased a mouse.", max_iter=5)
    assert result.metrics["iterations_survived"] == 1
    assert result.score == pytest.approx(1 / 5)
    assert result.details["transcript"][-1]["exact_match"] is True
    assert result.metrics["right_censored"] is False


def test_telephone_conditions_use_different_temperatures():
    temps: list[float] = []

    class RecordingClient(FakeClient):
        def generate(self, prompt, *, system=None, **kwargs):
            if "Summarize" not in prompt:
                temps.append(kwargs["temperature"])
            return f"unique story {len(temps)} with fresh words"

    client = RecordingClient(lambda _: "x")
    result = telephone_game(
        client,
        FakeEmbedder(),
        seed_text="seed",
        max_iter=2,
        conditions=("stochastic", "deterministic"),
    )
    # Two stochastic expansions (0.8) then two deterministic ones (0.0).
    assert temps == [0.8, 0.8, 0.0, 0.0]
    by_condition = {c["condition"] for c in result.details["chains"]}
    assert by_condition == {"stochastic", "deterministic"}
    assert set(result.metrics["survival_curves"]) == {"stochastic", "deterministic"}


def test_telephone_survival_curve_records_censoring_across_premises():
    mode = {"current": None}
    drift_counter = {"n": 0}

    def responder(messages):
        prompt = messages[-1]["content"]
        if "Summarize" in prompt:
            return "A stable tidy summary."
        # The first expansion prompt of a chain contains the premise keyword
        # and sets the chain's behavior; later prompts contain only summaries.
        if "collapse" in prompt:
            mode["current"] = "collapse"
            return "identical fixed point"
        if "drift" in prompt:
            mode["current"] = "drift"
            drift_counter["n"] += 1
            return f"drifting story {drift_counter['n']} with new words"
        if mode["current"] == "collapse":
            # Matches the chain's first expansion: observed collapse.
            return "identical fixed point"
        drift_counter["n"] += 1
        return f"drifting story {drift_counter['n']} with new words"

    client = FakeClient(responder)
    result = telephone_game(
        client,
        FakeEmbedder(),
        premises=["collapse", "drift", "collapse", "drift"],
        max_iter=4,
    )
    chains = result.details["chains"]
    assert len(chains) == 4
    censored = [c for c in chains if c["right_censored"]]
    observed = [c for c in chains if not c["right_censored"]]
    assert len(censored) == 2
    assert all(c["collapse_iter"] is None for c in censored)
    assert all(c["collapse_iter"] == 1 for c in observed)
    # Censored premises count as surviving; premises collapsed at round 1 do
    # not survive past round 1.
    curve = result.metrics["survival_curves"]["stochastic"]
    assert curve == [0.5, 0.5, 0.5, 0.5]
    assert result.metrics["censored_chains"] == 2
    assert result.metrics["observed_collapses"] == 2
    # Score: censored chains contribute the full budget, observed chains the
    # collapse time: mean(1/4, 4/4, 1/4, 4/4) = 0.625.
    assert result.score == pytest.approx(0.625)


def test_telephone_rejects_unknown_condition():
    with pytest.raises(ValueError):
        telephone_game(
            FakeClient(lambda _: "x"),
            FakeEmbedder(),
            seed_text="seed",
            conditions=("chaotic",),
        )
