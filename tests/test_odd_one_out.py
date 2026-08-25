import random

import numpy as np
import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.tasks import odd_one_out
from creativity_bench.tasks.odd_one_out import SEED_LISTS

REJECT_VERDICT = 'Sure! Here is my answer: {"qualifies": false}'


def fixed_embedder(candidate_vector):
    return FakeEmbedder(
        fixed={
            "alpha": np.array([1.0, 0.0]),
            "beta": np.array([1.0, 0.0]),
            "novel item": np.asarray(candidate_vector),
        }
    )


# --- seed data ---------------------------------------------------------------


def test_seed_lists_shape():
    assert 4 <= len(SEED_LISTS) <= 6
    for _theme, items in SEED_LISTS:
        assert 6 <= len(items) <= 10
        assert len(set(items)) == len(items)


# --- scoring -----------------------------------------------------------------


def test_odd_one_out_score_bounds():
    counter = {"n": 0}

    def responder(_):
        counter["n"] += 1
        return f"a wildly unusual item {counter['n']}"

    result = odd_one_out(
        FakeClient(responder),
        embedder=FakeEmbedder(),
        n_lists=2,
        rng=random.Random(0),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metrics["n_lists"] == 2
    assert len(result.details["lists"]) == 2
    assert result.metrics["judge_used"] is False


def test_odd_one_out_maximally_distant_item_scores_high():
    lists = [("dog breeds", ["alpha", "beta"])]
    client = FakeClient(lambda _: "novel item")
    # Antiparallel to every anchor: cosine distance 2 -> halved -> 1.0.
    result = odd_one_out(
        client,
        embedder=fixed_embedder([-1.0, 0.0]),
        lists=lists,
        rng=random.Random(0),
    )
    assert result.score == pytest.approx(1.0)
    assert result.details["lists"][0]["min_distance"] == pytest.approx(2.0)


def test_odd_one_out_item_identical_to_example_scores_low():
    lists = [("trees", ["oak", "birch", "willow"])]
    client = FakeClient(lambda _: "oak")
    # Same string as an anchor -> same hashed embedding -> cosine distance 0.
    result = odd_one_out(
        client,
        embedder=FakeEmbedder(),
        lists=lists,
        rng=random.Random(0),
    )
    assert result.score == pytest.approx(0.0)
    assert result.details["lists"][0]["min_distance"] == pytest.approx(0.0)


def test_odd_one_out_min_not_mean_distance_scores():
    # Far from one anchor, identical to the other: the minimum (not the mean)
    # drives the score, per the documented anchoring rationale.
    lists = [("kitchen tools", ["alpha", "beta"])]
    fixed = {
        "alpha": np.array([1.0, 0.0]),
        "beta": np.array([0.0, 1.0]),
        "novel item": np.array([1.0, 0.0]),
    }
    result = odd_one_out(
        FakeClient(lambda _: "novel item"),
        embedder=FakeEmbedder(fixed=fixed),
        lists=lists,
        rng=random.Random(0),
    )
    assert result.details["lists"][0]["min_distance"] == pytest.approx(0.0)
    assert result.details["lists"][0]["mean_distance"] == pytest.approx(0.5)
    assert result.score == pytest.approx(0.0)


def test_odd_one_out_requires_a_list():
    with pytest.raises(ValueError):
        odd_one_out(FakeClient(lambda _: "x"), embedder=FakeEmbedder(), n_lists=0)


# --- judge gate --------------------------------------------------------------


def test_odd_one_out_judge_rejection_zeroes_item():
    lists = [("dog breeds", ["alpha", "beta"])]
    client = FakeClient(lambda _: "novel item")
    judge = FakeClient(lambda _: REJECT_VERDICT)
    result = odd_one_out(
        client,
        embedder=fixed_embedder([-1.0, 0.0]),
        judge_client=judge,
        lists=lists,
        rng=random.Random(0),
    )
    assert result.metrics["judge_used"] is True
    assert result.metrics["judge_rejected"] == 1
    assert result.details["lists"][0]["qualified"] is False
    assert result.details["lists"][0]["min_distance"] == pytest.approx(2.0)
    assert result.score == 0.0


def test_odd_one_out_malformed_judge_response_counts_qualified():
    lists = [("dog breeds", ["alpha", "beta"])]
    client = FakeClient(lambda _: "novel item")
    judge = FakeClient(lambda _: "I cannot answer that question.")
    result = odd_one_out(
        client,
        embedder=fixed_embedder([-1.0, 0.0]),
        judge_client=judge,
        lists=lists,
        rng=random.Random(0),
    )
    assert result.metrics["judge_unparseable"] == 1
    assert result.details["lists"][0]["qualified"] is True
    assert result.details["lists"][0]["judge_unparseable"] is True
    assert result.score == pytest.approx(1.0)  # fail-open: ungated score kept
    assert judge.usage.requests == 2  # one retry before the fallback
