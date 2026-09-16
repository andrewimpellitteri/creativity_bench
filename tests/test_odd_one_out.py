import random

import numpy as np
import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.tasks import odd_one_out
from creativity_bench.tasks.odd_one_out import SEED_LISTS

REJECT_VERDICT = 'Sure! Here is my answer: {"qualifies": false}'
ACCEPT_VERDICT = '{"qualifies": true}'


def accepting_judge():
    return FakeClient(lambda _: ACCEPT_VERDICT)


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
        judge_client=accepting_judge(),
        n_lists=2,
        rng=random.Random(0),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metrics["n_lists"] == 2
    assert len(result.details["lists"]) == 2
    assert result.metrics["judge_used"] is True


def test_odd_one_out_maximally_distant_item_scores_high():
    lists = [("dog breeds", ["alpha", "beta"])]
    client = FakeClient(lambda _: "novel item")
    # Antiparallel to every anchor: cosine distance 2 -> halved -> 1.0.
    result = odd_one_out(
        client,
        embedder=fixed_embedder([-1.0, 0.0]),
        judge_client=accepting_judge(),
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
        judge_client=accepting_judge(),
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
        judge_client=accepting_judge(),
        lists=lists,
        rng=random.Random(0),
    )
    assert result.details["lists"][0]["min_distance"] == pytest.approx(0.0)
    assert result.details["lists"][0]["mean_distance"] == pytest.approx(0.5)
    assert result.score == pytest.approx(0.0)


def test_odd_one_out_requires_a_list():
    with pytest.raises(ValueError):
        odd_one_out(
            FakeClient(lambda _: "x"),
            embedder=FakeEmbedder(),
            judge_client=accepting_judge(),
            n_lists=0,
        )


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


def test_odd_one_out_malformed_judge_response_is_unresolved():
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
    assert result.details["lists"][0]["qualified"] is None
    assert result.details["lists"][0]["judge_unparseable"] is True
    assert result.score == 0.0
    assert result.metrics["judge_unresolved"] == 1
    assert judge.usage.requests == 2  # one retry before the fallback


@pytest.mark.parametrize("raw", ['{"qualifies": "false"}', '{"qualifies": 1}', "[]", "null"])
def test_odd_one_out_nonboolean_judge_cannot_award_credit(raw):
    result = odd_one_out(
        FakeClient(lambda _: "novel item"),
        embedder=fixed_embedder([-1, 0]),
        judge_client=FakeClient(lambda _: raw),
        lists=[("trees", ["alpha", "beta"])],
    )
    assert result.score == 0
    assert result.metrics["judge_unresolved"] == 1
    assert result.details["lists"][0]["judge_attempts"] == [raw, raw]


def test_odd_one_out_without_a_judge_refuses_to_score():
    # A missing gate is a configuration failure, not a model failure: it must
    # not masquerade as a zero score, which was the previous behavior.
    client = FakeClient(lambda _: "novel item")
    with pytest.raises(ValueError, match="judge_client"):
        odd_one_out(
            client,
            embedder=fixed_embedder([-1, 0]),
            lists=[("trees", ["alpha", "beta"])],
        )
    assert client.usage.requests == 0  # fails before spending a generation call


def test_odd_one_out_unresolved_judgment_scores_zero_and_marks_incomplete():
    # Fail-closed handling of unresolved judgments is unchanged: zero score AND
    # a judge_unresolved count, which the runner reads as an incomplete run.
    result = odd_one_out(
        FakeClient(lambda _: "novel item"),
        embedder=fixed_embedder([-1, 0]),
        judge_client=FakeClient(lambda _: "no verdict here"),
        lists=[("trees", ["alpha", "beta"]), ("dog breeds", ["alpha", "beta"])],
    )
    assert result.score == 0
    assert result.metrics["mean_min_distance"] == pytest.approx(2)
    assert result.metrics["judge_unresolved"] == 2
    assert result.metrics["validity_rate"] == 0
    assert [r["validity_status"] for r in result.details["lists"]] == ["unresolved"] * 2
