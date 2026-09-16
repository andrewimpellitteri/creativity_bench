"""Offline coverage for This & That--But Not Like That.

The geometry is pinned with FakeEmbedder's fixed vectors so every score in here
is arithmetic, not an embedding coincidence. Layout used throughout: the good
example sits on +x, the bad example on +y, so d(bad, good) = 0.5 (90 degrees)
and the headroom the score normalizes by is exactly 0.5.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.data import SAMPLE_STORIES
from creativity_bench.tasks.this_and_that_not import this_and_that_not

PASS = '{"draws_on_good": true, "comprehensible": true}'

STORIES = [
    {"genre": "alpha", "text": "AAA story about a lighthouse."},
    {"genre": "beta", "text": "BBB story about a submarine."},
    {"genre": "gamma", "text": "CCC story about a bakery."},
]

GOOD = np.array([1.0, 0.0, 0.0])
BAD = np.array([0.0, 1.0, 0.0])
# Unrelated baseline at 135 degrees from the bad example: an off-topic story
# covers half the headroom on geometry alone.
BASELINE = np.array([0.0, -1.0, 1.0]) / np.sqrt(2)


def responder(judge_reply=PASS, story="REPEL text"):
    def respond(messages):
        prompt = messages[-1]["content"]
        if "draws_on_good" in prompt:
            return judge_reply
        return story

    return respond


def embedder_with(candidate: np.ndarray) -> FakeEmbedder:
    return FakeEmbedder(
        fixed={"AAA": GOOD, "BBB": BAD, "CCC": BASELINE, "REPEL": candidate},
        dim=3,
    )


# 135 degrees from the bad example and 45 from the good one: further from bad
# than good is, while staying on the good example's side.
AWAY = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
# 45 degrees from the bad example: closer to it than the good example is.
TOWARD = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)


def run(**kwargs):
    defaults = dict(
        embedder=embedder_with(AWAY),
        judge_client=FakeClient(responder()),
        stories=STORIES,
        n_pairs=1,
        # Seed 4 samples STORIES in order, so good/bad/baseline are
        # alpha/beta/gamma.
        rng=random.Random(4),
    )
    defaults.update(kwargs)
    client = defaults.pop("client", FakeClient(responder()))
    return this_and_that_not(client, **defaults)


def test_further_from_the_bad_example_scores_its_headroom_fraction():
    result = run()
    pair = result.details["pairs"][0]
    assert pair["good_example"] == "alpha"
    assert pair["bad_example"] == "beta"
    # d(bad, good) = 0.5, d(bad, candidate) = 0.75, headroom = 0.5.
    assert pair["bad_good_angular_distance"] == pytest.approx(0.5, abs=1e-9)
    assert pair["bad_candidate_angular_distance"] == pytest.approx(0.75, abs=1e-9)
    assert pair["repulsion_gain"] == pytest.approx(0.25, abs=1e-9)
    assert result.score == pytest.approx(0.5, abs=1e-9)
    assert result.metrics["validity_rate"] == 1.0
    assert result.metrics["unresolved_judgments"] == 0
    assert result.metrics["generation_errors"] == 0


def test_candidate_no_further_than_the_good_example_scores_zero():
    """Moving toward the bad example earns nothing even with a clean gate."""
    result = run(embedder=embedder_with(TOWARD))
    pair = result.details["pairs"][0]
    assert pair["bad_candidate_angular_distance"] == pytest.approx(0.25, abs=1e-9)
    assert pair["repulsion_gain"] == pytest.approx(-0.25, abs=1e-9)
    assert pair["validity_status"] == "valid"  # the gate passed; the geometry did not
    assert result.score == 0.0


def test_matching_the_good_example_exactly_scores_zero():
    """Copying the good example is not an escape: gain is zero by construction."""
    result = run(embedder=embedder_with(GOOD))
    assert result.details["pairs"][0]["repulsion_gain"] == pytest.approx(0.0, abs=1e-9)
    assert result.score == 0.0


def test_antipodal_candidate_saturates_the_score_at_one():
    result = run(embedder=embedder_with(-BAD))
    pair = result.details["pairs"][0]
    assert pair["bad_candidate_angular_distance"] == pytest.approx(1.0, abs=1e-9)
    assert pair["repulsion"] == pytest.approx(1.0, abs=1e-9)
    assert result.score == 1.0


def test_raw_distances_are_reported_alongside_the_normalized_score():
    result = run()
    pair = result.details["pairs"][0]
    # Source-faithful cosine distances: 1 - cos(90) and 1 - cos(135).
    assert pair["bad_good_cosine_distance"] == pytest.approx(1.0, abs=1e-9)
    assert pair["bad_candidate_cosine_distance"] == pytest.approx(1 + np.sqrt(0.5), abs=1e-9)
    assert pair["good_candidate_angular_distance"] == pytest.approx(0.25, abs=1e-9)
    for key in (
        "mean_bad_good_cosine_distance",
        "mean_bad_candidate_cosine_distance",
        "mean_bad_good_angular_distance",
        "mean_bad_candidate_angular_distance",
        "mean_repulsion_gain",
    ):
        assert key in result.metrics


def test_unrelated_baseline_repulsion_quantifies_why_the_gate_exists():
    """An off-topic story covers half this pair's headroom without doing the task."""
    result = run()
    assert result.details["pairs"][0]["baseline_repulsion"] == pytest.approx(0.5, abs=1e-9)
    assert result.metrics["mean_baseline_repulsion"] == pytest.approx(0.5, abs=1e-9)


def test_failed_gate_zeroes_a_well_placed_candidate():
    fail = '{"draws_on_good": false, "comprehensible": true}'
    result = run(judge_client=FakeClient(responder(judge_reply=fail)))
    pair = result.details["pairs"][0]
    # The geometry was maximal for this pair; the gate withheld all of it.
    assert pair["repulsion"] == pytest.approx(0.5, abs=1e-9)
    assert pair["failed_gates"] == ["draws_on_good"]
    assert result.score == 0.0
    assert result.metrics["validity_rate"] == 0.0


def test_incomprehensible_candidate_fails_the_gate():
    fail = '{"draws_on_good": true, "comprehensible": false}'
    result = run(judge_client=FakeClient(responder(judge_reply=fail)))
    assert result.details["pairs"][0]["failed_gates"] == ["comprehensible"]
    assert result.score == 0.0


def test_unresolved_judgment_scores_zero_and_is_counted():
    result = run(judge_client=FakeClient(responder(judge_reply="not json")))
    assert result.score == 0.0
    assert result.metrics["unresolved_judgments"] == 1
    assert result.details["pairs"][0]["validity_status"] == "unresolved"
    assert len(result.details["pairs"][0]["judge_attempts"]) == 2


def test_non_boolean_gate_fields_are_unresolved():
    result = run(judge_client=FakeClient(responder(judge_reply='{"draws_on_good": 1}')))
    assert result.metrics["unresolved_judgments"] == 1
    assert result.score == 0.0


def test_generation_error_is_recorded_not_raised():
    def boom(_messages):
        raise RuntimeError("upstream 500")

    result = run(client=FakeClient(boom))
    assert result.score == 0.0
    assert result.metrics["generation_errors"] == 1
    assert result.metrics["n_pairs"] == 1
    assert "upstream 500" in result.details["pairs"][0]["generation_error"]
    # No embedding measurement exists for a pair that was never written.
    assert "repulsion" not in result.details["pairs"][0]


def test_degenerate_headroom_scores_zero_and_is_counted():
    """Antipodal examples leave nowhere to go; the pair cannot earn credit."""
    embedder = FakeEmbedder(
        fixed={"AAA": GOOD, "BBB": -GOOD, "CCC": BASELINE, "REPEL": -GOOD},
        dim=3,
    )
    result = run(embedder=embedder)
    assert result.details["pairs"][0]["degenerate_headroom"] is True
    assert result.metrics["degenerate_headrooms"] == 1
    assert result.score == 0.0


def test_requires_judge_and_enough_stories():
    with pytest.raises(ValueError, match="judge_client"):
        this_and_that_not(FakeClient(responder()), embedder=FakeEmbedder(), stories=STORIES)
    with pytest.raises(ValueError, match="at least 3"):
        this_and_that_not(
            FakeClient(responder()),
            embedder=FakeEmbedder(),
            judge_client=FakeClient(responder()),
            stories=STORIES[:2],
        )
    with pytest.raises(ValueError, match="at least 1 pair"):
        this_and_that_not(
            FakeClient(responder()),
            embedder=FakeEmbedder(),
            judge_client=FakeClient(responder()),
            stories=STORIES,
            n_pairs=0,
        )


def test_prompt_names_the_good_and_bad_examples():
    client = FakeClient(responder())
    run(client=client)
    prompt = client.calls[0][-1]["content"]
    assert "GOOD EXAMPLE (alpha)" in prompt
    assert "BAD EXAMPLE (beta)" in prompt
    assert "AAA story about a lighthouse." in prompt
    assert "BBB story about a submarine." in prompt


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_score_stays_in_bounds_on_the_shipped_corpus(seed):
    """Hashed random geometry over real stories must never leave [0, 1]."""
    result = run(
        stories=SAMPLE_STORIES,
        embedder=FakeEmbedder(),
        n_pairs=3,
        rng=random.Random(seed),
    )
    assert result.metrics["n_pairs"] == 3
    assert 0.0 <= result.score <= 1.0
    for pair in result.details["pairs"]:
        assert 0.0 <= pair["repulsion"] <= 1.0
        assert 0.0 <= pair["baseline_repulsion"] <= 1.0
        assert 0.0 <= pair["score"] <= 1.0
