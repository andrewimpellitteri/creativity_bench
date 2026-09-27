from __future__ import annotations

import random

import numpy as np
import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.data import SAMPLE_STORIES
from creativity_bench.tasks.this_and_that import this_and_that

PASS = '{"draws_on_a": true, "draws_on_b": true, "comprehensible": true}'

STORIES = [
    {"genre": "alpha", "text": "AAA story about a lighthouse."},
    {"genre": "beta", "text": "BBB story about a submarine."},
    {"genre": "gamma", "text": "CCC story about a bakery."},
]


def responder(judge_reply=PASS, blend="BLEND text"):
    def respond(messages):
        prompt = messages[-1]["content"]
        if "draws_on_a" in prompt:
            return judge_reply
        return blend

    return respond


def geometry_embedder():
    """Place the blend exactly between A and B, and the baseline off-axis."""
    return FakeEmbedder(
        fixed={
            "AAA": np.array([1.0, 0.0, 0.0]),
            "BBB": np.array([0.0, 1.0, 0.0]),
            "CCC": np.array([0.0, 0.0, 1.0]),
            "BLEND": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        },
        dim=3,
    )


def run(**kwargs):
    defaults = dict(
        embedder=geometry_embedder(),
        judge_client=FakeClient(responder()),
        stories=STORIES,
        n_pairs=1,
        # Seed 4 samples STORIES in order, so A/B/baseline are alpha/beta/gamma.
        rng=random.Random(4),
    )
    defaults.update(kwargs)
    client = defaults.pop("client", FakeClient(responder()))
    return this_and_that(client, **defaults)


def test_geodesic_blend_scores_near_one():
    result = run()
    assert result.score == pytest.approx(1.0, abs=1e-6)
    assert result.metrics["validity_rate"] == 1.0
    assert result.metrics["unresolved_judgments"] == 0


def test_raw_summed_cosine_distance_is_reported():
    result = run()
    pair = result.details["pairs"][0]
    # Blend sits at 45 degrees from each example: 1 - cos(45) = 0.2929 each.
    assert pair["summed_cosine_distance"] == pytest.approx(2 * (1 - np.sqrt(0.5)), abs=1e-6)
    assert "mean_summed_cosine_distance" in result.metrics


def test_verbatim_copy_of_an_example_scores_zero():
    """Endpoint of the geodesic: excess is 0, but the balance term is 1, so a
    copy-paste of example A must not earn what a true blend earns."""
    result = run(client=FakeClient(responder(blend="AAA copied verbatim")))
    pair = result.details["pairs"][0]
    assert pair["angular_excess"] == pytest.approx(0.0, abs=1e-9)
    assert pair["balance"] == pytest.approx(0.0, abs=1e-9)
    assert result.score == 0.0
    assert result.details["pairs"][0]["validity_status"] == "valid"
    assert result.metrics["mean_balance"] == pytest.approx(0.0, abs=1e-9)


def test_midpoint_blend_keeps_the_full_score():
    """The balance term must not tax a genuine midpoint blend."""
    result = run()
    pair = result.details["pairs"][0]
    assert pair["balance"] == pytest.approx(1.0, abs=1e-9)
    assert result.score == pytest.approx(1.0, abs=1e-6)


def test_excess_direction_and_magnitude_on_an_asymmetric_fixture():
    """A=x, B=y, candidate=(1,0,1)/sqrt(2), baseline=-z: d(a,c)=0.25 and
    d(b,c)=0.5, so excess = 0.25 + 0.5 - 0.5 = 0.25. A sign flip of the
    excess formula would clamp to 0 and fail this; the baseline excess is
    0.5 + 0.5 - 0.5 = 0.5. The fold is the product: (1 - 0.25/0.5) * 0.5."""
    embedder = FakeEmbedder(
        fixed={
            "AAA": np.array([1.0, 0.0, 0.0]),
            "BBB": np.array([0.0, 1.0, 0.0]),
            "CCC": np.array([0.0, 0.0, -1.0]),
            "SKEW": np.array([1.0, 0.0, 1.0]) / np.sqrt(2),
        },
        dim=3,
    )
    result = run(embedder=embedder, client=FakeClient(responder(blend="SKEW text")))
    pair = result.details["pairs"][0]
    assert pair["angular_excess"] == pytest.approx(0.25, abs=1e-9)
    assert pair["baseline_angular_excess"] == pytest.approx(0.5, abs=1e-9)
    assert pair["balance"] == pytest.approx(0.5, abs=1e-9)
    assert result.score == pytest.approx(0.25, abs=1e-9)


def test_uninformative_baseline_excludes_the_pair_instead_of_zeroing_it():
    """A baseline sitting on the geodesic makes the pair unmeasurable; it must
    drop out of the mean and be counted, not drag the score toward 0. Pair 1
    has the midpoint baseline gamma (excluded), pair 2 the off-axis delta
    (informative): the score is pair 2's own 1.0, not the zeroed mean 0.5."""
    stories = [*STORIES, {"genre": "delta", "text": "DDD story about a canal."}]
    embedder = FakeEmbedder(
        fixed={
            "AAA": np.array([1.0, 0.0, 0.0]),
            "BBB": np.array([0.0, 1.0, 0.0]),
            "CCC": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
            "DDD": np.array([0.0, 0.0, -1.0]),
            "BLEND": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        },
        dim=3,
    )
    result = run(embedder=embedder, stories=stories, n_pairs=2, rng=random.Random(49))
    assert result.metrics["excluded_pairs"] == 1
    assert result.details["pairs"][0]["baseline_excluded"] is True
    assert result.details["pairs"][1]["baseline_excluded"] is False
    assert result.score == pytest.approx(1.0, abs=1e-9)
    assert result.details["baseline_guard"] == 1e-3


def test_small_but_informative_baseline_still_scores():
    """Baseline excess of 0.002 sits above the 1e-3 guard: the pair keeps its
    normal weight and the reproduced 1.27e-4 failure stays excluded."""
    theta = float(np.arccos(np.sqrt(2) * np.cos(0.251 * np.pi)))
    baseline = np.array([np.cos(theta) / np.sqrt(2), np.cos(theta) / np.sqrt(2), np.sin(theta)])
    embedder = FakeEmbedder(
        fixed={
            "AAA": np.array([1.0, 0.0, 0.0]),
            "BBB": np.array([0.0, 1.0, 0.0]),
            "CCC": baseline,
            "BLEND": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        },
        dim=3,
    )
    result = run(embedder=embedder)
    pair = result.details["pairs"][0]
    assert pair["baseline_excluded"] is False
    assert pair["baseline_angular_excess"] == pytest.approx(0.002, abs=1e-9)
    assert result.metrics["excluded_pairs"] == 0
    assert result.score == pytest.approx(1.0, abs=1e-9)


def test_off_axis_story_scores_zero():
    """A candidate no closer to the pair than the unrelated baseline earns nothing."""
    embedder = FakeEmbedder(
        fixed={
            "AAA": np.array([1.0, 0.0, 0.0]),
            "BBB": np.array([0.0, 1.0, 0.0]),
            "CCC": np.array([0.0, 0.0, 1.0]),
            "BLEND": np.array([0.0, 0.0, 1.0]),
        },
        dim=3,
    )
    result = run(embedder=embedder)
    assert result.score == 0.0
    assert result.details["pairs"][0]["validity_status"] == "valid"


def test_failed_gate_zeroes_a_well_placed_blend():
    fail = '{"draws_on_a": true, "draws_on_b": false, "comprehensible": true}'
    result = run(judge_client=FakeClient(responder(judge_reply=fail)))
    assert result.score == 0.0
    assert result.details["pairs"][0]["failed_gates"] == ["draws_on_b"]
    assert result.metrics["validity_rate"] == 0.0


def test_unresolved_judgment_scores_zero_and_is_counted():
    result = run(judge_client=FakeClient(responder(judge_reply="not json")))
    assert result.score == 0.0
    assert result.metrics["unresolved_judgments"] == 1
    assert len(result.details["pairs"][0]["judge_attempts"]) == 2


def test_generation_error_is_recorded_not_raised():
    def boom(_messages):
        raise RuntimeError("upstream 500")

    result = run(client=FakeClient(boom))
    assert result.score == 0.0
    assert result.metrics["generation_errors"] == 1
    assert "upstream 500" in result.details["pairs"][0]["generation_error"]


def test_requires_judge_and_enough_stories():
    with pytest.raises(ValueError, match="judge_client"):
        this_and_that(FakeClient(responder()), embedder=FakeEmbedder(), stories=STORIES)
    with pytest.raises(ValueError, match="at least 3"):
        this_and_that(
            FakeClient(responder()),
            embedder=FakeEmbedder(),
            judge_client=FakeClient(responder()),
            stories=STORIES[:2],
        )


def test_runs_against_the_shipped_corpus():
    result = run(stories=SAMPLE_STORIES, embedder=FakeEmbedder(), n_pairs=2)
    assert result.metrics["n_pairs"] == 2
    assert 0.0 <= result.score <= 1.0
