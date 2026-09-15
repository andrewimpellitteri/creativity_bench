"""Offline tests for Shaggy Dog's comprehensibility gate and multi-judge
scaffolding.

Design audit: "Disagreement about morals can reward incoherence; repeated
calls to one judge are not independent judges. Check basic comprehensibility...
and report agreement across several fixed judge models."
"""

from __future__ import annotations

import random

from conftest import FakeClient

from creativity_bench.tasks.shaggy_dog import shaggy_dog

GATE_OK = '{"comprehensible": true}'


def make_storyteller(story="A man walks into a bar. Nothing comes of it. The end."):
    return FakeClient(lambda _: story)


def make_judge(responses):
    replies = iter(responses)
    return FakeClient(lambda _: next(replies))


def test_comprehensible_story_proceeds_to_agreement_scoring():
    judge = make_judge([GATE_OK, "A submarine race, obviously.", "Kindness towards animals."])
    result = shaggy_dog(make_storyteller(), judge, k=2, rng=random.Random(0))
    assert result.metrics["comprehensible"] is True
    assert result.metrics["judge_unresolved"] == 0
    # One gate call plus k agreement samples.
    assert judge.usage.requests == 3


def test_incomprehensible_story_scores_zero():
    judge = make_judge(['{"comprehensible": false}', "unused", "unused"])
    result = shaggy_dog(make_storyteller("gibberish salad words"), judge, k=2)
    assert result.score == 0.0
    assert result.metrics["comprehensible"] is False
    assert result.metrics["explicit_moral"] is False
    assert judge.usage.requests == 1  # gate only; agreement judging skipped


def test_unparseable_gate_verdict_fails_closed():
    judge = make_judge(["not json at all", '{"comprehensible": "maybe"}'])
    result = shaggy_dog(make_storyteller(), judge, k=2)
    assert result.score == 0.0
    assert result.metrics["judge_unresolved"] == 1
    assert result.metrics["comprehensible"] is None


def test_multi_judge_agreement_is_recorded_per_judge_and_across():
    # Only judges[0] runs the comprehensibility gate, so judge_b's list holds
    # exactly its k explanations.
    judge_a = make_judge([GATE_OK, "shared tidy moral about hats", "shared tidy moral about hats"])
    judge_b = make_judge(["a submarine race, obviously", "soup, naturally"])
    result = shaggy_dog(
        make_storyteller(),
        judge_a,
        k=2,
        judge_clients=[judge_a, judge_b],
        rng=random.Random(0),
    )
    assert result.metrics["n_judges"] == 2
    assert result.metrics["judge_models"] == ["fake-model", "fake-model"]
    # Judge A agrees with itself (1.0); judge B's two explanations share no
    # content words (0.0); mean within-judge agreement is 0.5.
    assert result.metrics["mean_within_judge_agreement"] == 0.5
    assert result.metrics["mean_cross_judge_agreement"] == 0.0
    assert len(result.details["explanations_per_judge"]) == 2
    assert len(result.details["explanations"]) == 4


def test_single_judge_cross_agreement_is_none():
    judge = make_judge([GATE_OK, "one", "two"])
    result = shaggy_dog(make_storyteller(), judge, k=2, rng=random.Random(0))
    assert result.metrics["n_judges"] == 1
    assert result.metrics["mean_cross_judge_agreement"] is None
