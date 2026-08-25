"""Offline tests for the shaggy dog storytelling contest task."""

from __future__ import annotations

import random

import pytest
from conftest import FakeClient

from creativity_bench.tasks import TASKS
from creativity_bench.tasks.shaggy_dog import shaggy_dog


def make_storyteller(story="A man walks into a bar. Nothing comes of it. The end."):
    return FakeClient(lambda _: story)


def make_judge(responses):
    replies = iter(responses)
    return FakeClient(lambda _: next(replies))


def test_registered_in_tasks():
    assert "shaggy_dog" in TASKS
    assert TASKS["shaggy_dog"] is shaggy_dog


def test_varying_judges_score_high():
    # Judges that cannot agree on an interpretation => high score.
    storyteller = make_storyteller()
    judge = make_judge(
        [
            "Kindness towards animals, clearly.",
            "A submarine race, obviously.",
            "Nothing; it just ends abruptly.",
        ]
    )
    result = shaggy_dog(storyteller, judge, k=3, rng=random.Random(0))
    assert result.name == "shaggy_dog"
    assert result.score == pytest.approx(1.0)
    assert result.metrics["explicit_moral"] is False
    assert not result.metrics["degenerate"]
    assert len(result.details["explanations"]) == 3


def test_agreeing_judges_score_low():
    # Identical tidy interpretations => the storyteller steered its judges,
    # which Gwern scores as bad ("the more similar the explanations are, the
    # worse the score").
    shared = "The moral is to never trust a talking mule."
    judge = make_judge([shared, shared, shared])
    result = shaggy_dog(make_storyteller(), judge, k=3, rng=random.Random(0))
    assert result.score == pytest.approx(0.0)
    assert result.metrics["mean_pairwise_agreement"] == pytest.approx(1.0)


def test_partial_agreement_scores_between_extremes():
    explanation = "The story is about a farmer losing his hat."
    paraphrase = "The story concerns a farmer who lost his hat."
    judge = make_judge([explanation, paraphrase])
    result = shaggy_dog(make_storyteller(), judge, k=2, rng=random.Random(0))
    assert 0.0 < result.score < 1.0
    assert 0.0 < result.metrics["mean_pairwise_agreement"] < 1.0


def test_explicit_moral_is_automatic_failure():
    # Gwern: ChatGPT "wants to conclude with some clear moral or punchline,
    # and a tidy pat interpretation". A stated moral fails outright.
    moralizing_story = (
        "A tortoise raced a hare. In the end the tortoise won. "
        "Moral: slow and steady wins the race."
    )
    judge = make_judge(["anything", "else", "at all"])
    result = shaggy_dog(make_storyteller(moralizing_story), judge, k=3)
    assert result.score == 0.0
    assert result.metrics["explicit_moral"] is True
    assert judge.usage.requests == 0  # judges never consulted


def test_lesson_is_also_automatic_failure():
    story = "He climbed the mountain for no reason. The lesson here is perseverance matters."
    result = shaggy_dog(make_storyteller(story), make_judge([]), k=2)
    assert result.score == 0.0


def test_k_controls_number_of_judge_samples():
    judge = make_judge(["a", "b", "c"])
    shaggy_dog(make_storyteller(), judge, k=3, rng=random.Random(0))
    assert judge.usage.requests == 3


def test_k_one_degenerate_case():
    # With one sample no pairwise comparison exists; handled gracefully with
    # score 1.0 and an explicit degenerate flag (documented in module).
    judge = make_judge(["some interpretation"])
    result = shaggy_dog(make_storyteller(), judge, k=1, rng=random.Random(0))
    assert judge.usage.requests == 1
    assert result.score == 1.0
    assert result.metrics["degenerate"] is True


def test_k_zero_rejected():
    with pytest.raises(ValueError):
        shaggy_dog(make_storyteller(), make_judge([]), k=0)


def test_seeded_runs_are_reproducible():
    def run():
        storyteller = make_storyteller()
        judge = make_judge(
            [
                "One reading about loss.",
                "Another about hope entirely.",
                "Yet another about soup.",
            ]
        )
        result = shaggy_dog(storyteller, judge, k=3, rng=random.Random(42))
        return result.score

    assert run() == run()


def test_empty_explanations_trivially_agree():
    # Both judges return nothing at all: they agree on no interpretation,
    # which token-overlap treats as trivial agreement (low score).
    judge = make_judge(["", "", ""])
    result = shaggy_dog(make_storyteller(), judge, k=3, rng=random.Random(0))
    assert result.score == pytest.approx(0.0)
