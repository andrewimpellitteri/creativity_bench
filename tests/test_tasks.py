import random

import numpy as np
import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.tasks import (
    camels_back,
    dont_repeat_yourself,
    free_association,
    style_transfer,
    telephone_game,
)

PASS_VERDICT = '{"coherent": true, "edits_applied": true, "quality_maintained": true}'
FAIL_VERDICT = '{"coherent": false, "edits_applied": false, "quality_maintained": false}'


# --- free association -------------------------------------------------------


def test_free_association_all_unique():
    words = iter(
        [
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
            "golf",
            "hotel",
            "india",
            "juliet",
        ]
    )
    client = FakeClient(lambda _: next(words))
    result = free_association(client, n_words=10)
    assert result.score == 1.0
    assert result.metrics["unique_words"] == 10
    assert result.metrics["first_repeat_index"] is None


def test_free_association_with_repeats():
    script = ["apple", "banana", "apple", "cherry", "banana"]
    responses = iter(script)
    client = FakeClient(lambda _: next(responses))
    result = free_association(client, n_words=5)
    assert result.metrics["unique_words"] == 3
    assert result.metrics["first_repeat_index"] == 2
    assert result.score == pytest.approx(3 / 5)


def test_free_association_strips_noise():
    responses = iter(['"Apple."', "  BANANA!  ", "self-aware"])
    client = FakeClient(lambda _: next(responses))
    result = free_association(client, n_words=3)
    assert result.details["words"] == ["apple", "banana", "self-aware"]


# --- telephone game ---------------------------------------------------------


def test_telephone_converges_when_summary_stops_changing():
    # Model always produces the same summary -> semantic and lexical sim are 1.0
    def responder(messages):
        prompt = messages[-1]["content"]
        return "A cat chased a mouse." if "Summarize" in prompt else "story text"

    client = FakeClient(responder)
    embedder = FakeEmbedder(fixed={"cat": np.array([1.0, 0.0])})
    result = telephone_game(client, embedder, seed_text="A cat chased a mouse.", max_iter=5)
    assert result.metrics["iterations_survived"] == 0
    assert result.score == 0.0


def test_telephone_survives_when_drifting():
    counter = {"n": 0}

    def responder(messages):
        prompt = messages[-1]["content"]
        if "Summarize" in prompt:
            counter["n"] += 1
            return f"A completely different summary number {counter['n']}."
        return "story text"

    client = FakeClient(responder)
    # Hash-based embeddings: distinct summaries land far apart -> never converges
    result = telephone_game(client, FakeEmbedder(), seed_text="seed story", max_iter=4)
    assert result.metrics["iterations_survived"] == 4
    assert result.score == 1.0


def test_telephone_rejects_empty_seed():
    client = FakeClient(lambda _: "text")
    with pytest.raises(ValueError):
        telephone_game(client, FakeEmbedder(), seed_text="   ")


# --- camel's back -----------------------------------------------------------


def make_judge(verdicts):
    responses = iter(verdicts)
    return FakeClient(lambda _: next(responses))


def test_camels_back_counts_rounds_until_failure():
    client = FakeClient(lambda _: "a story, slightly modified")
    judge = make_judge([PASS_VERDICT, PASS_VERDICT, FAIL_VERDICT])
    result = camels_back(
        client,
        judge,
        seed_text="premise",
        edit_requests=["a", "b", "c", "d"],
        max_edits=5,
        rng=random.Random(0),
    )
    assert result.metrics["rounds_survived"] == 2
    assert result.score == pytest.approx(2 / 5)
    assert len(result.details["rounds"]) == 3


def test_camels_back_perfect_run():
    client = FakeClient(lambda _: "modified story")
    judge = make_judge([PASS_VERDICT] * 3)
    result = camels_back(
        client,
        judge,
        seed_text="premise",
        edit_requests=["a", "b", "c", "d"],
        max_edits=3,
        rng=random.Random(0),
    )
    assert result.score == 1.0


# --- diversity --------------------------------------------------------------


def test_diversity_orthogonal_stories_score_high():
    counter = {"n": 0}

    def responder(_):
        counter["n"] += 1
        return f"story {counter['n']}"

    fixed = {f"story {i}": np.eye(4)[i - 1] for i in range(1, 5)}
    result = dont_repeat_yourself(
        FakeClient(responder), FakeEmbedder(fixed=fixed), samples=4, rng=random.Random(0)
    )
    assert result.score == pytest.approx(1.0)


def test_diversity_identical_stories_score_zero():
    client = FakeClient(lambda _: "the same story every time")
    result = dont_repeat_yourself(client, FakeEmbedder(), samples=3, rng=random.Random(0))
    assert result.score == pytest.approx(0.0)


def test_diversity_requires_two_samples():
    with pytest.raises(ValueError):
        dont_repeat_yourself(FakeClient(lambda _: "x"), FakeEmbedder(), samples=1)


# --- style transfer ---------------------------------------------------------


def test_style_transfer_scores_divergence():
    def responder(messages):
        prompt = messages[-1]["content"]
        return "the summary text" if "Summarize" in prompt else "transferred story"

    fixed = {
        "original tale": np.array([1.0, 0.0, 0.0]),
        "transferred story": np.array([0.0, 1.0, 0.0]),
        "the summary text": np.array([0.0, 1.0, 0.0]),
    }
    stories = [{"genre": "horror", "text": "original tale"}]
    result = style_transfer(
        FakeClient(responder),
        FakeEmbedder(fixed=fixed),
        stories=stories,
        genres=["horror", "noir"],
        rng=random.Random(0),
    )
    assert result.score == pytest.approx(1.0)  # orthogonal to original
    assert result.metrics["mean_fidelity"] == pytest.approx(1.0)  # identical to summary
    assert result.details["transfers"][0]["target_genre"] == "noir"
