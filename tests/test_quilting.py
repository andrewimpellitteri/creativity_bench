from __future__ import annotations

import random

import numpy as np
import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.data import QUILT_FRAGMENTS
from creativity_bench.tasks.quilting import quilting

FRAGMENTS = [
    {"id": "F1", "text": "the last ferry had already gone"},
    {"id": "F2", "text": "a key that fit nothing"},
    {"id": "F3", "text": "the dog knew first"},
    {"id": "F4", "text": "there was salt on the windows"},
    {"id": "F5", "text": "nobody had watered the plants"},
]

PASS = '{"comprehensible": true, "integrated": true}'


def _response(ids, story_tag="story", fragments=FRAGMENTS):
    by_id = {f["id"]: f["text"] for f in fragments}
    listing = "\n".join(f"- {by_id[i]}" for i in ids)
    used = " ".join(f"Then {by_id[i]}." for i in ids)
    return f"FRAGMENTS:\n{listing}\n\nSTORY:\n{story_tag}: {used}"


def writer(sequence):
    """Return each canned response in turn, and PASS for judge prompts."""
    state = {"index": 0}

    def respond(messages):
        prompt = messages[-1]["content"]
        if "comprehensible" in prompt:
            return PASS
        reply = sequence[min(state["index"], len(sequence) - 1)]
        state["index"] += 1
        return reply

    return respond


def run(sequence, **kwargs):
    responder = kwargs.pop("responder", None) or writer(sequence)
    client = FakeClient(responder)
    defaults = dict(
        embedder=FakeEmbedder(),
        judge_client=client,
        fragments=FRAGMENTS,
        runs=len(sequence),
        subset_size=2,
        rng=random.Random(0),
    )
    defaults.update(kwargs)
    return quilting(client, **defaults)


def test_distinct_recipes_and_distinct_stories_score_high():
    result = run(
        [
            _response(["F1", "F2"], "alpha"),
            _response(["F3", "F4"], "beta"),
        ]
    )
    assert result.metrics["validity_rate"] == 1.0
    assert result.metrics["unique_recipes"] == 2
    assert result.metrics["selection_diversity"] == 1.0
    assert result.metrics["story_diversity"] > 0.2
    assert result.score > 0.4
    assert result.details["fragment_usage"] == {"F1": 1, "F2": 1, "F3": 1, "F4": 1}


def test_repeated_recipe_lowers_selection_diversity():
    result = run([_response(["F1", "F2"], "alpha"), _response(["F1", "F2"], "beta")])
    assert result.metrics["unique_recipes"] == 1
    assert result.metrics["selection_diversity"] == 0.5


def test_identical_stories_collapse_embedding_diversity():
    same = _response(["F1", "F2"], "identical")
    result = run([same, same], embedder=FakeEmbedder())
    assert result.metrics["story_diversity"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["selection_diversity"] == 0.5
    assert result.score == pytest.approx(0.25, abs=1e-6)


def test_fragment_listed_but_not_used_fails_the_gate():
    listing = "FRAGMENTS:\n- the last ferry had already gone\n- a key that fit nothing"
    result = run([f"{listing}\n\nSTORY:\nNothing here quotes anything at all."])
    assert result.score == 0.0
    assert result.details["runs"][0]["failed_gates"] == ["fragment_not_used"]
    assert result.details["runs"][0]["unused_fragments"] == ["F1", "F2"]


def test_wrong_subset_size_fails_the_gate():
    result = run([_response(["F1", "F2", "F3"], "alpha")])
    assert "wrong_subset_size" in result.details["runs"][0]["failed_gates"]
    assert result.score == 0.0


def test_malformed_response_is_invalid_not_fatal():
    result = run(["I chose some fragments and wrote a story about them."])
    assert result.score == 0.0
    assert "malformed_response" in result.details["runs"][0]["failed_gates"]


def test_single_valid_run_is_degenerate_and_flagged():
    result = run([_response(["F1", "F2"], "alpha")])
    assert result.metrics["degenerate"] is True
    assert result.metrics["story_diversity"] is None
    assert result.score == 1.0  # gate result only, not a diversity measurement


def test_unresolved_judgment_is_counted_and_uncredited():
    def responder(messages):
        prompt = messages[-1]["content"]
        if "comprehensible" in prompt:
            return "no verdict here"
        return _response(["F1", "F2"], "alpha")

    result = run([""] * 2, responder=responder)
    assert result.metrics["unresolved_judgments"] == 2
    assert result.score == 0.0


def test_matching_ignores_case_and_punctuation():
    listing = "FRAGMENTS:\n- The Last Ferry Had Already Gone!\n- 'a key that fit nothing'"
    story = "STORY:\nThe last ferry had already gone, and a key that fit nothing lay there."
    result = run([f"{listing}\n\n{story}"])
    assert result.details["runs"][0]["chosen_ids"] == ["F1", "F2"]
    assert result.details["runs"][0]["validity_status"] == "valid"


def test_shuffle_order_is_recorded_and_varies():
    result = run([_response(["F1", "F2"], "alpha"), _response(["F3", "F4"], "beta")])
    orders = [r["shown_order"] for r in result.details["runs"]]
    assert all(sorted(o) == sorted(f["id"] for f in FRAGMENTS) for o in orders)
    assert orders[0] != orders[1]


def test_generation_error_is_recorded():
    def boom(messages):
        if "comprehensible" in messages[-1]["content"]:
            return PASS
        raise RuntimeError("upstream 500")

    result = run([""], responder=boom)
    assert result.metrics["generation_errors"] == 1
    assert result.score == 0.0


def test_requires_judge_and_enough_fragments():
    with pytest.raises(ValueError, match="judge_client"):
        quilting(FakeClient(writer([])), embedder=FakeEmbedder(), fragments=FRAGMENTS)
    with pytest.raises(ValueError, match="subset_size fragments"):
        quilting(
            FakeClient(writer([])),
            embedder=FakeEmbedder(),
            judge_client=FakeClient(writer([])),
            fragments=FRAGMENTS[:2],
            subset_size=4,
        )


def test_shipped_fragment_pool_is_usable():
    assert len(QUILT_FRAGMENTS) >= 16
    ids = [f["id"] for f in QUILT_FRAGMENTS]
    assert len(ids) == len(set(ids))
    client = FakeClient(writer([_response(["F01", "F02"], "alpha", QUILT_FRAGMENTS)]))
    result = quilting(
        client,
        embedder=FakeEmbedder(),
        judge_client=client,
        fragments=QUILT_FRAGMENTS,
        runs=1,
        subset_size=2,
        rng=random.Random(1),
    )
    assert result.details["runs"][0]["chosen_ids"] == ["F01", "F02"]


def test_score_is_bounded():
    result = run([_response(["F1", "F2"], f"tag{i}") for i in range(4)])
    assert 0.0 <= result.score <= 1.0
    assert np.isfinite(result.score)
