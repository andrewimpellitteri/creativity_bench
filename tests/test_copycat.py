from __future__ import annotations

import json
import random
import re

import pytest
from conftest import FakeClient

from creativity_bench.data import COPYCAT_OPENINGS
from creativity_bench.tasks.copycat import copycat

OPENINGS = [
    {"id": "noir", "voice": "noir", "text": "The rain had opinions about my client."},
    {"id": "folk", "voice": "folktale", "text": "Now in that country there lived a miller."},
    {"id": "memo", "voice": "memo", "text": "INCIDENT REPORT 12-B. The corridor was measured."},
]


def _continuation_for(prompt: str) -> str | None:
    match = re.search(r"OPENING:\n(.*)", prompt, re.DOTALL)
    return match.group(1).strip() if match else None


def matching_judge(accuracy: str = "perfect", comprehensible: bool = True):
    """Judge that recovers the true opening from a tagged continuation."""

    def respond(messages):
        prompt = messages[-1]["content"]
        if '"opening"' not in prompt:  # writer call: tag the continuation
            opening = _continuation_for(prompt)
            tag = next(o["id"] for o in OPENINGS if o["text"] == opening)
            return f"[[{tag}]] a continuation long enough to not restate the opening at all."
        tag = re.search(r"\[\[(\w+)\]\]", prompt).group(1)
        labels = re.findall(r"^(\d+)\. (.*)$", prompt, re.MULTILINE)
        truth = next(int(label) for label, text in labels if text.startswith(_text_for(tag)[:20]))
        # "chance": always answer with label 1, a matcher that cannot tell voices apart.
        choice = truth if accuracy == "perfect" else 1
        return json.dumps({"opening": choice, "comprehensible": comprehensible})

    return respond


def _text_for(tag: str) -> str:
    return next(o["text"] for o in OPENINGS if o["id"] == tag)


def run(responder=None, **kwargs):
    defaults = dict(
        judge_client=FakeClient(responder or matching_judge()),
        openings=OPENINGS,
        n_openings=3,
        rng=random.Random(0),
    )
    defaults.update(kwargs)
    client = defaults.pop("client", FakeClient(responder or matching_judge()))
    return copycat(client, **defaults)


def test_perfect_matching_scores_one():
    result = run()
    assert result.score == pytest.approx(1.0)
    assert result.metrics["matching_accuracy"] == 1.0
    assert result.metrics["chance_level"] == pytest.approx(1 / 3)
    assert result.metrics["validity_rate"] == 1.0


def test_chance_level_matching_scores_zero():
    """One continuation in three lands on label 1 by luck: exactly chance."""
    result = run(responder=matching_judge(accuracy="chance"))
    assert result.metrics["matching_accuracy"] == pytest.approx(1 / 3)
    assert result.score == pytest.approx(0.0)


def test_incomprehensible_continuation_is_invalid_and_uncredited():
    result = run(responder=matching_judge(comprehensible=False))
    assert result.score == 0.0
    assert result.metrics["validity_rate"] == 0.0
    assert all(r["failed_gates"] == ["comprehensible"] for r in result.details["openings"])


def test_restated_opening_fails_before_judging():
    def echo(messages):
        prompt = messages[-1]["content"]
        opening = _continuation_for(prompt)
        return opening if opening else '{"opening": 1, "comprehensible": true}'

    result = run(client=FakeClient(echo))
    assert result.score == 0.0
    assert result.metrics["restatements"] == 3
    assert result.details["openings"][0]["failed_gates"] == ["restates_opening"]


def test_unresolved_judgment_counts_as_a_miss_and_is_reported():
    def bad_judge(messages):
        prompt = messages[-1]["content"]
        if '"opening"' not in prompt:
            return "a perfectly ordinary continuation of some length here."
        return "I cannot answer that"

    result = run(responder=bad_judge)
    assert result.score == 0.0
    assert result.metrics["unresolved_judgments"] == 3


def test_out_of_range_label_is_rejected():
    def wild_judge(messages):
        prompt = messages[-1]["content"]
        if '"opening"' not in prompt:
            return "a perfectly ordinary continuation of some length here."
        return '{"opening": 99, "comprehensible": true}'

    result = run(responder=wild_judge)
    assert result.metrics["unresolved_judgments"] == 3


def test_label_permutation_is_saved_for_rejudging():
    result = run()
    permutations = [r["label_permutation"] for r in result.details["openings"]]
    assert all(sorted(p) == sorted(o["id"] for o in OPENINGS) for p in permutations)


def test_requires_judge_and_two_openings():
    with pytest.raises(ValueError, match="judge_client"):
        copycat(FakeClient(matching_judge()), openings=OPENINGS)
    with pytest.raises(ValueError, match="at least 2 openings"):
        copycat(
            FakeClient(matching_judge()),
            judge_client=FakeClient(matching_judge()),
            openings=OPENINGS,
            n_openings=1,
        )


def test_shipped_openings_have_distinct_ids_and_voices():
    ids = [o["id"] for o in COPYCAT_OPENINGS]
    assert len(ids) == len(set(ids)) >= 5
    assert all(o["text"].strip() and o["voice"].strip() for o in COPYCAT_OPENINGS)
