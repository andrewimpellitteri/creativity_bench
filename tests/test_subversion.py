import re

import pytest
from conftest import FakeClient

from creativity_bench.tasks import subversion


def make_writer():
    """Writer fake: emits tagged stories and their subversions so the judge
    fake can tell within-pairs (story_i vs sub_i) from cross-pairs."""
    counter = {"n": 0}

    def responder(messages):
        prompt = messages[-1]["content"]
        if "opposite" in prompt:
            return f"subverted-{counter['n']}"
        counter["n"] += 1
        return f"original-{counter['n']}"

    return FakeClient(responder)


def make_judge(mode):
    """Judge fake: mode 'correct' flags only true within-pairs as opposite;
    'always_yes' flags everything; 'always_no' flags nothing."""

    def responder(messages):
        prompt = messages[-1]["content"]
        if mode == "always_no":
            verdict = False
        elif mode == "always_yes":
            verdict = True
        else:
            story = re.search(r"original-(\d+)", prompt)
            sub = re.search(r"subverted-(\d+)", prompt)
            verdict = story.group(1) == sub.group(1)
        return f'{{"opposite": {"true" if verdict else "false"}}}'

    return FakeClient(responder)


PREMISES = ["A mysterious letter arrives.", "The old clock ticks backward."]


def test_subversion_perfect_discrimination_scores_one():
    # Correct judge: all within-pairs flagged, no cross-pair false positives.
    result = subversion(make_writer(), make_judge("correct"), premises=PREMISES, runs=2)
    assert result.score == pytest.approx(1.0)
    assert result.metrics["within_opposite_rate"] == pytest.approx(1.0)
    assert result.metrics["cross_opposite_rate"] == pytest.approx(0.0)
    # n runs per premise -> n^2 judged pairs ("all the possible pairs").
    assert result.metrics["within_pairs"] == 4  # 2 premises x diagonal of 2 runs
    assert result.metrics["cross_pairs"] == 4  # 2 premises x off-diagonal 2x2
    assert len(result.details["pairs"]) == 8


def test_subversion_degenerate_always_opposite_scores_zero():
    # A judge that answers "opposite" to every pair gets tpr=1 but fpr=1:
    # Youden's J exposes it. This is why the cross-pairs exist.
    result = subversion(make_writer(), make_judge("always_yes"), premises=PREMISES, runs=2)
    assert result.score == pytest.approx(0.0)
    assert result.metrics["within_opposite_rate"] == pytest.approx(1.0)
    assert result.metrics["cross_opposite_rate"] == pytest.approx(1.0)


def test_subversion_never_opposite_scores_zero():
    result = subversion(make_writer(), make_judge("always_no"), premises=PREMISES, runs=2)
    assert result.score == pytest.approx(0.0)
    assert result.metrics["within_opposite_rate"] == pytest.approx(0.0)


def test_subversion_single_run_has_no_cross_pairs():
    # With runs=1 there are no cross-pairs, so the score falls back to the
    # raw within-pair hit rate (documented limitation in the module docstring).
    result = subversion(make_writer(), make_judge("correct"), premises=PREMISES, runs=1)
    assert result.score == pytest.approx(1.0)
    assert result.metrics["cross_pairs"] == 0
    assert result.metrics["cross_opposite_rate"] is None


def test_subversion_partial_within_hits_clamped():
    # Judge correct on cross-pairs but misses half the within-pairs:
    # score = tpr - fpr = 0.5 - 0.0.
    def responder(messages):
        prompt = messages[-1]["content"]
        story = re.search(r"original-(\d+)", prompt)
        sub = re.search(r"subverted-(\d+)", prompt)
        verdict = int(story.group(1)) % 2 == 0 and story.group(1) == sub.group(1)
        return f'{{"opposite": {"true" if verdict else "false"}}}'

    result = subversion(make_writer(), FakeClient(responder), premises=PREMISES, runs=2)
    assert result.score == pytest.approx(0.5)


def test_subversion_rejects_zero_runs():
    with pytest.raises(ValueError):
        subversion(make_writer(), make_judge("correct"), premises=PREMISES, runs=0)
