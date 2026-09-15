import json

import numpy as np
import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.tasks import free_association, style_transfer


@pytest.mark.parametrize("response", ["", "123", "two words", "Apple!", "I refuse to answer"])
def test_invalid_association_cannot_score(response):
    result = free_association(FakeClient(lambda _: response), n_words=3)
    assert result.score == 0
    assert result.metrics["invalid_rate"] == 1
    assert result.metrics["unique_words"] == 0
    assert [a["raw_response"] for a in result.details["attempts"]] == [response] * 3


def test_invalid_association_does_not_advance_survival():
    responses = iter(["apple", "", "banana", "apple"])
    result = free_association(FakeClient(lambda _: next(responses)), n_words=4)
    assert result.score == 0.25
    assert result.metrics["first_repeat_index"] == 3
    assert result.metrics["unique_words"] == 2


@pytest.mark.parametrize("field", ["plot_preserved", "genre_achieved", "comprehensible"])
@pytest.mark.parametrize("value", [False, "true", 1, None])
def test_style_transfer_invalid_or_unresolved_never_rewards_distance(field, value):
    verdict = dict(plot_preserved=True, genre_achieved=True, comprehensible=True, reason="Evidence")
    verdict[field] = value
    raw = json.dumps(verdict)
    judge = FakeClient(lambda _: raw)
    result = style_transfer(
        FakeClient(lambda m: "summary" if "Summarize" in m[-1]["content"] else "unrelated"),
        FakeEmbedder(
            fixed={
                "original": np.array([1, 0]),
                "summary": np.array([1, 0]),
                "unrelated": np.array([-1, 0]),
            }
        ),
        judge_client=judge,
        stories=[dict(text="original", genre="horror")],
        genres=["horror", "comedy"],
    )
    assert result.score == 0
    assert result.metrics["mean_divergence"] == pytest.approx(2)
    assert result.metrics["validity_rate"] == 0
    record = result.details["transfers"][0]
    assert record["original"] == "original"
    assert record["judge_attempts"][0] == raw
    assert result.metrics["judge_unresolved"] == (0 if value is False else 1)
