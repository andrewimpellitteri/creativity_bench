import pytest
from conftest import FakeClient

from creativity_bench.judge import judge_edit


def test_judge_parses_clean_json():
    client = FakeClient(
        lambda _: '{"coherent": true, "edits_applied": true, "quality_maintained": false}'
    )
    verdict = judge_edit(client, "orig", "mod", ["add humor"])
    assert verdict.coherent and verdict.edits_applied and not verdict.quality_maintained
    assert verdict.passed


def test_judge_parses_json_with_surrounding_text():
    client = FakeClient(
        lambda _: (
            'Here is my assessment:\n{"coherent": false, "edits_applied": true,'
            ' "quality_maintained": true}\nDone.'
        )
    )
    verdict = judge_edit(client, "orig", "mod", ["edit"])
    assert not verdict.coherent
    assert not verdict.passed


def test_judge_raises_after_two_bad_responses():
    client = FakeClient(lambda _: "I cannot answer in JSON, sorry.")
    with pytest.raises(RuntimeError, match="unparseable"):
        judge_edit(client, "orig", "mod", ["edit"])
    assert client.usage.requests == 2
