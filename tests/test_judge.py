import pytest
from conftest import FakeClient

from creativity_bench.judge import extract_json_object, judge_edit


def test_judge_parses_clean_json():
    client = FakeClient(
        lambda _: '{"coherent": true, "edits_applied": true, "quality_maintained": true}'
    )
    verdict = judge_edit(client, "orig", "mod", ["add humor"])
    assert verdict.coherent and verdict.edits_applied and verdict.quality_maintained
    assert verdict.passed


def test_low_quality_edit_fails_even_if_applied():
    # Gwern: the run ends when "the edit fails or the quality is low", so a
    # correctly-applied but low-quality edit does not pass.
    client = FakeClient(
        lambda _: '{"coherent": true, "edits_applied": true, "quality_maintained": false}'
    )
    verdict = judge_edit(client, "orig", "mod", ["add humor"])
    assert not verdict.passed


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


@pytest.mark.parametrize("value", ['"false"', "0", "null", "[]"])
def test_judge_rejects_non_boolean_verdicts(value):
    client = FakeClient(
        lambda _: '{"coherent": ' + value + ', "edits_applied": true, "quality_maintained": true}'
    )
    with pytest.raises(RuntimeError, match="unparseable"):
        judge_edit(client, "orig", "mod", ["edit"])


def test_extract_returns_answer_object_not_schema_example():
    text = (
        "Recall the schema:"
        ' {"coherent": true, "edits_applied": true, "quality_maintained": true}.'
        " My answer:"
        ' {"coherent": false, "edits_applied": false, "quality_maintained": false}'
    )
    assert extract_json_object(text) == {
        "coherent": False,
        "edits_applied": False,
        "quality_maintained": False,
    }


def test_judge_answers_after_restating_schema_with_placeholders():
    client = FakeClient(
        lambda _: (
            "Recall the schema:"
            ' {"coherent": <bool>, "edits_applied": <bool>, "quality_maintained": <bool>}.'
            " My answer:"
            ' {"coherent": true, "edits_applied": false, "quality_maintained": true}'
        )
    )
    verdict = judge_edit(client, "orig", "mod", ["edit"])
    assert not verdict.edits_applied
    assert not verdict.passed


def test_extract_parses_code_fenced_object():
    text = (
        "Assessment:\n```json\n"
        '{"coherent": true, "edits_applied": true, "quality_maintained": false}'
        "\n```"
    )
    assert extract_json_object(text)["quality_maintained"] is False


def test_extract_ignores_braces_inside_string_literals():
    text = '{"note": "closing } brace and opening { brace quoted", "coherent": false}'
    assert extract_json_object(text)["coherent"] is False


def test_extract_tolerates_trailing_prose_after_object():
    text = (
        '{"coherent": true, "edits_applied": true, "quality_maintained": true}'
        "\n\nNote: the story keeps its ending, so quality holds."
    )
    assert extract_json_object(text)["quality_maintained"] is True


def test_extract_raises_on_unparseable_text():
    with pytest.raises(ValueError, match="No JSON object"):
        extract_json_object("I cannot answer in JSON, sorry.")
