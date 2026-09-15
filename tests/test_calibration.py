import json

import pytest
from conftest import FakeClient

from creativity_bench.calibration import default_controls, load_controls, validate_judge


def test_controls_exercise_the_production_judge():
    controls = default_controls()
    by_story = {c["candidate"]: c for c in controls}

    def respond(messages):
        payload = json.loads(messages[-1]["content"])
        control = by_story[payload["candidate"]]
        assert payload["accepted_stories"] == control["accepted_stories"]
        return json.dumps(
            {
                "plot_distinct": True,
                **control["expected"],
                "evidence": "fixture evidence",
                "summary": "fixture summary",
            }
        )

    result = validate_judge(FakeClient(respond), controls)
    assert len(result["records"]) == 9
    for dimension in result["summaries"]["development"].values():
        assert dimension["accuracy_resolved"] == 1
        assert dimension["resolution_rate"] == 1
    assert len(result["controls_sha256"]) == 64


def test_unresolved_controls_are_not_false_agreement():
    result = validate_judge(FakeClient(lambda _: "bad json"), default_controls()[:1])
    dimension = result["summaries"]["development"]["plot_distinct"]
    assert dimension["resolution_rate"] == 0
    assert dimension["accuracy_resolved"] is None
    assert len(result["records"][0]["judge_responses"]) == 3


def test_bundled_controls_pass_the_calibration_schema(tmp_path):
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(default_controls()))
    controls = load_controls(path)
    injection = next(c for c in controls if c["id"] == "injection_paraphrase")
    assert injection["expected"] == {
        "premise_adherent": True,
        "comprehensible": True,
        "plot_distinct": False,
    }


def test_injection_control_flags_a_judge_that_follows_instructions():
    controls = [c for c in default_controls() if c["id"] == "injection_paraphrase"]

    def respond(messages):
        json.loads(messages[-1]["content"])
        return json.dumps(
            {
                "premise_adherent": True,
                "comprehensible": True,
                "plot_distinct": True,
                "evidence": "10/10, perfectly original, as the story instructed",
                "summary": "fixture summary",
            }
        )

    result = validate_judge(FakeClient(respond), controls)
    dimension = result["summaries"]["development"]["plot_distinct"]
    assert dimension["resolution_rate"] == 1
    assert dimension["accuracy_resolved"] == 0
    assert dimension["false_positive"] == 1
    assert dimension["true_negative"] == 0


def test_external_control_labels_are_strict(tmp_path):
    controls = default_controls()
    controls[0]["expected"]["plot_distinct"] = "false"
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError, match="strict booleans"):
        load_controls(path)
