import json
import re

import pytest
from conftest import FakeClient

from creativity_bench.calibration import (
    CONTROL_OPENINGS,
    COPYCAT,
    QUILTING,
    THIS_AND_THAT,
    default_controls,
    gate_failures,
    gate_names,
    load_controls,
    validate_gates,
    validate_judge,
)
from creativity_bench.tasks.quilting import QUILT_JUDGE_PROMPT
from creativity_bench.tasks.this_and_that import BLEND_JUDGE_PROMPT


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


# ---------------------------------------------------------------------------
# Gates for the three 2026-09-16 tasks
# ---------------------------------------------------------------------------

# The text the gate is asked about, and the marker after which it appears in the
# production prompt. Controls are identified from the prompt itself, so a test
# judge cannot cheat by reading anything the real judge would not see.
_MARKER = {
    THIS_AND_THAT: "CANDIDATE STORY:",
    COPYCAT: "Here is one continuation:",
    QUILTING: "\nSTORY:\n",
}
_SUBJECT = {THIS_AND_THAT: "candidate", COPYCAT: "continuation", QUILTING: "story"}


def _control_for(gate: str, prompt: str) -> dict:
    tail = prompt.split(_MARKER[gate])[-1]
    # Longest first: one control's subject text can embed another's (an injected
    # copy of example A contains example A).
    for control in sorted(
        default_controls(gate), key=lambda c: len(c[_SUBJECT[gate]]), reverse=True
    ):
        if control[_SUBJECT[gate]] in tail:
            return control
    raise AssertionError(f"No {gate} control matches the judge prompt")


def _label_of(prompt: str, opening_id: str) -> int:
    """The shuffled integer label the copycat matcher must answer with."""
    text = next(o["text"] for o in CONTROL_OPENINGS if o["id"] == opening_id)
    labels = re.findall(r"^(\d+)\. (.*)$", prompt, re.MULTILINE)
    return next(int(label) for label, shown in labels if shown.strip() == text)


def _expected_verdict(gate: str, control: dict, prompt: str) -> dict:
    expected = control["expected"]
    if gate == COPYCAT:
        return {
            "opening": _label_of(prompt, control["expected_opening_id"]),
            "comprehensible": expected["comprehensible"],
        }
    # Dimensions left unlabeled are undefined for that control; answer anything.
    fields = ("draws_on_a", "draws_on_b", "comprehensible") if gate == THIS_AND_THAT else ()
    fields = fields or ("comprehensible", "integrated")
    return {field: expected.get(field, True) for field in fields}


def _agreeing_judge(gate: str):
    """A judge that returns exactly the control's proposed development label."""

    def respond(messages):
        prompt = messages[-1]["content"]
        return json.dumps(_expected_verdict(gate, _control_for(gate, prompt), prompt))

    return respond


@pytest.mark.parametrize("gate", [THIS_AND_THAT, COPYCAT, QUILTING])
def test_new_gate_controls_are_development_split_and_fully_labeled(gate):
    controls = default_controls(gate)
    assert controls
    assert {c["split"] for c in controls} == {"development"}
    assert {c["gate"] for c in controls} == {gate}
    assert len({c["id"] for c in controls}) == len(controls)
    # Every control labels comprehensibility; the gate-specific axes may be
    # deliberately unlabeled where they are undefined.
    assert all("comprehensible" in c["expected"] for c in controls)


@pytest.mark.parametrize("gate", [THIS_AND_THAT, COPYCAT, QUILTING])
def test_agreeing_judge_scores_perfectly_on_every_new_gate(gate):
    result = validate_judge(FakeClient(_agreeing_judge(gate)), default_controls(gate), gate=gate)
    assert result["gate"] == gate
    assert len(result["records"]) == len(default_controls(gate))
    for name, dimension in result["summaries"]["development"].items():
        assert dimension["resolution_rate"] == 1, name
        assert dimension["accuracy_resolved"] == 1, name
    assert gate_failures(result) == []


@pytest.mark.parametrize("gate", [THIS_AND_THAT, COPYCAT, QUILTING])
def test_unresolved_new_gate_judgments_are_not_agreement(gate):
    result = validate_judge(FakeClient(lambda _: "no json here"), default_controls(gate), gate=gate)
    for dimension in result["summaries"]["development"].values():
        assert dimension["resolved"] == 0
        assert dimension["resolution_rate"] == 0
        assert dimension["accuracy_resolved"] is None
        assert dimension["true_positive"] == dimension["false_positive"] == 0
    # One parse retry, as in the task loop.
    assert all(len(r["judge_attempts"]) == 2 for r in result["records"])
    assert any("resolved" in reason for reason in gate_failures(result))


@pytest.mark.parametrize("gate", [THIS_AND_THAT, COPYCAT, QUILTING])
def test_new_gate_controls_round_trip_through_the_external_schema(tmp_path, gate):
    path = tmp_path / f"{gate}.json"
    path.write_text(json.dumps(default_controls(gate)))
    assert load_controls(path, gate=gate) == default_controls(gate)


def test_blend_controls_use_the_production_prompt_verbatim():
    client = FakeClient(_agreeing_judge(THIS_AND_THAT))
    validate_judge(client, default_controls(THIS_AND_THAT), gate=THIS_AND_THAT)
    control = next(c for c in default_controls(THIS_AND_THAT) if c["id"] == "genuine_blend")
    expected = BLEND_JUDGE_PROMPT.format(
        story_a=control["story_a"], story_b=control["story_b"], candidate=control["candidate"]
    )
    assert any(call[-1]["content"] == expected for call in client.calls)


def test_quilt_controls_use_the_production_prompt_verbatim():
    client = FakeClient(_agreeing_judge(QUILTING))
    validate_judge(client, default_controls(QUILTING), gate=QUILTING)
    control = next(c for c in default_controls(QUILTING) if c["id"] == "genuine_integration")
    expected = QUILT_JUDGE_PROMPT.format(
        fragments="\n".join(f"- {text}" for text in control["fragments"]),
        story=control["story"],
    )
    assert any(call[-1]["content"] == expected for call in client.calls)


def test_blend_gate_catches_a_judge_that_answers_draws_on_a_by_position():
    """copy_of_b is the mirror of copy_of_a: a positional judge fails exactly one."""

    def respond(_messages):
        return json.dumps({"draws_on_a": True, "draws_on_b": False, "comprehensible": True})

    controls = [c for c in default_controls(THIS_AND_THAT) if c["id"] in ("copy_of_a", "copy_of_b")]
    result = validate_judge(FakeClient(respond), controls, gate=THIS_AND_THAT)
    dimensions = result["summaries"]["development"]
    assert dimensions["draws_on_a"]["false_positive"] == 1
    assert dimensions["draws_on_b"]["false_negative"] == 1
    assert gate_failures(result)


def test_blend_gate_catches_a_judge_that_obeys_the_injected_story():
    controls = [c for c in default_controls(THIS_AND_THAT) if c["id"] == "injected_copy_of_a"]

    def respond(_messages):
        return json.dumps({"draws_on_a": True, "draws_on_b": True, "comprehensible": True})

    result = validate_judge(FakeClient(respond), controls, gate=THIS_AND_THAT)
    dimension = result["summaries"]["development"]["draws_on_b"]
    assert dimension["resolution_rate"] == 1
    assert dimension["accuracy_resolved"] == 0
    assert dimension["false_positive"] == 1


def test_copycat_gate_catches_a_matcher_that_keys_on_topic():
    """swapped_voice is hardboiled prose about the Victorian letter's subject."""
    controls = [c for c in default_controls(COPYCAT) if c["id"] == "swapped_voice"]

    def respond(messages):
        prompt = messages[-1]["content"]
        return json.dumps({"opening": _label_of(prompt, "victorian"), "comprehensible": True})

    result = validate_judge(FakeClient(respond), controls, gate=COPYCAT)
    dimension = result["summaries"]["development"]["matched"]
    assert dimension["resolution_rate"] == 1
    assert dimension["false_negative"] == 1
    assert dimension["accuracy_resolved"] == 0
    assert result["records"][0]["chosen_id"] == "victorian"


def test_copycat_leaves_matching_unlabeled_where_ground_truth_is_destroyed():
    """A voice-stripped or nonsense continuation has no recoverable true opening."""
    controls = {c["id"]: c for c in default_controls(COPYCAT)}
    assert "matched" not in controls["house_voice"]["expected"]
    assert "matched" not in controls["nonsense"]["expected"]
    result = validate_judge(
        FakeClient(_agreeing_judge(COPYCAT)), default_controls(COPYCAT), gate=COPYCAT
    )
    dimensions = result["summaries"]["development"]
    assert dimensions["matched"]["labeled"] == 3
    assert dimensions["comprehensible"]["labeled"] == 5
    # The unscored choice is still recorded for inspection.
    house = next(r for r in result["records"] if r["id"] == "house_voice")
    assert house["chosen_id"] in {o["id"] for o in house["openings"]}


def test_copycat_label_shuffle_is_seeded_and_reproducible():
    controls = default_controls(COPYCAT)
    runs = [
        validate_judge(FakeClient(_agreeing_judge(COPYCAT)), controls, gate=COPYCAT)
        for _ in range(2)
    ]
    permutations = [[r["label_permutation"] for r in run["records"]] for run in runs]
    assert permutations[0] == permutations[1]
    # Shuffled, not the corpus order: label position must carry no information.
    assert permutations[0][0] != [c["id"] for c in controls[0]["openings"]]


def test_quilting_gate_catches_a_judge_that_calls_an_appended_list_integrated():
    controls = [
        c for c in default_controls(QUILTING) if c["id"] in ("listed_not_woven", "injected_listing")
    ]

    def respond(_messages):
        return json.dumps({"comprehensible": True, "integrated": True})

    result = validate_judge(FakeClient(respond), controls, gate=QUILTING)
    dimension = result["summaries"]["development"]["integrated"]
    assert dimension["labeled"] == 2
    assert dimension["false_positive"] == 2
    assert dimension["accuracy_resolved"] == 0
    assert result["summaries"]["development"]["comprehensible"]["accuracy_resolved"] == 1


def test_quilting_leaves_integration_unlabeled_for_nonsense():
    control = next(c for c in default_controls(QUILTING) if c["id"] == "woven_nonsense")
    assert control["expected"] == {"comprehensible": False}


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: c.update(expected_opening_id="nope"), "must name one of the openings"),
        (lambda c: c.update(openings=[c["openings"][0]]), "at least two openings"),
        (lambda c: c.update(shuffle_seed="0"), "shuffle_seed must be an integer"),
        (lambda c: c.update(continuation="  "), "must be nonempty strings"),
        (lambda c: c.update(expected={"matched": "true"}), "strict booleans"),
        (lambda c: c.update(expected={"unknown": True}), "strict booleans"),
    ],
)
def test_copycat_external_controls_are_schema_validated(tmp_path, mutate, message):
    controls = default_controls(COPYCAT)
    mutate(controls[0])
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError, match=message):
        load_controls(path, gate=COPYCAT)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: c.update(fragments=[]), "nonempty list of nonempty strings"),
        (lambda c: c.update(fragments=["ok", 3]), "nonempty list of nonempty strings"),
        (lambda c: c.update(story=""), "must be nonempty strings"),
        (lambda c: c.update(split=""), "must be nonempty strings"),
    ],
)
def test_quilting_external_controls_are_schema_validated(tmp_path, mutate, message):
    controls = default_controls(QUILTING)
    mutate(controls[0])
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError, match=message):
        load_controls(path, gate=QUILTING)


def test_blend_external_controls_require_both_examples(tmp_path):
    controls = default_controls(THIS_AND_THAT)
    del controls[0]["story_b"]
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError, match="must be nonempty strings"):
        load_controls(path, gate=THIS_AND_THAT)


def test_loading_an_unknown_gate_is_refused():
    with pytest.raises(ValueError, match="Unknown gate"):
        load_controls(gate="taste")


def test_controls_declaring_a_gate_override_the_load_default(tmp_path):
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(default_controls(QUILTING)))
    # Items carry gate="quilting", so the SBD default does not apply to them.
    assert load_controls(path) == default_controls(QUILTING)


def test_mixed_gate_controls_must_be_split_explicitly():
    controls = default_controls(QUILTING) + default_controls(COPYCAT)
    with pytest.raises(ValueError, match="mix gates"):
        validate_judge(FakeClient(lambda _: "{}"), controls)


def test_a_control_from_another_gate_is_refused():
    controls = default_controls(QUILTING)[:1] + default_controls(COPYCAT)[:1]
    with pytest.raises(ValueError, match="is for gate"):
        validate_judge(FakeClient(lambda _: "{}"), controls, gate=QUILTING)


def test_validate_gates_summarizes_every_gate_in_one_result():
    def respond(messages):
        prompt = messages[-1]["content"]
        for gate in (THIS_AND_THAT, COPYCAT, QUILTING):
            if _MARKER[gate] in prompt:
                return json.dumps(_expected_verdict(gate, _control_for(gate, prompt), prompt))
        payload = json.loads(prompt)
        control = next(c for c in default_controls() if c["candidate"] == payload["candidate"])
        return json.dumps(
            {
                "plot_distinct": True,
                **control["expected"],
                "evidence": "fixture evidence",
                "summary": "fixture summary",
            }
        )

    result = validate_gates(FakeClient(respond))
    assert set(result["gates"]) == set(gate_names())
    assert result["kind"] == "judge_control_validation_suite"
    assert len(result["controls_sha256"]) == 64
    for gate, single in result["gates"].items():
        assert single["gate"] == gate
        assert single["summaries"]["development"]
        for dimension in single["summaries"]["development"].values():
            assert dimension["resolution_rate"] == 1
            assert dimension["accuracy_resolved"] == 1
    assert gate_failures(result) == []


def test_gate_failures_names_the_gate_and_dimension_that_blocks_a_pilot():
    result = validate_gates(
        FakeClient(lambda _: "not json"),
        {QUILTING: default_controls(QUILTING)},
    )
    failures = gate_failures(result)
    assert failures
    assert all(reason.startswith("quilting.") for reason in failures)


def test_gate_failures_reports_a_missing_split():
    controls = [{**c, "split": "held_out"} for c in default_controls(QUILTING)]
    result = validate_judge(FakeClient(_agreeing_judge(QUILTING)), controls, gate=QUILTING)
    assert result["summaries"]["held_out"]
    assert gate_failures(result) == ["quilting: no controls on split 'development'"]
    assert gate_failures(result, split="held_out") == []


def test_group_by_gate_splits_a_mixed_external_file(tmp_path):
    from creativity_bench.calibration import SAME_BUT_DIFFERENT, group_by_gate

    mixed = default_controls() + default_controls(COPYCAT) + default_controls(QUILTING)
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(mixed))
    grouped = group_by_gate(load_controls(path))
    assert set(grouped) == {SAME_BUT_DIFFERENT, COPYCAT, QUILTING}
    assert grouped[COPYCAT] == default_controls(COPYCAT)
    # Items with no "gate" key are Same But Different, as every legacy file is.
    assert grouped[SAME_BUT_DIFFERENT] == default_controls()
