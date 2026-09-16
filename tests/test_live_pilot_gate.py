import importlib.util
from pathlib import Path

import pytest

from creativity_bench.calibration import gate_names

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_pilot.py"
SPEC = importlib.util.spec_from_file_location("live_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)

# Dimensions per gate, as the production summaries report them.
DIMENSIONS = {
    "same_but_different": ("premise_adherent", "comprehensible", "plot_distinct"),
    "this_and_that": ("draws_on_a", "draws_on_b", "comprehensible"),
    "copycat": ("matched", "comprehensible"),
    "quilting": ("comprehensible", "integrated"),
}


def perfect(labeled: int = 4) -> dict:
    return {
        "labeled": labeled,
        "resolved": labeled,
        "true_positive": labeled,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "accuracy_resolved": 1,
        "resolution_rate": 1,
    }


def gate_result(gate: str) -> dict:
    return {
        "kind": "judge_control_validation",
        "gate": gate,
        "summaries": {"development": {d: perfect() for d in DIMENSIONS[gate]}},
    }


def validation(**overrides) -> dict:
    """A passing four-gate suite, shaped like what `live_pilot.py validate` saves."""
    base = {
        "kind": "judge_control_validation_suite",
        "judge_model": "judge-a",
        "protocol_fingerprint": "fp",
        "controls_sha256": "sha",
        "gates": {gate: gate_result(gate) for gate in gate_names()},
    }
    base.update(overrides)
    return base


def check(v, judge="judge-a", fp="fp", sha="sha"):
    return pilot.validation_gate_error(v, judge=judge, fingerprint=fp, controls_sha=sha)


def test_passing_validation_gates_nothing():
    assert check(validation()) is None


def test_the_gate_covers_every_registered_judge_gate():
    assert set(validation()["gates"]) == set(gate_names())
    assert len(gate_names()) == 4


@pytest.mark.parametrize(
    "kwargs,judge,fp,sha",
    [
        ({"judge_model": "judge-b"}, "judge-a", "fp", "sha"),
        ({"protocol_fingerprint": "stale"}, "judge-a", "fp", "sha"),
        ({"controls_sha256": "different-controls"}, "judge-a", "fp", "sha"),
    ],
)
def test_identity_mismatches_are_rejected(kwargs, judge, fp, sha):
    assert check(validation(**kwargs), judge=judge, fp=fp, sha=sha)


def test_imperfect_same_but_different_control_rejects_with_matching_identity():
    v = validation()
    dimension = v["gates"]["same_but_different"]["summaries"]["development"]["plot_distinct"]
    dimension["accuracy_resolved"] = 0.875
    dimension["false_positive"] = 1
    error = check(v)
    assert error
    assert "same_but_different.plot_distinct" in error


@pytest.mark.parametrize("gate", ["this_and_that", "copycat", "quilting"])
def test_an_unresolved_new_gate_blocks_the_pilot(gate):
    """A gate whose resolution rate is below 1.0 blocks exactly as SBD does."""
    v = validation()
    dimension = v["gates"][gate]["summaries"]["development"][DIMENSIONS[gate][-1]]
    dimension.update(resolved=2, resolution_rate=0.5, accuracy_resolved=1)
    error = check(v)
    assert error
    assert f"{gate}.{DIMENSIONS[gate][-1]}: 2/4 judgments resolved" in error
    assert "unresolved judgments are not agreement" in error


@pytest.mark.parametrize("gate", ["this_and_that", "copycat", "quilting"])
def test_a_disagreeing_new_gate_blocks_the_pilot(gate):
    v = validation()
    dimension = v["gates"][gate]["summaries"]["development"][DIMENSIONS[gate][0]]
    dimension.update(accuracy_resolved=0.75, true_positive=3, false_positive=1)
    error = check(v)
    assert error
    assert f"{gate}.{DIMENSIONS[gate][0]}" in error
    assert "false positive" in error


def test_a_same_but_different_only_validation_no_longer_covers_the_pilot():
    """The pre-four-gate shape is rejected by name, not silently accepted."""
    v = gate_result("same_but_different")
    v.update(judge_model="judge-a", protocol_fingerprint="fp", controls_sha256="sha")
    error = check(v)
    assert error.startswith("Validation does not cover judge gate(s):")
    for gate in ("this_and_that", "copycat", "quilting"):
        assert gate in error


# --- the controls-hash gate ---------------------------------------------------


def test_bundled_same_but_different_controls_are_unchanged_by_the_new_gates():
    """The hash the saved pilot/extended validations carry must still match."""
    from creativity_bench.calibration import load_controls

    assert (
        pilot.controls_sha256(load_controls())
        == "1d51d602faa373ceba5a906438f200c8f581680c69b356994ddb47c9c5781145"
    )


def test_expected_hash_follows_the_shape_of_what_was_validated():
    from creativity_bench.calibration import all_default_controls, load_controls

    suite = validation()
    assert pilot.expected_controls_sha(suite) == pilot.controls_sha256(all_default_controls())
    single = gate_result("quilting")
    assert pilot.expected_controls_sha(single) == pilot.controls_sha256(
        load_controls(gate="quilting")
    )
    # The four-gate hash covers every gate's controls, so a validation run over a
    # subset cannot pass the hash check.
    assert pilot.expected_controls_sha(suite) != pilot.controls_sha256(load_controls())


def test_a_hash_over_the_wrong_control_set_is_rejected():
    from creativity_bench.calibration import all_default_controls, load_controls

    v = validation(controls_sha256=pilot.controls_sha256(load_controls()))
    error = check(v, sha=pilot.controls_sha256(all_default_controls()))
    assert error == "Validation controls do not match this pilot's control set"


# --- against a real validate_gates() result, not a fixture --------------------


def real_suite(responder) -> dict:
    """What `live_pilot.py validate` saves, built by the production code path."""
    from conftest import FakeClient

    from creativity_bench.calibration import validate_gates

    result = validate_gates(FakeClient(responder, model="judge-a"))
    result["protocol_fingerprint"] = "fp"
    return result


def test_a_real_unresolved_suite_blocks_on_every_gate_including_the_new_ones():
    suite = real_suite(lambda _messages: "not json at all")
    error = check(suite, sha=pilot.expected_controls_sha(suite))
    assert error.startswith("Judge did not pass all development controls")
    for gate in gate_names():
        assert f"{gate}." in error
    assert "quilting.integrated" in error
    assert "unresolved judgments are not agreement" in error


def test_a_real_suite_clears_the_identity_and_coverage_checks():
    """The saved shape lines up, so only control results can block a real suite."""
    suite = real_suite(lambda _messages: "not json at all")
    assert pilot.validated_gates(suite) == gate_names()
    assert suite["controls_sha256"] == pilot.expected_controls_sha(suite)
    error = check(suite, sha=pilot.expected_controls_sha(suite))
    assert "does not cover" not in error
    assert "do not match" not in error
