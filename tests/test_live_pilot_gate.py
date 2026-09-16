import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_pilot.py"
SPEC = importlib.util.spec_from_file_location("live_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def validation(**overrides):
    base = {
        "judge_model": "judge-a",
        "protocol_fingerprint": "fp",
        "controls_sha256": "sha",
        "summaries": {
            "development": {
                "premise_adherent": {"resolution_rate": 1, "accuracy_resolved": 1},
                "comprehensible": {"resolution_rate": 1, "accuracy_resolved": 1},
                "plot_distinct": {"resolution_rate": 1, "accuracy_resolved": 1},
            }
        },
    }
    base.update(overrides)
    return base


def test_passing_validation_gates_nothing():
    assert pilot.validation_gate_error(
        validation(), judge="judge-a", fingerprint="fp", controls_sha="sha"
    ) is None


@pytest.mark.parametrize(
    "kwargs,judge,fp,sha",
    [
        ({"judge_model": "judge-b"}, "judge-a", "fp", "sha"),
        ({"protocol_fingerprint": "stale"}, "judge-a", "fp", "sha"),
        ({"controls_sha256": "different-controls"}, "judge-a", "fp", "sha"),
    ],
)
def test_identity_mismatches_are_rejected(kwargs, judge, fp, sha):
    error = pilot.validation_gate_error(
        validation(**kwargs), judge=judge, fingerprint=fp, controls_sha=sha
    )
    assert error


def test_imperfect_control_rejects_even_with_matching_identity():
    v = validation()
    v["summaries"]["development"]["plot_distinct"]["accuracy_resolved"] = 0.875
    assert pilot.validation_gate_error(v, judge="judge-a", fingerprint="fp", controls_sha="sha")
