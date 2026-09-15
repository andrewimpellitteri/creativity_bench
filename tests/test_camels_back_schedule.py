"""Offline tests for Camel's Back precomputed compatible edit schedules.

Design audit: "Random edit bundles can contradict one another... Use shared
starting stories and precomputed, compatible edit schedules. Specify whether
constraints persist or are replaced."
"""

from __future__ import annotations

import itertools
import random

from conftest import FakeClient

from creativity_bench.tasks.camels_back import (
    CONFLICTING_EDITS,
    _edits_conflict,
    build_edit_schedule,
    camels_back,
)

PASS_VERDICT = '{"coherent": true, "edits_applied": true, "quality_maintained": true}'
FAIL_VERDICT = '{"coherent": false, "edits_applied": false, "quality_maintained": false}'


def test_edit_schedule_is_deterministic_for_a_seed():
    requests = ["make it rhyme", "add more cowbell", "make it more concise"]
    first = build_edit_schedule(requests, 6, random.Random(7))
    second = build_edit_schedule(requests, 6, random.Random(7))
    assert first == second
    assert len(first) == 6


def test_edit_schedule_bundles_are_mutually_compatible():
    # The bundled edit pool contains contradictory pairs; no schedule round
    # may contain one.
    requests = [
        "make it rhyme",
        "add more cowbell",
        "rewrite it as a noir detective mystery",
        "translate it into Japanese",
        "make it more humorous",
        "add more suspense",
        "make it more poetic",
        "add a plot twist",
        "change the tone to be more serious",
        "add more descriptive details",
        "make it more concise",
        "add dialogue",
        "add more emotional depth",
        "add more action",
    ]
    for seed in range(20):
        schedule = build_edit_schedule(requests, 8, random.Random(seed))
        for bundle in schedule:
            assert 1 <= len(bundle) <= 3
            for a, b in itertools.combinations(bundle, 2):
                assert not _edits_conflict(a, b)


def test_conflict_table_matches_expected_pairs():
    assert _edits_conflict("Make it more concise", "add dialogue")
    assert _edits_conflict("add dialogue", "make it more concise")
    assert _edits_conflict("translate it into Japanese", "make it rhyme")
    assert not _edits_conflict("make it rhyme", "add more cowbell")
    # Every declared conflict is detectable in both orders.
    for a, b in CONFLICTING_EDITS:
        assert _edits_conflict(a.capitalize(), b)
        assert _edits_conflict(b.capitalize(), a)


def test_camels_back_follows_the_precomputed_schedule():
    prompts: list[str] = []
    counter = {"n": 0}

    def responder(messages):
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        counter["n"] += 1
        return f"a genuinely new story, version {counter['n']}"

    client = FakeClient(responder)
    judge = FakeClient(lambda _: PASS_VERDICT)
    result = camels_back(
        client,
        judge,
        seed_text="premise",
        edit_requests=["make it rhyme", "add more cowbell", "make it more concise"],
        max_edits=3,
        rng=random.Random(11),
    )
    assert result.score == 1.0
    schedule = result.details["schedule"]
    assert len(schedule) == 3
    # Round instructions shown to the model match the precomputed schedule.
    instruction_blocks = [p.split("INSTRUCTIONS:\n")[1] for p in prompts[1:]]
    assert instruction_blocks == ["\n".join(f"- {e}" for e in bundle) for bundle in schedule]
    assert result.metrics["constraints"] == "replaced_each_round"
    assert "REPLACE all earlier instructions" in prompts[1]


def test_camels_back_failed_round_kept_for_audit():
    counter = {"n": 0}

    def responder(_):
        counter["n"] += 1
        return f"modified story {counter['n']}"

    client = FakeClient(responder)
    # Opening story consumes n=1; round 0 (n=2) passes, round 1 (n=3) fails.
    judge = FakeClient(lambda _: PASS_VERDICT if counter["n"] < 3 else FAIL_VERDICT)
    result = camels_back(
        client,
        judge,
        seed_text="premise",
        edit_requests=["a", "b"],
        max_edits=5,
        rng=random.Random(0),
    )
    assert result.metrics["rounds_survived"] == 1
    assert result.metrics["failed_round_index"] == 1
    failed = [r for r in result.details["rounds"] if r["failed"]]
    assert len(failed) == 1
    assert failed[0]["round_index"] == 1
    assert len(result.details["rounds"]) == 2  # failed round is retained


def test_camels_back_same_seed_same_schedule_across_models():
    def run_once() -> list[list[str]]:
        counter = {"n": 0}

        def responder(_):
            counter["n"] += 1
            return f"story version {counter['n']}"

        judge = FakeClient(lambda _: PASS_VERDICT)
        result = camels_back(
            FakeClient(responder),
            judge,
            seed_text="premise",
            edit_requests=["make it rhyme", "add more cowbell", "make it more concise"],
            max_edits=4,
            rng=random.Random(3),
        )
        return result.details["schedule"]

    assert run_once() == run_once()
