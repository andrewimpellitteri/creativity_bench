"""Offline tests for the judge-swap sensitivity tool (fake judge client only)."""

import importlib.util
import json
from pathlib import Path

import pytest
from conftest import FakeClient

from creativity_bench.tasks.same_but_different import JUDGE_SYSTEM

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rescore_judge_swap.py"
SPEC = importlib.util.spec_from_file_location("rescore_judge_swap", SCRIPT)
swap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(swap)


def verdict(**updates):
    result = dict(
        premise_adherent=True,
        comprehensible=True,
        plot_distinct=True,
        evidence="The conflict and resolution satisfy the rubric.",
        summary="A courier sacrifices her cargo to save a stranded rival.",
    )
    result.update(updates)
    return json.dumps(result)


def attempt(number, story, accepted, *, verdict_json=None, reasons=None, responses=None):
    return {
        "attempt": number,
        "story": story,
        "judge_responses": responses or [],
        "verdict": json.loads(verdict_json) if verdict_json else None,
        "accepted": accepted,
        "rejection_reasons": reasons or [],
    }


def write_run(path, transcript, *, premise="premise one", fingerprint="fp-0001"):
    premises = [
        {
            "premise": premise,
            "transcript": transcript,
            "accepted_stories": [
                {"story": entry["story"], "summary": entry["verdict"]["summary"]}
                for entry in transcript
                if entry["accepted"]
            ],
        }
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": "writer-model",
                "provider": "deepseek",
                "seed": 0,
                "metadata": {"protocol_fingerprint": fingerprint, "judge_model": "original-judge"},
                "tasks": {
                    "same_but_different": {
                        "score": 0.5,
                        "details": {"premises": premises},
                    }
                },
            }
        )
    )
    return path


def test_full_agreement_reuses_production_prompt_and_context(tmp_path):
    run = write_run(
        tmp_path / "run_a.json",
        [
            attempt(1, "story one", True, verdict_json=verdict()),
            attempt(2, "story two", True, verdict_json=verdict()),
        ],
    )
    judge = FakeClient(lambda _: verdict(), model="swap-judge")
    report = swap.rescore_runs([run], judge)
    run_report = report["runs"][0]
    assert run_report["attempted"] == 2
    assert all(
        cell["rate"] == 1.0 and cell["comparable"] == 2 for cell in run_report["agreement"].values()
    )
    assert run_report["flips"] == {dim: [] for dim in swap.DIMENSIONS}
    assert run_report["unresolved_original"] == 0
    assert run_report["unresolved_swapped"] == 0
    assert report["alternative_judge"]["model"] == "swap-judge"
    assert len(judge.calls) == 2
    for call in judge.calls:
        assert call[0]["content"] == JUDGE_SYSTEM
    first_prompt = json.loads(judge.calls[0][-1]["content"])
    second_prompt = json.loads(judge.calls[1][-1]["content"])
    assert first_prompt["accepted_stories"] == []
    assert second_prompt["accepted_stories"] == ["story one"]
    assert second_prompt["candidate"] == "story two"


def test_plot_distinct_flip_records_attempt_id_and_evidence(tmp_path):
    run = write_run(
        tmp_path / "run_b.json",
        [
            attempt(1, "story one", True, verdict_json=verdict()),
            attempt(2, "story two", True, verdict_json=verdict()),
        ],
    )
    responses = iter(
        [verdict(), verdict(plot_distinct=False, evidence="same causal plot, renamed surface")]
    )
    judge = FakeClient(lambda _: next(responses))
    report = swap.rescore_runs([run], judge)
    run_report = report["runs"][0]
    flips = run_report["flips"]["plot_distinct"]
    assert len(flips) == 1
    assert flips[0]["attempt_id"] == "run_b:0:2"
    assert flips[0]["original"] is True
    assert flips[0]["swapped"] is False
    assert flips[0]["swapped_evidence"] == "same causal plot, renamed surface"
    assert run_report["agreement"]["plot_distinct"]["rate"] == 0.5
    assert run_report["agreement"]["premise_adherent"]["rate"] == 1.0
    assert report["totals"]["flips"]["plot_distinct"] == 1


def test_unresolved_swapped_and_missing_original_verdict(tmp_path):
    run = write_run(
        tmp_path / "run_c.json",
        [
            attempt(1, "story one", True, verdict_json=verdict()),
            attempt(
                2,
                "story two",
                False,
                reasons=["unresolved_judgment"],
                responses=["original garbage"],
            ),
        ],
    )
    judge = FakeClient(lambda _: "not JSON at all")
    report = swap.rescore_runs([run], judge)
    run_report = report["runs"][0]
    assert run_report["attempted"] == 2
    assert run_report["unresolved_swapped"] == 2
    assert run_report["unresolved_original"] == 1
    assert all(
        cell["comparable"] == 0 and cell["rate"] is None
        for cell in run_report["agreement"].values()
    )
    assert len(judge.calls) == 6
    second = run_report["attempts"][1]
    assert second["original_verdict"] is None
    assert second["swapped_verdict"] is None
    assert second["swapped_status"] == "unresolved"
    assert len(second["swapped_judge_responses"]) == 3
    assert second["original_judge_responses"] == ["original garbage"]


def test_malformed_and_old_runs_skipped_with_clear_message(tmp_path, capsys):
    good = write_run(
        tmp_path / "run_d.json", [attempt(1, "story one", True, verdict_json=verdict())]
    )
    missing_sbd = tmp_path / "old_schema.json"
    missing_sbd.write_text(json.dumps({"schema_version": 1, "tasks": {}}))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json at all")
    judge = FakeClient(lambda _: verdict())
    report = swap.rescore_runs([missing_sbd, broken, good], judge)
    assert len(report["runs"]) == 1
    assert report["runs"][0]["run_id"] == "run_d"
    reasons = {item["path"]: item["reason"] for item in report["skipped_runs"]}
    assert reasons[str(missing_sbd)] == "no saved same_but_different task result"
    assert reasons[str(broken)].startswith("unreadable or invalid JSON")
    stderr = capsys.readouterr().err
    assert f"Skipping {missing_sbd}" in stderr
    assert f"Skipping {broken}" in stderr


def test_limit_bounds_attempts_per_run(tmp_path):
    transcript_a = [
        attempt(1, "story a1", True, verdict_json=verdict()),
        attempt(2, "story a2", False, verdict_json=verdict(plot_distinct=False)),
    ]
    transcript_b = [attempt(1, "story b1", True, verdict_json=verdict())]
    run = tmp_path / "run_e.json"
    run.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": "writer-model",
                "provider": "deepseek",
                "seed": 3,
                "metadata": {"protocol_fingerprint": "fp-0002"},
                "tasks": {
                    "same_but_different": {
                        "score": 0.5,
                        "details": {
                            "premises": [
                                {
                                    "premise": "p a",
                                    "transcript": transcript_a,
                                    "accepted_stories": [],
                                },
                                {
                                    "premise": "p b",
                                    "transcript": transcript_b,
                                    "accepted_stories": [],
                                },
                            ]
                        },
                    }
                },
            }
        )
    )
    judge = FakeClient(lambda _: verdict())
    limited = swap.rescore_runs([run], judge, limit=2)
    assert limited["runs"][0]["attempted"] == 2
    assert limited["runs"][0]["skipped_attempts"] == {"limit": 1}
    assert [a["attempt_id"] for a in limited["runs"][0]["attempts"]] == ["run_e:0:1", "run_e:0:2"]
    assert len(judge.calls) == 2
    unlimited = swap.rescore_runs([run], FakeClient(lambda _: verdict()))
    assert unlimited["runs"][0]["attempted"] == 3


def test_attempts_never_sent_to_judge_are_skipped(tmp_path):
    run = write_run(
        tmp_path / "run_f.json",
        [
            attempt(1, "story one", True, verdict_json=verdict()),
            attempt(2, "story one", False, reasons=["exact_duplicate"]),
            attempt(3, None, False, reasons=["generation_error"]),
        ],
    )
    judge = FakeClient(lambda _: verdict())
    report = swap.rescore_runs([run], judge)
    run_report = report["runs"][0]
    assert run_report["attempted"] == 1
    assert run_report["skipped_attempts"] == {"not_sent_to_judge": 1, "no_story": 1}
    assert len(judge.calls) == 1
    assert run_report["attempts"][0]["accepted_stories_context"] == []


def test_report_shape_and_context_warning(tmp_path):
    run = write_run(
        tmp_path / "run_g.json",
        [
            attempt(1, "story one", True, verdict_json=verdict()),
            attempt(2, "story two", False, verdict_json=verdict(plot_distinct=False)),
        ],
    )
    judge = FakeClient(lambda _: verdict())
    report = swap.rescore_runs([run], judge)
    assert report["tool"] == "rescore_judge_swap"
    assert report["schema"] == "judge-swap-v1"
    assert report["notice"] == ("Judge-swap measures judge sensitivity, not story quality.")
    assert set(report) >= {
        "generated",
        "alternative_judge",
        "source_protocol_fingerprints",
        "runs",
        "skipped_runs",
        "totals",
    }
    assert report["source_protocol_fingerprints"] == ["fp-0001"]
    run_report = report["runs"][0]
    assert run_report["source_judge_model"] == "original-judge"
    assert run_report["model"] == "writer-model"
    attempt_record = run_report["attempts"][0]
    assert set(attempt_record) >= {
        "attempt_id",
        "premise_index",
        "story",
        "accepted_stories_context",
        "original_verdict",
        "original_judge_responses",
        "swapped_verdict",
        "swapped_judge_responses",
        "comparable",
        "agreement",
    }
    assert attempt_record["original_verdict"]["plot_distinct"] is True
    assert attempt_record["swapped_verdict"]["plot_distinct"] is True
    totals = report["totals"]
    assert totals["attempts_rejudged"] == 2
    assert totals["agreement"]["plot_distinct"]["rate"] == 0.5
    assert totals["unresolved_original"] == 0
    assert run_report["context_warnings"] == []


def test_context_mismatch_is_flagged(tmp_path):
    run = write_run(
        tmp_path / "run_h.json",
        [attempt(1, "story one", True, verdict_json=verdict())],
    )
    payload = json.loads(run.read_text())
    payload["tasks"]["same_but_different"]["details"]["premises"][0]["accepted_stories"] = [
        {"story": "tampered", "summary": "s"}
    ]
    run.write_text(json.dumps(payload))
    report = swap.rescore_runs([run], FakeClient(lambda _: verdict()))
    assert report["runs"][0]["context_warnings"] == [0]


def test_cli_writes_report_without_network(tmp_path, monkeypatch):
    run = write_run(
        tmp_path / "run_i.json",
        [attempt(1, "story one", True, verdict_json=verdict())],
    )
    fake = FakeClient(lambda _: verdict(), model="alt-judge")
    monkeypatch.setattr(swap, "build_judge", lambda provider, model: fake)
    out = tmp_path / "reports" / "swap.json"
    rc = swap.main(
        [
            "--run",
            str(run),
            "--judge-model",
            "alt-judge",
            "--judge-provider",
            "deepseek",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    saved = json.loads(out.read_text())
    assert saved["alternative_judge"]["model"] == "alt-judge"
    assert saved["runs"][0]["attempted"] == 1


def test_cli_rejects_nonpositive_limit(tmp_path):
    run = tmp_path / "run_j.json"
    run.write_text("{}")
    with pytest.raises(SystemExit):
        swap.main(
            [
                "--run",
                str(run),
                "--judge-model",
                "m",
                "--judge-provider",
                "deepseek",
                "--limit",
                "0",
            ]
        )


def test_cli_refuses_to_overwrite_existing_report(tmp_path, monkeypatch):
    run = write_run(
        tmp_path / "run_o.json",
        [attempt(1, "story one", True, verdict_json=verdict())],
    )
    fake = FakeClient(lambda _: verdict(), model="alt-judge")
    monkeypatch.setattr(swap, "build_judge", lambda provider, model: fake)
    out = tmp_path / "swap.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        swap.main(
            [
                "--run",
                str(run),
                "--judge-model",
                "alt-judge",
                "--judge-provider",
                "deepseek",
                "--out",
                str(out),
            ]
        )
    assert out.read_text() == "{}"
