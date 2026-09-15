import json

import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.runner import composite_score, print_results, run_benchmark, save_run
from creativity_bench.tasks.base import TaskResult
from creativity_bench.visualize import load_runs

PASS_VERDICT = '{"coherent": true, "edits_applied": true, "quality_maintained": true}'


def full_responder(messages):
    prompt = messages[-1]["content"]
    # Check the subversion judge first: stories embedded in its prompt may
    # themselves contain the word "coherent".
    if "plot_preserved" in prompt and "genre_achieved" in prompt:
        return (
            '{"plot_preserved": true, "genre_achieved": true, '
            '"comprehensible": true, "reason": "ok"}'
        )
    if "qualifies" in prompt:
        return '{"qualifies": true}'
    if "premise_adherent" in messages[0]["content"]:
        return (
            '{"premise_adherent": true, "comprehensible": true, "plot_distinct": true, '
            '"evidence": "ok", "summary": "plot"}'
        )
    if "opposite" in prompt and "JSON object with this boolean field" in prompt:
        return '{"opposite": true}'
    if "coherent" in prompt:  # judge prompt
        return PASS_VERDICT
    if "free-association" in (messages[0].get("content") or ""):
        return f"word{len(messages)}"
    if "Summarize" in prompt:
        return f"summary variant {hash(prompt) % 10_000}"
    return f"generated text for: {prompt[:40]} ({hash(prompt) % 10_000})"


def test_composite_score_weighted_mean():
    results = {
        "a": TaskResult(name="a", score=1.0),
        "b": TaskResult(name="b", score=0.0),
    }
    assert composite_score(results, {"a": 3.0, "b": 1.0}) == pytest.approx(0.75)


def test_composite_ignores_missing_tasks():
    results = {"a": TaskResult(name="a", score=0.5)}
    assert composite_score(results, {"a": 0.2, "b": 0.8}) == pytest.approx(0.5)


def test_run_benchmark_end_to_end(tmp_path, capsys):
    client = FakeClient(full_responder)
    result = run_benchmark(client, client, FakeEmbedder(), seed=42, fast=True)

    assert set(result.task_results) == {
        "same_but_different",
        "free_association",
        "telephone",
        "camels_back",
        "diversity",
        "style_transfer",
        "odd_one_out",
        "subversion",
        "shaggy_dog",
    }
    assert 0.0 <= result.composite <= 1.0
    for task_result in result.task_results.values():
        assert 0.0 <= task_result.score <= 1.0

    path = save_run(result, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert payload["model"] == "fake-model"
    assert payload["seed"] == 42

    print_results(result)
    assert "Composite:" in capsys.readouterr().out

    loaded = load_runs(tmp_path)
    assert list(loaded) == ["fake-model"]


def test_run_benchmark_task_subset():
    client = FakeClient(full_responder)
    result = run_benchmark(client, client, FakeEmbedder(), tasks=["diversity"], seed=1, fast=True)
    assert list(result.task_results) == ["diversity"]


def test_run_benchmark_rejects_unknown_task():
    client = FakeClient(full_responder)
    with pytest.raises(ValueError, match="Unknown tasks"):
        run_benchmark(client, client, FakeEmbedder(), tasks=["nope"], fast=True)


def test_run_benchmark_reproducible_with_seed():
    r1 = run_benchmark(
        FakeClient(full_responder),
        FakeClient(full_responder),
        FakeEmbedder(),
        tasks=["camels_back"],
        seed=7,
        fast=True,
    )
    r2 = run_benchmark(
        FakeClient(full_responder),
        FakeClient(full_responder),
        FakeEmbedder(),
        tasks=["camels_back"],
        seed=7,
        fast=True,
    )
    edits1 = [r["edits"] for r in r1.task_results["camels_back"].details["rounds"]]
    edits2 = [r["edits"] for r in r2.task_results["camels_back"].details["rounds"]]
    assert edits1 == edits2


def test_load_runs_skips_old_format(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"some-model": {"composite": 1.5}}))
    (tmp_path / "junk.json").write_text("not json {")
    assert load_runs(tmp_path) == {}


def test_task_inputs_do_not_depend_on_other_tasks():
    def run(tasks):
        return run_benchmark(
            FakeClient(full_responder),
            FakeClient(full_responder),
            FakeEmbedder(),
            tasks=tasks,
            seed=7,
            fast=True,
        )

    alone = run(["diversity"])
    together = run(["camels_back", "diversity"])
    assert alone.task_results["diversity"].details == together.task_results["diversity"].details
    assert alone.metadata["protocol_version"] == "0.4-validity"


def test_run_usage_is_snapshot_and_not_cumulative():
    client = FakeClient(full_responder)
    first = run_benchmark(client, client, None, tasks=["free_association"], seed=1, fast=True)
    second = run_benchmark(client, client, None, tasks=["free_association"], seed=2, fast=True)
    assert first.metadata["generation_usage"]["requests"] == 10
    assert second.metadata["generation_usage"]["requests"] == 10
    assert first.metadata["embed_model"] is None
    assert len(first.metadata["protocol_fingerprint"]) == 64


def test_unresolved_judge_marks_evaluation_incomplete():
    client = FakeClient(lambda _: "story")
    judge = FakeClient(lambda _: "invalid verdict")
    result = run_benchmark(client, judge, None, tasks=["same_but_different"], seed=0, fast=True)
    assert result.metadata["evaluation_complete"] is False
    assert result.task_results["same_but_different"].score == 0
