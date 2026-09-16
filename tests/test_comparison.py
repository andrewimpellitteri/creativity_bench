import json

import pytest

from creativity_bench.comparison import (
    PROVENANCE_FIELDS,
    cohort_key,
    group_cohorts,
    paired_difference,
    verified_provenance,
)
from creativity_bench.report import build_leaderboard, collect_rows
from creativity_bench.visualize import load_runs, plot_comparison


def run(model="a", seed=0, score=0.5):
    return {
        "schema_version": 2,
        "model": model,
        "provider": "fixture",
        "seed": seed,
        "composite": score,
        "scores": {"diversity": score},
        "weights": {"diversity": 1},
        "metadata": {
            **dict.fromkeys(PROVENANCE_FIELDS, "fixture"),
            "generation_provider": "fixture",
            "generation_base_url": None,
            "judge_base_url": None,
            "embed_base_url": None,
            "selected_tasks": ["diversity"],
            "fast": False,
            "task_sizes": {"n": 2},
            "generation_settings": {"policy": "test"},
            "judge_settings": {"policy": "test"},
            "evaluation_complete": True,
        },
    }


@pytest.mark.parametrize("field", PROVENANCE_FIELDS)
def test_incompatible_provenance_never_pooled(field):
    a, b = run(), run()
    b["metadata"][field] = ["other"] if field == "selected_tasks" else "other"
    assert len(group_cohorts({"a": [a, b]})) == 2
    with pytest.raises(ValueError, match="cohort"):
        collect_rows({"a": [a, b]})


@pytest.mark.parametrize(
    "field,value", [("weights", {"diversity": 0.5}), ("scores", {"telephone": 0.5})]
)
def test_scores_and_weights_separate(field, value):
    a, b = run(), run()
    b[field] = value
    assert len(group_cohorts({"a": [a, b]})) == 2


def test_seed_pairing_averages_duplicates_and_excludes_unmatched():
    left = [
        run(seed=0, score=0.2),
        run(seed=0, score=0.8),
        run(seed=1, score=0.7),
        run(seed=2, score=0),
    ]
    right = [run("b", seed=0, score=0.3), run("b", seed=1, score=0.5), run("b", seed=3, score=1)]
    result = paired_difference(left, right, "diversity")
    assert result["n_matched_seeds"] == 2
    assert result["difference"] == pytest.approx(0.2)
    assert result["ci95"] == pytest.approx([0.2, 0.2])


def test_missing_data_and_single_pair_have_no_ci():
    assert paired_difference([run(seed=None)], [run()], "diversity")["difference"] is None
    result = paired_difference([run()], [run("b")], "diversity")
    assert result["n_matched_seeds"] == 1
    assert result["ci95"] is None
    assert paired_difference([run()], [run("b")], "missing")["n_matched_seeds"] == 0
    assert (
        paired_difference([run(score=float("nan"))], [run("b")], "diversity")["difference"] is None
    )


def test_unverified_models_separate():
    a, b = run(), run("b")
    del a["metadata"]["protocol_version"]
    del b["metadata"]["protocol_version"]
    assert len(group_cohorts({"a": [a], "b": [b]})) == 2


def test_writer_vendor_does_not_split_compatible_cohorts():
    """One pinned protocol and judge spans API vendors; providers stay in metadata."""
    a, b = run(), run("b")
    b["provider"] = "openrouter"
    b["metadata"]["generation_provider"] = "openrouter"
    b["metadata"]["generation_base_url"] = "https://openrouter.ai/api/v1"
    assert verified_provenance(b)
    assert cohort_key(a) == cohort_key(b)
    assert len(group_cohorts({"a": [a], "b": [b]})) == 1


def test_mixed_report_has_separate_tables_and_chart_refuses(tmp_path, capsys):
    a, b = run(), run("b")
    b["metadata"]["fast"] = True
    for i, payload in enumerate([a, b]):
        (tmp_path / f"{i}.json").write_text(json.dumps(payload))
    report = build_leaderboard(tmp_path)
    assert "## Cohort 1" in report and "## Cohort 2" in report
    assert "| Rank" not in report
    assert plot_comparison(tmp_path, tmp_path / "chart.png") == 1
    assert "Cannot chart mixed" in capsys.readouterr().out
    assert not (tmp_path / "chart.png").exists()


def test_incomplete_runs_excluded_from_report_and_pairing(tmp_path):
    payload = run()
    payload["metadata"]["evaluation_complete"] = False
    (tmp_path / "incomplete.json").write_text(json.dumps(payload))
    report = build_leaderboard(tmp_path)
    assert "Excluded 1 incomplete evaluations" in report
    assert "| `a` |" not in report
    with pytest.raises(ValueError, match="Incomplete"):
        paired_difference([payload], [payload], "diversity")


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_fingerprint", None),
        ("judge_model", ""),
        ("judge_provider", None),
        ("generation_settings", None),
        ("judge_settings", {}),
        ("task_sizes", None),
        ("selected_tasks", ["telephone"]),
        ("selected_tasks", ["diversity", "diversity"]),
        ("fast", "false"),
        ("generation_provider", "different"),
        ("embed_model", None),
        ("evaluation_complete", None),
    ],
)
def test_invalid_provenance_cannot_verify(field, value):
    payload = run()
    payload["metadata"][field] = value
    assert not verified_provenance(payload)
    with pytest.raises(ValueError, match="verified provenance"):
        paired_difference([payload], [payload], "diversity")


def test_missing_completion_is_unverified():
    payload = run()
    del payload["metadata"]["evaluation_complete"]
    assert not verified_provenance(payload)


def test_nonembedding_tasks_allow_explicit_null_embedding_and_default_endpoints():
    payload = run()
    payload["scores"] = {"same_but_different": 0.5}
    payload["weights"] = {"same_but_different": 1}
    metadata = payload["metadata"]
    metadata["selected_tasks"] = ["same_but_different"]
    metadata.update(
        embed_model=None,
        embed_provider=None,
        embed_base_url=None,
        judge_base_url=None,
        generation_base_url=None,
    )
    assert verified_provenance(payload)
    metadata["generation_provider"] = payload["provider"] = "custom"
    assert not verified_provenance(payload)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        {"schema_version": 2},
        {**run(), "composite": float("nan")},
        {**run(), "scores": {"diversity": float("inf")}},
        {**run(), "scores": {"diversity": None}},
        {**run(), "scores": {"diversity": 1.1}},
        {**run(), "metadata": []},
        {**run(), "metadata": {"judge_model": {}}},
        {**run(), "seed": []},
        {**run(), "model": []},
    ],
)
def test_load_skips_malformed_payloads(tmp_path, payload):
    (tmp_path / "invalid.json").write_text(json.dumps(payload))
    (tmp_path / "valid.json").write_text(json.dumps(run()))
    assert load_runs(tmp_path) == {"a": [run()]}
