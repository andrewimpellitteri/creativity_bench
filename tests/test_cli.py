import json

import pytest

from creativity_bench.cli import build_parser, parse_weights


def test_run_parses_custom_timeout():
    args = build_parser().parse_args(["run", "--model", "test-model", "--timeout", "240"])
    assert args.timeout == 240.0


def test_run_timeout_defaults_to_120():
    args = build_parser().parse_args(["run", "--model", "test-model"])
    assert args.timeout == 120.0


def test_custom_judge_inherits_endpoint_without_embedding_dependency(monkeypatch):
    from conftest import FakeClient

    from creativity_bench import cli

    providers = []

    def make_client(*, provider, model, **kwargs):
        providers.append(provider)
        client = FakeClient(lambda _: "word", model=model)
        client.provider = provider
        return client

    def no_embedder(**kwargs):
        raise AssertionError("This task should not initialize an embedding service")

    monkeypatch.setattr(cli, "LLMClient", make_client)
    monkeypatch.setattr(cli, "Embedder", no_embedder)
    assert (
        cli.main(
            [
                "run",
                "--provider",
                "custom",
                "--base-url",
                "https://example.test/v1",
                "--model",
                "writer",
                "--judge-model",
                "judge",
                "--tasks",
                "free_association",
                "--fast",
                "--no-save",
            ]
        )
        == 0
    )
    assert [p.base_url for p in providers] == ["https://example.test/v1"] * 2


# --- --weights ---------------------------------------------------------------


def test_parse_weights_accepts_task_value_pairs():
    assert parse_weights("diversity=2,telephone=0.5") == {"diversity": 2.0, "telephone": 0.5}


def test_parse_weights_tolerates_spacing_and_zero():
    assert parse_weights(" diversity = 1.5 , telephone=0 ") == {
        "diversity": 1.5,
        "telephone": 0.0,
    }


def test_parse_weights_rejects_unknown_task_and_lists_the_valid_ones():
    with pytest.raises(ValueError) as excinfo:
        parse_weights("diversity=1,oddoneout=2")
    message = str(excinfo.value)
    assert "oddoneout" in message
    assert "odd_one_out" in message  # the valid names are listed
    assert "same_but_different" in message


@pytest.mark.parametrize(
    "spec",
    [
        "diversity=-1",  # negative
        "diversity=nan",  # not finite
        "diversity=inf",
        "diversity=high",  # not a number
        "diversity",  # no value
        "diversity=",
        "=1",
        "",  # nothing at all
        "diversity=1,diversity=2",  # duplicate
    ],
)
def test_parse_weights_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        parse_weights(spec)


def test_run_weights_flag_defaults_to_none():
    args = build_parser().parse_args(["run", "--model", "test-model"])
    assert args.weights is None


def _fake_run(monkeypatch, argv, tmp_path, tasks="free_association"):
    """Run `creativity-bench run` end to end against fake clients."""
    from conftest import FakeClient

    from creativity_bench import cli

    def make_client(*, provider, model, **kwargs):
        client = FakeClient(lambda _: "word", model=model)
        client.provider = provider
        return client

    monkeypatch.setattr(cli, "LLMClient", make_client)
    monkeypatch.setattr(cli, "Embedder", lambda **kwargs: None)
    code = cli.main(
        [
            "run",
            "--model",
            "writer",
            "--tasks",
            tasks,
            "--fast",
            "--seed",
            "0",
            "--runs-dir",
            str(tmp_path),
            *argv,
        ]
    )
    saved = sorted(tmp_path.glob("*.json"))
    return code, [json.loads(path.read_text()) for path in saved]


def test_custom_weights_reach_the_saved_run_and_its_cohort_key(monkeypatch, tmp_path):
    from creativity_bench.comparison import cohort_key

    code, runs = _fake_run(monkeypatch, ["--weights", "free_association=3"], tmp_path)
    assert code == 0
    assert runs[0]["weights"] == {"free_association": 3.0}
    # An unweighted run of the same tasks must not pool with it.
    default_code, default_runs = _fake_run(monkeypatch, [], tmp_path / "default")
    assert default_code == 0
    assert default_runs[0]["weights"]["free_association"] == 0.2
    assert cohort_key(runs[0]) != cohort_key(default_runs[0])


def test_unknown_weight_task_fails_the_run_cleanly(monkeypatch, tmp_path, capsys):
    code, runs = _fake_run(monkeypatch, ["--weights", "not_a_task=1"], tmp_path)
    assert code == 1
    assert runs == []  # nothing saved
    assert "not_a_task" in capsys.readouterr().err


def test_weights_that_zero_out_every_selected_task_are_refused(monkeypatch, tmp_path, capsys):
    # A 0.0 composite must never be a configuration artifact.
    code, runs = _fake_run(monkeypatch, ["--weights", "diversity=1"], tmp_path)
    assert code == 1
    assert runs == []
    assert "zero total weight" in capsys.readouterr().err


def test_custom_weights_change_the_composite(monkeypatch, tmp_path):
    tasks = "free_association,shaggy_dog"
    _, weighted = _fake_run(
        monkeypatch, ["--weights", "free_association=3,shaggy_dog=1"], tmp_path, tasks=tasks
    )
    _, equal = _fake_run(monkeypatch, [], tmp_path / "equal", tasks=tasks)
    scores = weighted[0]["scores"]
    assert scores == equal[0]["scores"]  # same seed, same fake client
    assert weighted[0]["composite"] == pytest.approx(
        (3 * scores["free_association"] + scores["shaggy_dog"]) / 4
    )
    assert equal[0]["composite"] == pytest.approx(
        (scores["free_association"] + scores["shaggy_dog"]) / 2
    )
    assert weighted[0]["composite"] != pytest.approx(equal[0]["composite"])


def test_embedding_tasks_get_an_embedder(monkeypatch):
    # runner.run_benchmark rejects this_and_that and quilting without an
    # embedder, so the CLI must build one for them.
    from conftest import FakeClient, FakeEmbedder

    from creativity_bench import cli, runner

    seen = {}

    def make_client(*, provider, model, **kwargs):
        client = FakeClient(lambda _: "word", model=model)
        client.provider = provider
        return client

    def fake_run_benchmark(client, judge_client, embedder, **kwargs):
        seen["embedder"] = embedder
        raise SystemExit(0)

    monkeypatch.setattr(cli, "LLMClient", make_client)
    monkeypatch.setattr(cli, "Embedder", lambda **kwargs: FakeEmbedder())
    monkeypatch.setattr(runner, "run_benchmark", fake_run_benchmark)
    for task in ("this_and_that", "quilting", "odd_one_out"):
        seen.clear()
        with pytest.raises(SystemExit):
            cli.main(["run", "--model", "writer", "--tasks", task, "--fast", "--no-save"])
        assert seen["embedder"] is not None, task


# --- validate-judge --gate ----------------------------------------------------


def _fake_judge(monkeypatch, responder):
    """Route the CLI's judge client through an offline fake. No network."""
    from conftest import FakeClient

    from creativity_bench import cli

    clients = []

    def make_client(*, provider, model, **kwargs):
        client = FakeClient(responder, model=model)
        client.provider = provider
        clients.append(client)
        return client

    monkeypatch.setattr(cli, "LLMClient", make_client)
    return clients


def _superset_verdict(_messages):
    """One response that satisfies every gate's schema at once.

    Agreement is beside the point here: the CLI's job is to run the right gates
    and save the right shape, not to be right about the labels.
    """
    return json.dumps(
        {
            "premise_adherent": True,
            "comprehensible": True,
            "plot_distinct": True,
            "draws_on_a": True,
            "draws_on_b": True,
            "integrated": True,
            "opening": 1,
            "evidence": "fixture evidence",
            "summary": "fixture summary",
        }
    )


def test_validate_judge_gate_defaults_to_same_but_different(monkeypatch, tmp_path):
    from creativity_bench import cli

    args = build_parser().parse_args(["validate-judge", "--judge-model", "judge"])
    assert args.gate == "same_but_different"

    _fake_judge(monkeypatch, _superset_verdict)
    out = tmp_path / "validation.json"
    assert cli.main(["validate-judge", "--judge-model", "judge", "--out", str(out)]) == 0
    saved = json.loads(out.read_text())
    # Unchanged shape for the pre-existing invocation.
    assert saved["kind"] == "judge_control_validation"
    assert saved["gate"] == "same_but_different"
    assert set(saved["summaries"]["development"]) == {
        "premise_adherent",
        "comprehensible",
        "plot_distinct",
    }
    assert len(saved["records"]) == 9
    assert saved["protocol_fingerprint"]


def test_validate_judge_gate_selects_one_new_gate(monkeypatch, tmp_path):
    from creativity_bench import cli
    from creativity_bench.calibration import default_controls

    _fake_judge(monkeypatch, _superset_verdict)
    out = tmp_path / "validation.json"
    assert (
        cli.main(
            ["validate-judge", "--judge-model", "judge", "--gate", "quilting", "--out", str(out)]
        )
        == 0
    )
    saved = json.loads(out.read_text())
    assert saved["gate"] == "quilting"
    assert set(saved["summaries"]["development"]) == {"comprehensible", "integrated"}
    assert len(saved["records"]) == len(default_controls("quilting"))


def test_validate_judge_gate_all_reports_every_gate_separately(monkeypatch, tmp_path):
    from creativity_bench import cli
    from creativity_bench.calibration import all_default_controls, gate_names

    _fake_judge(monkeypatch, _superset_verdict)
    out = tmp_path / "validation.json"
    assert (
        cli.main(["validate-judge", "--judge-model", "judge", "--gate", "all", "--out", str(out)])
        == 0
    )
    saved = json.loads(out.read_text())
    assert saved["kind"] == "judge_control_validation_suite"
    assert list(saved["gates"]) == gate_names()
    for name, single in saved["gates"].items():
        assert single["gate"] == name
        assert single["summaries"]["development"]
        assert len(single["records"]) == len(all_default_controls()[name])
    assert saved["protocol_fingerprint"]
    # 9 + 7 + 5 + 4 development controls.
    assert sum(len(g["records"]) for g in saved["gates"].values()) == 25


def test_validate_judge_gate_all_accepts_a_mixed_control_file(monkeypatch, tmp_path):
    from creativity_bench import cli
    from creativity_bench.calibration import default_controls

    mixed = default_controls("same_but_different")[:1] + default_controls("quilting")[:1]
    controls = tmp_path / "controls.json"
    controls.write_text(json.dumps(mixed))
    _fake_judge(monkeypatch, _superset_verdict)
    out = tmp_path / "validation.json"
    assert (
        cli.main(
            [
                "validate-judge",
                "--judge-model",
                "judge",
                "--gate",
                "all",
                "--controls",
                str(controls),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    saved = json.loads(out.read_text())
    assert list(saved["gates"]) == ["same_but_different", "quilting"]
    assert all(len(g["records"]) == 1 for g in saved["gates"].values())


def test_validate_judge_reports_blockers_without_spending_a_pilot(monkeypatch, tmp_path, capsys):
    from creativity_bench import cli

    _fake_judge(monkeypatch, lambda _messages: "not json at all")
    out = tmp_path / "validation.json"
    assert (
        cli.main(
            ["validate-judge", "--judge-model", "judge", "--gate", "quilting", "--out", str(out)]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "blocker: quilting.integrated" in printed
    assert "resolved; unresolved judgments are not agreement" in printed


def test_validate_judge_rejects_a_bad_control_file_before_any_judge_call(monkeypatch, tmp_path):
    from creativity_bench import cli

    controls = tmp_path / "controls.json"
    controls.write_text(json.dumps([{"id": "x"}]))
    clients = _fake_judge(monkeypatch, _superset_verdict)
    assert (
        cli.main(
            [
                "validate-judge",
                "--judge-model",
                "judge",
                "--controls",
                str(controls),
                "--out",
                str(tmp_path / "v.json"),
            ]
        )
        == 1
    )
    assert clients == []
