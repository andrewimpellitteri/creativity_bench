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
