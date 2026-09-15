from creativity_bench.cli import build_parser


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
