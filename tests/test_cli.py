from creativity_bench.cli import build_parser


def test_run_parses_custom_timeout():
    args = build_parser().parse_args(["run", "--model", "test-model", "--timeout", "240"])
    assert args.timeout == 240.0


def test_run_timeout_defaults_to_120():
    args = build_parser().parse_args(["run", "--model", "test-model"])
    assert args.timeout == 120.0
