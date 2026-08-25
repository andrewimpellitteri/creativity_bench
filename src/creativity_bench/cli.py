"""Command-line interface: `creativity-bench run` and `creativity-bench viz`."""

from __future__ import annotations

import argparse
import sys

from .client import PROVIDERS, Embedder, LLMClient, resolve_provider
from .tasks import TASKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="creativity-bench",
        description="Benchmark the creative capabilities of an LLM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the benchmark against a model")
    run.add_argument(
        "--model", required=True, help="Model to benchmark, e.g. gpt-5-mini or glm-4.6"
    )
    run.add_argument(
        "--provider",
        default="openai",
        choices=list(PROVIDERS),
        help="API provider preset (default: openai). Use 'zai-coding' for GLM Coding Plan keys.",
    )
    run.add_argument("--base-url", default=None, help="Override the provider base URL")
    run.add_argument(
        "--judge-model",
        default=None,
        help="Model used to judge edit quality (default: same as --model). "
        "Keep this fixed when comparing models.",
    )
    run.add_argument(
        "--judge-provider",
        default=None,
        choices=list(PROVIDERS),
        help="Provider for the judge model (default: same as --provider)",
    )
    run.add_argument(
        "--embed-provider",
        default="openai",
        choices=list(PROVIDERS),
        help="Provider for embeddings (default: openai)",
    )
    run.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="Embedding model (default: text-embedding-3-small)",
    )
    run.add_argument(
        "--tasks",
        default=None,
        help=f"Comma-separated subset of tasks to run. Available: {', '.join(TASKS)}",
    )
    run.add_argument("--n", type=int, default=1, help="Number of benchmark repetitions")
    run.add_argument(
        "--seed", type=int, default=None, help="Random seed (per-run seeds derive from it)"
    )
    run.add_argument("--fast", action="store_true", help="Smaller task sizes for a cheap smoke run")
    run.add_argument("--verbose", action="store_true", help="Print full transcripts while running")
    run.add_argument(
        "--no-save", action="store_true", help="Do not write results to the runs/ directory"
    )
    run.add_argument("--runs-dir", default="runs", help="Directory for result JSON files")

    viz = sub.add_parser("viz", help="Plot a comparison chart from saved runs")
    viz.add_argument("--runs-dir", default="runs", help="Directory containing run JSON files")
    viz.add_argument("--out", default="model_comparison.png", help="Output image path")
    viz.add_argument("--show", action="store_true", help="Open an interactive window as well")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import print_results, run_benchmark, save_run

    provider = resolve_provider(args.provider, args.base_url)
    client = LLMClient(provider=provider, model=args.model)

    if args.judge_model:
        judge_provider = resolve_provider(args.judge_provider or args.provider)
        judge_client = LLMClient(provider=judge_provider, model=args.judge_model)
    else:
        judge_client = client

    embedder = Embedder(provider=resolve_provider(args.embed_provider), model=args.embed_model)
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    for i in range(args.n):
        if args.n > 1:
            print(f"\n########## Run {i + 1}/{args.n} ##########")
        seed = None if args.seed is None else args.seed + i
        result = run_benchmark(
            client,
            judge_client,
            embedder,
            tasks=tasks,
            seed=seed,
            fast=args.fast,
            verbose=args.verbose,
        )
        print_results(result)
        if not args.no_save:
            path = save_run(result, args.runs_dir)
            print(f"Results saved to {path}")
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    from .visualize import plot_comparison

    return plot_comparison(runs_dir=args.runs_dir, out_path=args.out, show=args.show)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        return cmd_viz(args)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
