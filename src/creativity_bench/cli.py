"""Command-line interface: `creativity-bench run` and `creativity-bench viz`."""

from __future__ import annotations

import argparse
import sys

from .client import (
    PROVIDERS,
    Embedder,
    LLMClient,
    resolve_provider,
    warn_if_paid_openrouter_model,
)
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
    run.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (raise this for heavily rate-limited endpoints)",
    )
    run.add_argument("--verbose", action="store_true", help="Print full transcripts while running")
    run.add_argument(
        "--no-save", action="store_true", help="Do not write results to the runs/ directory"
    )
    run.add_argument("--runs-dir", default="runs", help="Directory for result JSON files")

    viz = sub.add_parser("viz", help="Plot a comparison chart from saved runs")
    viz.add_argument("--runs-dir", default="runs", help="Directory containing run JSON files")
    viz.add_argument("--out", default="model_comparison.png", help="Output image path")
    viz.add_argument("--show", action="store_true", help="Open an interactive window as well")

    report = sub.add_parser("report", help="Write a markdown leaderboard from saved runs")
    report.add_argument("--runs-dir", default="runs", help="Directory containing run JSON files")
    report.add_argument("--out", default="results/leaderboard.md", help="Output markdown path")
    report.add_argument(
        "--chart",
        default=None,
        help="Also render the comparison chart to this image path",
    )

    gallery = sub.add_parser("gallery", help="Inspect stories and judgments from a saved run")
    gallery.add_argument("--run", required=True, help="Saved run JSON")
    gallery.add_argument("--out", default="results/gallery.html")

    validate = sub.add_parser("validate-judge", help="Evaluate a judge on labeled controls")
    validate.add_argument("--judge-model", required=True)
    validate.add_argument("--judge-provider", default="openai", choices=list(PROVIDERS))
    validate.add_argument("--base-url", default=None)
    validate.add_argument(
        "--controls", default=None, help="JSON controls; default: development set"
    )
    validate.add_argument("--out", default="results/judge_validation.json")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import print_results, run_benchmark, save_run

    provider = resolve_provider(args.provider, args.base_url)
    warn_if_paid_openrouter_model(provider, args.model)
    client = LLMClient(provider=provider, model=args.model, timeout=args.timeout)

    if args.judge_model:
        judge_provider_name = args.judge_provider or args.provider
        judge_provider = resolve_provider(
            judge_provider_name, args.base_url if judge_provider_name == args.provider else None
        )
        warn_if_paid_openrouter_model(judge_provider, args.judge_model)
        judge_client = LLMClient(
            provider=judge_provider, model=args.judge_model, timeout=args.timeout
        )
    else:
        judge_client = client

    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    if args.n < 1:
        raise ValueError("--n must be positive")
    needs_embeddings = {"telephone", "diversity", "style_transfer", "odd_one_out"}
    embedder = (
        Embedder(provider=resolve_provider(args.embed_provider), model=args.embed_model)
        if needs_embeddings.intersection(TASKS if tasks is None else tasks)
        else None
    )

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


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_leaderboard

    path = write_leaderboard(args.runs_dir, args.out)
    print(f"Wrote {path}")
    if args.chart:
        from .visualize import plot_comparison

        return plot_comparison(runs_dir=args.runs_dir, out_path=args.chart, show=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "gallery":
            from .gallery import write_gallery

            print(f"Wrote {write_gallery(args.run, args.out)}")
            return 0
        if args.command == "validate-judge":
            import json
            from pathlib import Path

            from .calibration import load_controls, validate_judge
            from .runner import protocol_fingerprint

            controls = load_controls(args.controls)
            judge = LLMClient(
                provider=resolve_provider(args.judge_provider, args.base_url),
                model=args.judge_model,
            )
            result = validate_judge(judge, controls)
            result["protocol_fingerprint"] = protocol_fingerprint()
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
            print(f"Wrote {out}")
            return 0
        if args.command == "run":
            return cmd_run(args)
        if args.command == "report":
            return cmd_report(args)
        return cmd_viz(args)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
