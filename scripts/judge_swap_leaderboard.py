"""Alternative Same But Different leaderboard: replay acceptance per judge.

Consumes two judge-swap reports (scripts/rescore_judge_swap.py output) plus
the original run files, replays each run's acceptance cascade three ways
(original verdicts, judge A's verdicts, judge B's verdicts), and compares the
per-model rankings. Unresolved verdicts count as rejections in every replay
(fail-closed lower bound); identical cascade rules make the three columns
directly comparable.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

DIMS = ("premise_adherent", "comprehensible", "plot_distinct")


def replay(records: list[dict], verdict_key: str) -> float | None:
    """Greedy production-style cascade; score = accepted / scheduled attempts."""
    accepted: list[str] = []
    premise_max: dict[int, int] = {}
    for rec in records:
        premise = rec["premise_index"]
        premise_max[premise] = max(premise_max.get(premise, 0), rec.get("attempt") or 0)
        verdict = rec.get(verdict_key)
        if not isinstance(verdict, dict):
            continue  # unresolved: attempt consumed, no acceptance
        story = rec.get("story") or ""
        if all(bool(verdict[d]) for d in DIMS) and not any(
            story.strip() == a.strip() for a in accepted
        ):
            accepted.append(story)
    scheduled = sum(premise_max.values())
    return len(accepted) / scheduled if scheduled else None


def per_model(report: dict, key: str) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for run in report["runs"]:
        score = replay(run["attempts"], key)
        if score is not None:
            per.setdefault(run["model"], []).append(score)
    return {m: float(np.mean(v)) for m, v in per.items()}


def production_scores(runs_dir: Path) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for p in glob.glob(str(runs_dir / "*.json")):
        r = json.loads(Path(p).read_text())
        per.setdefault(r["model"], []).append(r["tasks"]["same_but_different"]["score"])
    return {m: float(np.mean(v)) for m, v in per.items()}


def ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: -kv[1])
    out: dict[str, float] = {}
    for i, (model, _) in enumerate(ordered, 1):
        ties = [m for m, s in ordered if s == ordered[i - 1][1]]
        out[model] = sum(range(i, i + len(ties))) / len(ties)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("results/extended-20260915/suite_runs")
    )
    parser.add_argument(
        "--flash", type=Path, required=True, help="Swap report, deepseek-flash judge"
    )
    parser.add_argument(
        "--glm", type=Path, help="Optional second swap report (glm-5.3-flash judge)"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prod = production_scores(args.runs_dir)
    orig_a = per_model(json.loads(args.flash.read_text()), "original_verdict")
    flash = per_model(json.loads(args.flash.read_text()), "swapped_verdict")
    orig_b, glm = {}, {}
    if args.glm:
        orig_b = per_model(json.loads(args.glm.read_text()), "original_verdict")
        glm = per_model(json.loads(args.glm.read_text()), "swapped_verdict")

    models = sorted(set(prod) & set(orig_a) & set(flash) & set(orig_b) & set(glm))
    if not args.glm:
        models = sorted(set(prod) & set(orig_a) & set(flash))
    if len(models) < 5:
        import sys

        print("Too few models with complete swaps; refusing to rank", file=sys.stderr)
        return 1

    replayed_prod = {
        m: ((orig_a[m] + orig_b[m]) / 2 if m in orig_b else orig_a[m])
        for m in models
    }
    columns = {
        "replayed original (deepseek-v4-pro)": replayed_prod,
        "deepseek-flash as judge": flash,
    }
    if args.glm:
        columns["glm-5.3-flash as judge"] = glm
    lines = [
        "# Alternative Same But Different leaderboard (swapped judges)",
        "",
        "Acceptance cascades replayed identically from each judge's verdicts over the",
        "same saved stories; unresolved verdicts count as rejections (fail-closed).",
        "Production score shown for reference. n=2 seeds per model; fast budget.",
        "",
        "| model | production | replayed original | flash judge |"
        + (" glm judge |" if args.glm else ""),
        "|---|---|---|---|" + "---|" if args.glm else "|---|---|---|---|",
    ]
    for m in sorted(models, key=lambda m: -columns["replayed original (deepseek-v4-pro)"][m]):
        row = (
            f"| {m} | {prod.get(m, float('nan')):.2f} | {replayed_prod[m]:.2f} "
            f"| {flash[m]:.2f} "
        )
        row += f"| {glm[m]:.2f} |" if args.glm else "|"
        lines.append(row)
    lines += ["", "Rank correlations across judges (Spearman):"]
    rank_sets = {k: ranks(v) for k, v in columns.items()}
    keys = list(rank_sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            shared = sorted(set(rank_sets[a]) & set(rank_sets[b]))
            ra = [rank_sets[a][m] for m in shared]
            rb = [rank_sets[b][m] for m in shared]
            lines.append(f"- {a} vs {b}: {float(np.corrcoef(ra, rb)[0, 1]):.2f}")
    self_orig = replayed_prod.get("deepseek-v4-pro")
    self_flash = flash.get("deepseek-v4-pro")
    if self_orig is not None and self_flash is not None:
        lines += [
            "",
            "Self-judge check (deepseek-v4-pro): replayed original "
            f"{self_orig:.2f} vs under flash judge {self_flash:.2f}.",
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
