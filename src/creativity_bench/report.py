"""Aggregate saved benchmark runs into a markdown leaderboard."""

from __future__ import annotations

import datetime as dt
from itertools import combinations
from pathlib import Path

import numpy as np

from .comparison import group_cohorts, paired_difference, verified_provenance
from .visualize import TASK_LABELS, TASK_ORDER, load_runs


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _fmt(value: float, best: float | None = None) -> str:
    if np.isnan(value):
        return "—"
    text = f"{value:.3f}"
    return f"**{text}**" if best is not None and value == best else text


def collect_rows(runs: dict[str, list[dict]]) -> list[dict]:
    """Summarise each model's runs: means, spread, provenance, and per-task scores."""
    if len(group_cohorts(runs)) > 1:
        raise ValueError("collect_rows requires one compatible protocol cohort")
    rows = []
    for model, model_runs in runs.items():
        composites = [r["composite"] for r in model_runs]
        judges = sorted(
            {(r.get("metadata") or {}).get("judge_model") or "self-judged" for r in model_runs}
        )
        seeds = sorted({r["seed"] for r in model_runs if r.get("seed") is not None})
        providers = sorted({r.get("provider", "?") for r in model_runs})
        dates = sorted((r.get("metadata") or {}).get("timestamp", "")[:10] for r in model_runs)
        fast = any((r.get("metadata") or {}).get("fast") for r in model_runs)
        rows.append(
            {
                "model": model,
                "fast": fast,
                "provider": ", ".join(providers),
                "composite": float(np.mean(composites)),
                "std": float(np.std(composites)),
                "n": len(model_runs),
                "seeds": ", ".join(str(s) for s in seeds) or "—",
                "judge": ", ".join(judges),
                "dates": dates[0] if dates[0] == dates[-1] else f"{dates[0]} → {dates[-1]}",
                "tasks": {
                    t: _mean_or_nan([r["scores"][t] for r in model_runs if t in r["scores"]])
                    for t in TASK_ORDER
                },
            }
        )
    rows.sort(key=lambda row: row["model"])
    return rows


def build_leaderboard(runs_dir: str | Path, *, generated: str | None = None) -> str:
    runs = load_runs(runs_dir)
    if not runs:
        raise ValueError(f"No usable run files in {runs_dir}/. Run `creativity-bench run` first.")
    incomplete = sum(
        (r.get("metadata") or {}).get("evaluation_complete") is False
        for rs in runs.values()
        for r in rs
    )
    runs = {
        model: [r for r in rs if (r.get("metadata") or {}).get("evaluation_complete") is not False]
        for model, rs in runs.items()
    }
    runs = {model: rs for model, rs in runs.items() if rs}
    generated = generated or dt.date.today().isoformat()
    n_runs = sum(len(v) for v in runs.values())
    lines = [
        "# Creativity Bench — Leaderboard",
        "",
        f"Generated {generated} from {n_runs} runs of {len(runs)} models in `{runs_dir}/`.",
        f"Excluded {incomplete} incomplete evaluations: unresolved judgments or generation "
        "errors yield audit lower bounds, not comparable creativity scores.",
        "Task profiles are primary. The composite is exploratory: its weighting has not "
        "been validated as a measure of creativity. Models are listed alphabetically.",
        "Different protocol cohorts are not directly comparable; no cross-cohort ranking is made.",
        "",
    ]
    all_rows = []
    for index, cohort in enumerate(group_cohorts(runs).values(), 1):
        rows = collect_rows(cohort)
        all_rows.extend(rows)
        example = next(iter(cohort.values()))[0]
        verified = verified_provenance(example)
        metadata = example.get("metadata") or {}
        status = "verified provenance" if verified else "UNVERIFIED legacy/incomplete provenance"
        lines += [
            f"## Cohort {index} — {status}",
            "",
            f"Protocol: `{metadata.get('protocol_version', 'unknown')}`; "
            f"fast: `{metadata.get('fast', 'unknown')}`; "
            f"judge: `{metadata.get('judge_model', 'unknown')}`.",
            "",
        ]
        headers = [
            "Model",
            *[TASK_LABELS[t].replace("\n", " ") for t in TASK_ORDER],
            "Exploratory composite ± SD",
            "n runs",
            "Seeds",
            "Judge",
            "Runs from",
        ]
        lines += [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            cells = [
                f"`{row['model']}`" + (" ⚡" if row["fast"] else ""),
                *[_fmt(row["tasks"][t]) for t in TASK_ORDER],
                f"{row['composite']:.3f} ± {row['std']:.3f}",
                str(row["n"]),
                row["seeds"],
                row["judge"],
                row["dates"],
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if verified and len(cohort) >= 2:
            lines += [
                "### Paired task differences",
                "",
                "Differences are left minus right; percentile bootstrap 95% intervals "
                "resample matched seeds. Duplicate seed runs are averaged first. "
                "Intervals are descriptive, without multiple-comparison correction.",
                "",
                "| Left - right | Task | Matched seeds | Difference | 95% interval |",
                "|---|---|---:|---:|---|",
            ]
            for left, right in combinations(sorted(cohort), 2):
                for task in TASK_ORDER:
                    if task not in example.get("scores", {}):
                        continue
                    comparison = paired_difference(cohort[left], cohort[right], task)
                    difference = comparison["difference"]
                    ci = comparison["ci95"]
                    value = "—" if difference is None else f"{difference:.3f}"
                    interval = (
                        "— (need ≥2 matched seeds)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                    )
                    lines.append(
                        f"| `{left}` - `{right}` | {task} | "
                        f"{comparison['n_matched_seeds']} | {value} | {interval} |"
                    )
            lines.append("")
    lines += [
        "## Notes",
        "",
        "- ⚡ denotes fast task budgets. Cohorts split by protocol, tasks, weights, "
        "budgets, judge, embedding and generation settings.",
        "- Legacy or incomplete provenance cannot establish compatibility; these runs "
        "are shown separately by model and are excluded from paired inference.",
        "- Profile means weight saved runs equally; paired differences weight matched "
        "seeds equally after averaging duplicates, so their differences may differ.",
        "- SD describes variation across saved runs, not uncertainty from independent "
        "samples. Pairwise story distances are dependent and are never bootstrap units.",
        "- Judge-dependent scores inherit the judge model's biases.",
    ]
    if any(row["model"] in row["judge"] for row in all_rows):
        lines.append(
            "- At least one model graded its own outputs (see Judge); interpret "
            "judge-dependent scores with caution."
        )
    return "\n".join(lines) + "\n"


def write_leaderboard(runs_dir: str | Path, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_leaderboard(runs_dir))
    return out_path
