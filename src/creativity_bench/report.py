"""Aggregate saved benchmark runs into a markdown leaderboard."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

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
    rows.sort(key=lambda row: -row["composite"])
    return rows


def build_leaderboard(runs_dir: str | Path, *, generated: str | None = None) -> str:
    runs = load_runs(runs_dir)
    if not runs:
        raise ValueError(f"No usable run files in {runs_dir}/. Run `creativity-bench run` first.")
    rows = collect_rows(runs)
    generated = generated or dt.date.today().isoformat()
    n_runs = sum(len(v) for v in runs.values())

    task_headers = [TASK_LABELS[t].replace("\n", " ") for t in TASK_ORDER]
    best_tasks = {
        t: max(
            (row["tasks"][t] for row in rows if not np.isnan(row["tasks"][t])),
            default=float("nan"),
        )
        for t in TASK_ORDER
    }

    header_cells = [*task_headers, "n", "Seeds", "Judge", "Runs from"]
    lines = [
        "# Creativity Bench — Leaderboard",
        "",
        f"Generated {generated} from {n_runs} runs of {len(rows)} models in `{runs_dir}/`.",
        "Scores are in [0, 1]; the composite is the weighted mean over the eight tasks.",
        "",
        "| Rank | Model | Composite | " + " | ".join(header_cells) + " |",
        "|---:|---|---:|" + "---:|" * len(TASK_ORDER) + ":--:|:--|:--|:--|",
    ]

    for i, row in enumerate(rows, start=1):
        composite = f"{row['composite']:.3f} ± {row['std']:.3f}"
        if i == 1:
            composite = f"**{composite}**"
        task_cells = [
            _fmt(row["tasks"][t], None if np.isnan(best_tasks[t]) else best_tasks[t])
            for t in TASK_ORDER
        ]
        model_cell = f"`{row['model']}`" + (" ⚡" if row["fast"] else "")
        lines.append(
            f"| {i} | {model_cell} | {composite} | "
            + " | ".join(task_cells)
            + f" | {row['n']} | {row['seeds']} | {row['judge']} | {row['dates']} |"
        )

    lines += ["", "## Notes", ""]
    if any(row["fast"] for row in rows):
        lines.append(
            "- ⚡ marks a model run with `--fast` (about 3x smaller task sizes), used when an "
            "endpoint cannot sustain full-size runs; its scores are not directly comparable "
            "to full-size runs."
        )
    if any(row["model"] in row["judge"] for row in rows):
        lines.append(
            "- At least one model graded its own outputs (see the Judge column); treat its "
            "judge-dependent task scores with extra caution."
        )
    lines += [
        "- Judge-dependent tasks (`camels_back`, `odd_one_out`, `subversion`, `shaggy_dog`) "
        "inherit the judge model's biases; the judge is held fixed across models to keep "
        "scores comparable.",
        "- Repeat runs (n > 1) vary only the RNG seed; error spreads are population std-dev "
        "across repeats.",
        "- Reproduce with `creativity-bench run` (see the README), then "
        "`creativity-bench report --runs-dir runs`.",
    ]
    return "\n".join(lines) + "\n"


def write_leaderboard(runs_dir: str | Path, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_leaderboard(runs_dir))
    return out_path
