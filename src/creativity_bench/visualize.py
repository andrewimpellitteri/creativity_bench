"""Comparison chart for saved benchmark runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Validated categorical palette (light surface), fixed slot order — never cycled.
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

TASK_ORDER = ["free_association", "telephone", "camels_back", "diversity", "style_transfer"]
TASK_LABELS = {
    "free_association": "Free\nassociation",
    "telephone": "Telephone\ngame",
    "camels_back": "Camel's\nback",
    "diversity": "Diversity",
    "style_transfer": "Style\ntransfer",
}


def load_runs(runs_dir: str | Path) -> dict[str, list[dict]]:
    """Group run payloads by model, skipping files with unknown schemas."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(Path(runs_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"Skipping {path.name}: not valid JSON")
            continue
        if payload.get("schema_version") != 2:
            print(f"Skipping {path.name}: old or unknown result format")
            continue
        runs[payload["model"]].append(payload)
    return dict(runs)


def plot_comparison(
    runs_dir: str | Path = "runs",
    out_path: str | Path = "model_comparison.png",
    show: bool = False,
) -> int:
    import matplotlib.pyplot as plt

    runs = load_runs(runs_dir)
    if not runs:
        print(f"No usable run files in {runs_dir}/. Run `creativity-bench run` first.")
        return 1
    if len(runs) > len(SERIES_COLORS):
        print(f"Plotting the first {len(SERIES_COLORS)} models; fold the rest into another chart.")
        runs = dict(list(runs.items())[: len(SERIES_COLORS)])

    # Sort models by mean composite, best first; color follows the model.
    models = sorted(runs, key=lambda m: -np.mean([r["composite"] for r in runs[m]]))
    colors = {model: SERIES_COLORS[i] for i, model in enumerate(models)}

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(11, 8), height_ratios=[1, 1.4], facecolor=SURFACE
    )

    # Top: composite score per model, with std-dev error bars across repeat runs.
    composites = [np.mean([r["composite"] for r in runs[m]]) for m in models]
    errors = [np.std([r["composite"] for r in runs[m]]) for m in models]
    x = np.arange(len(models))
    ax_top.bar(
        x, composites, width=0.55,
        color=[colors[m] for m in models],
        yerr=errors, capsize=4,
        error_kw={"elinewidth": 1, "ecolor": INK_MUTED},
        zorder=3,
    )
    for xi, value in zip(x, composites):
        ax_top.text(xi, value + 0.02, f"{value:.2f}", ha="center", va="bottom",
                    fontsize=10, color=INK_PRIMARY)
    ax_top.set_xticks(x, models, fontsize=10)
    ax_top.set_title("Composite creativity score", loc="left", fontsize=12, color=INK_PRIMARY)

    # Bottom: per-task mean scores, grouped by task, one series per model.
    tasks = [t for t in TASK_ORDER if any(t in r["scores"] for m in models for r in runs[m])]
    group_x = np.arange(len(tasks))
    bar_width = min(0.8 / max(len(models), 1), 0.25)
    for i, model in enumerate(models):
        means = [
            np.mean([r["scores"][t] for r in runs[model] if t in r["scores"]] or [np.nan])
            for t in tasks
        ]
        offset = (i - (len(models) - 1) / 2) * bar_width
        ax_bottom.bar(group_x + offset, means, width=bar_width * 0.92,
                      color=colors[model], label=model, zorder=3)
    ax_bottom.set_xticks(group_x, [TASK_LABELS.get(t, t) for t in tasks], fontsize=10)
    ax_bottom.set_title("Per-task scores", loc="left", fontsize=12, color=INK_PRIMARY)
    ax_bottom.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK_PRIMARY)

    for ax in (ax_top, ax_bottom):
        ax.set_facecolor(SURFACE)
        ax.set_ylim(0, 1.05)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED, length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out_path} ({len(models)} models, {sum(len(v) for v in runs.values())} runs)")
    if show:
        plt.show()
    return 0
