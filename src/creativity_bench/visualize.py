"""Comparison chart for saved benchmark runs.

A PNG output also gets a same-basename SVG copy written next to it (a vector
version for publications); both paths are printed. Other extensions keep the
single-file behavior.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .comparison import group_cohorts, verified_provenance

# Validated categorical palette (light surface), fixed slot order — never cycled.
# Tail slots are Okabe-Ito additions so a single cohort can chart 12+ writers.
SERIES_COLORS = [
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#e87ba4",
    "#eb6834",
    "#e69f00",
    "#56b4e9",
    "#f0e442",
    "#7f7f7f",
    "#8c564b",
    "#17becf",
]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

TASK_ORDER = [
    "same_but_different",
    "free_association",
    "telephone",
    "camels_back",
    "diversity",
    "style_transfer",
    "this_and_that",
    "this_and_that_not",
    "copycat",
    "quilting",
    "odd_one_out",
    "subversion",
    "shaggy_dog",
]
TASK_LABELS = {
    "same_but_different": "Same but\ndifferent",
    "free_association": "Free\nassociation",
    "telephone": "Telephone\ngame",
    "camels_back": "Camel's\nback",
    "diversity": "Diversity",
    "style_transfer": "Style\ntransfer",
    "this_and_that": "This &\nthat",
    "this_and_that_not": "This & that\n(not that)",
    "copycat": "Copycat",
    "quilting": "Quilting",
    "odd_one_out": "Odd one\nout",
    "subversion": "Subversion",
    "shaggy_dog": "Shaggy\ndog",
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
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            print(f"Skipping {path.name}: old or unknown result format")
            continue

        def valid_score(value):
            return (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and np.isfinite(value)
                and 0 <= value <= 1
            )

        scores = payload.get("scores")
        metadata = payload.get("metadata")
        seed = payload.get("seed")
        metadata_malformed = isinstance(metadata, dict) and (
            not isinstance(metadata.get("timestamp", ""), str)
            or any(
                metadata.get(field) is not None and not isinstance(metadata[field], str)
                for field in (
                    "judge_model",
                    "judge_provider",
                    "embed_model",
                    "embed_provider",
                    "generation_provider",
                    "protocol_version",
                    "protocol_fingerprint",
                )
            )
            or any(
                field in metadata and type(metadata[field]) is not bool
                for field in ("fast", "evaluation_complete")
            )
        )
        if (
            not isinstance(payload.get("model"), str)
            or not payload["model"].strip()
            or not isinstance(payload.get("provider"), str)
            or not valid_score(payload.get("composite"))
            or not isinstance(scores, dict)
            or not scores
            or any(not isinstance(k, str) or not valid_score(v) for k, v in scores.items())
            or (metadata is not None and not isinstance(metadata, dict))
            or (seed is not None and type(seed) is not int)
            or metadata_malformed
        ):
            print(f"Skipping {path.name}: malformed run payload or non-finite/out-of-range scores")
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
    if any(
        (r.get("metadata") or {}).get("evaluation_complete") is False
        for rs in runs.values()
        for r in rs
    ):
        print(
            "Cannot chart incomplete evaluations. Generate a report to inspect exclusions "
            "and use a directory of complete evaluations."
        )
        return 1
    cohorts = group_cohorts(runs)
    if len(cohorts) != 1 or not all(verified_provenance(r) for rs in runs.values() for r in rs):
        print(
            "Cannot chart mixed or unverified protocol cohorts. Generate a report to inspect "
            "cohorts, then copy one verified cohort into a separate runs directory."
        )
        return 1
    if len(runs) > len(SERIES_COLORS):
        print(f"Plotting the first {len(SERIES_COLORS)} models; fold the rest into another chart.")
        runs = dict(list(runs.items())[: len(SERIES_COLORS)])

    # Sort models by mean composite, best first; color follows the model.
    models = sorted(runs)
    colors = {model: SERIES_COLORS[i] for i, model in enumerate(models)}

    fig, (ax_bottom, ax_top) = plt.subplots(
        2, 1, figsize=(16, 8.5), height_ratios=[1.4, 1], facecolor=SURFACE
    )

    # Top: composite score per model, with std-dev error bars across repeat runs.
    composites = [np.mean([r["composite"] for r in runs[m]]) for m in models]
    errors = [np.std([r["composite"] for r in runs[m]]) for m in models]
    x = np.arange(len(models))
    ax_top.bar(
        x,
        composites,
        width=0.55,
        color=[colors[m] for m in models],
        yerr=errors,
        capsize=4,
        error_kw={"elinewidth": 1, "ecolor": INK_MUTED},
        zorder=3,
    )
    # Individual runs as dots: n per model is visible, not hidden by the mean.
    for xi, model in zip(x, models, strict=True):
        ax_top.scatter(
            [xi] * len(runs[model]),
            [r["composite"] for r in runs[model]],
            s=14,
            color=INK_PRIMARY,
            zorder=4,
        )
    for xi, value in zip(x, composites, strict=True):
        ax_top.text(
            xi,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK_PRIMARY,
        )
    ax_top.set_xticks(x, models, fontsize=9)
    ax_top.tick_params(axis="x", labelrotation=30)
    for label in ax_top.get_xticklabels():
        label.set_horizontalalignment("right")
    ax_top.set_ylabel("score", fontsize=10, color=INK_MUTED)
    ax_top.set_title(
        "Exploratory composite (unvalidated weighting)", loc="left", fontsize=12, color=INK_PRIMARY
    )

    # Bottom: per-task mean scores, grouped by task, one series per model.
    tasks = [t for t in TASK_ORDER if any(t in r["scores"] for m in models for r in runs[m])]
    group_x = np.arange(len(tasks))
    bar_width = min(0.8 / max(len(models), 1), 0.25)
    label_rotation = 0 if len(models) <= 3 else 90
    for i, model in enumerate(models):
        means = [
            np.mean([r["scores"][t] for r in runs[model] if t in r["scores"]] or [np.nan])
            for t in tasks
        ]
        offset = (i - (len(models) - 1) / 2) * bar_width
        ax_bottom.bar(
            group_x + offset,
            means,
            width=bar_width * 0.92,
            color=colors[model],
            label=model,
            zorder=3,
        )
        for gx, value in zip(group_x + offset, means, strict=True):
            if np.isfinite(value):
                ax_bottom.text(
                    gx,
                    value + 0.02,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=label_rotation,
                    color=INK_PRIMARY,
                )
    ax_bottom.set_xticks(group_x, [TASK_LABELS.get(t, t) for t in tasks], fontsize=10)
    ax_bottom.set_ylabel("score", fontsize=10, color=INK_MUTED)
    ax_bottom.set_title("Per-task scores", loc="left", fontsize=12, color=INK_PRIMARY)
    ax_bottom.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        labelcolor=INK_PRIMARY,
        columnspacing=1.2,
    )

    # Provenance subtitle: one verified cohort must be self-describing.
    first = next(iter(runs.values()))[0]
    meta = first.get("metadata") or {}
    budget = "fast" if meta.get("fast") else "full"
    judge = meta.get("judge_model") or "model-as-judge"
    protocol = meta.get("protocol_version") or "unknown protocol"
    run_counts = {len(rs) for rs in runs.values()}
    n_note = f"{sorted(run_counts)[0]}" if len(run_counts) == 1 else "mixed"
    fig.text(
        0.01,
        0.995,
        f"{protocol} · {budget} budget · judge: {judge} · runs per model: {n_note}",
        fontsize=8.5,
        color=INK_MUTED,
        va="top",
    )

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
    out = Path(out_path)
    if out.suffix.lower() == ".png":
        svg_path = out.with_suffix(".svg")
        fig.savefig(svg_path, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {svg_path} (vector copy)")
    if show:
        plt.show()
    return 0


def acceptance_curves(runs: dict[str, list[dict]]) -> dict[str, list[list[int]]]:
    """Per-model cumulative acceptance curves from saved Same But Different runs.

    One curve per premise per run: ``curve[i]`` is how many distinct plots had
    been accepted after ``i + 1`` scheduled attempts. Runs without the task, or
    with ragged curves, are skipped rather than padded — a padded curve would
    invent acceptances that never happened.
    """
    curves: dict[str, list[list[int]]] = {}
    for model, payloads in runs.items():
        for payload in payloads:
            task = (payload.get("tasks") or {}).get("same_but_different")
            if not isinstance(task, dict):
                continue
            attempts = (task.get("details") or {}).get("attempts_per_premise")
            for premise in (task.get("details") or {}).get("premises") or []:
                curve = premise.get("acceptance_curve")
                if not isinstance(curve, list) or not curve or len(curve) != attempts:
                    continue
                if any(type(value) is not int for value in curve):
                    continue
                curves.setdefault(model, []).append(curve)
    return curves


def plot_acceptance_curves(
    runs_dir: str | Path = "runs",
    out_path: str | Path = "acceptance_curves.png",
    show: bool = False,
) -> int:
    """Chart cumulative accepted plots against scheduled attempts.

    This is the figure that shows whether a budget was large enough: a curve
    that keeps climbing at the last attempt means the benchmark measured its own
    budget, not the model's supply of distinct plots. One faint line per premise
    keeps the spread visible; the bold line is the model's mean.
    """
    import matplotlib.pyplot as plt

    runs = load_runs(runs_dir)
    if not runs:
        print(f"No usable run files in {runs_dir}/. Run `creativity-bench run` first.")
        return 1
    if len(group_cohorts(runs)) != 1 or not all(
        verified_provenance(r) for rs in runs.values() for r in rs
    ):
        print(
            "Cannot chart mixed or unverified protocol cohorts. Generate a report to inspect "
            "cohorts, then copy one verified cohort into a separate runs directory."
        )
        return 1
    curves = acceptance_curves(runs)
    if not curves:
        print(f"No Same But Different acceptance curves in {runs_dir}/.")
        return 1

    models = sorted(curves)
    colors = {model: SERIES_COLORS[i % len(SERIES_COLORS)] for i, model in enumerate(models)}
    budget = max(len(curve) for series in curves.values() for curve in series)
    attempts = np.arange(1, budget + 1)

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=SURFACE)

    # Reference: every scheduled attempt accepted. The gap to it is the failure.
    ax.plot(
        attempts,
        attempts,
        color=INK_MUTED,
        linewidth=1.2,
        linestyle=(0, (5, 4)),
        alpha=0.7,
        zorder=2,
    )
    ax.annotate(
        "every attempt accepted",
        xy=(budget, budget),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK_MUTED,
    )

    for model in models:
        series = [curve for curve in curves[model] if len(curve) == budget]
        if not series:
            continue
        for curve in series:
            ax.plot(attempts, curve, color=colors[model], linewidth=1.1, alpha=0.45, zorder=3)
        mean = np.mean(series, axis=0)
        # 2px mean line over a surface ring, so crossing means stay legible.
        ax.plot(attempts, mean, color=SURFACE, linewidth=4.5, zorder=4)
        ax.plot(attempts, mean, color=colors[model], linewidth=2, zorder=5)
        ax.scatter([attempts[-1]], [mean[-1]], s=40, color=colors[model], zorder=6)
        # Direct label in ink, not the series color: the mark carries identity.
        ax.annotate(
            f"{model}  {mean[-1]:.1f}/{budget}",
            xy=(budget, mean[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=INK_PRIMARY,
        )

    ax.set_title(
        "Same But Different: cumulative distinct plots accepted",
        fontsize=13,
        color=INK_PRIMARY,
        loc="left",
        pad=30,
    )
    first = next(iter(runs.values()))[0]
    meta = first.get("metadata") or {}
    premise_count = sum(len(series) for series in curves.values())
    # Provenance sits between title and plot so the chart is self-describing.
    ax.text(
        0,
        1.02,
        f"{meta.get('protocol_version') or 'unknown protocol'} · "
        f"{'fast' if meta.get('fast') else 'full'} budget · "
        f"judge: {meta.get('judge_model') or 'model-as-judge'} · "
        f"{premise_count} premise curves · faint lines are single premises",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_MUTED,
        va="bottom",
    )
    ax.set_xlabel("scheduled attempt", fontsize=10, color=INK_MUTED)
    ax.set_ylabel("accepted distinct plots", fontsize=10, color=INK_MUTED)
    ax.set_xlim(1, budget + 0.02 * budget)
    ax.set_ylim(0, budget + 0.5)
    ax.set_xticks(attempts)
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, length=0)
    handles = [
        plt.Line2D([], [], color=colors[model], linewidth=2, label=model) for model in models
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        fontsize=9.5,
        labelcolor=INK_PRIMARY,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out_path} ({len(models)} models, {premise_count} curves)")
    out = Path(out_path)
    if out.suffix.lower() == ".png":
        svg_path = out.with_suffix(".svg")
        fig.savefig(svg_path, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {svg_path} (vector copy)")
    if show:
        plt.show()
    return 0
