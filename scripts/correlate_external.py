"""Correlate this repo's cohort scores with EQ-Bench creative-writing boards.

Joins our 23-model fast-suite scores with the published EQ-Bench Creative
Writing v3 (Elo) and Longform (0-100) tables over manually verified name
matches. Rank correlations only; approximate-variant matches are flagged.

The EQ-Bench assets (CSV embedded in JS) are fetched beforehand to
/tmp/opencode/cw.js and /tmp/opencode/lf.js; this script is offline.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

OUT = Path("results/extended-20260915")

# our model id -> (cw_v3 exact name | None, longform exact name | None, approx?)
MAPPING = {
    "deepseek-v4-pro": ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro", False),
    "deepseek-flash": ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash", False),
    "glm-5.3-flash": ("GLM-5.3", "GLM-5.3", True),
    "glm-4.6": ("zai-org/GLM-4.6", "zai-org/GLM-4.6", False),
    "moonshotai/kimi-k3": ("kimi-k3", "kimi-k3", False),
    "openai/gpt-4o-mini": ("gpt-4o-mini", None, False),
    "openai/gpt-5-mini": ("gpt-5-mini-2025-08-07", "gpt-5-mini-2025-08-07", True),
    "google/gemini-2.5-flash": (
        "gemini-2.5-flash-preview",
        "google/gemini-2.5-flash-preview-05-20",
        True,
    ),
    "google/gemini-3-flash-preview": (None, "gemini-3-flash-preview", False),
    "anthropic/claude-haiku-4.5": (None, "claude-haiku-4.5", False),
    "meta-llama/llama-4-scout": (
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        False,
    ),
    "mistralai/mistral-small-3.2-24b-instruct": (
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        False,
    ),
    "nvidia/nemotron-3-ultra-550b-a55b": (
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        None,
        False,
    ),
    "cohere/command-a": ("CohereForAI/c4ai-command-a-03-2025", None, False),
    "x-ai/grok-4.20": ("grok-4.20-beta", "grok-4.20-beta", True),
    "microsoft/phi-4": (None, "microsoft/phi-4-multimodal-instruct", True),
}


def parse_eqbench(path: str) -> dict[str, dict]:
    text = Path(path).read_text(errors="ignore")
    block = re.search(r"let leaderboardData[A-Za-z0-9]* = `(.*?)`", text, re.S)
    header = None
    rows: dict[str, dict] = {}
    for line in block.group(1).splitlines():
        line = line.strip().lstrip("*")
        if not line or "," not in line:
            continue
        cells = line.split(",")
        if header is None:
            if cells[0] == "model_name":
                header = cells
            continue
        if len(cells) != len(header):
            continue
        rows[cells[0]] = dict(zip(header, cells, strict=True))
    return rows


def our_scores() -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for p in glob.glob(str(OUT / "suite_runs" / "*.json")):
        r = json.loads(Path(p).read_text())
        slot = agg.setdefault(r["model"], {"composite": [], "tasks": {}})
        slot["composite"].append(r["composite"])
        for t, tr in r["tasks"].items():
            slot["tasks"].setdefault(t, []).append(tr["score"])
    for slot in agg.values():
        slot["composite"] = float(np.mean(slot["composite"]))
        slot["tasks"] = {t: float(np.mean(v)) for t, v in slot["tasks"].items()}
    return agg


def ranks(x: list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    order = np.argsort(arr, kind="stable")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    for v in np.unique(arr):
        mask = arr == v
        if mask.sum() > 1:
            r[mask] = r[mask].mean()
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall(a: list[float], b: list[float]) -> float:
    n = len(a)
    num = sum(
        np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        for i in range(n)
        for j in range(i + 1, n)
    )
    return float(num / (n * (n - 1) / 2))


def scatter(board: str, score_col: str, join: list, our: list[float], sp: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#fcfcfb")
    ext = [j[3] for j in join]
    ax.scatter(ext, our, s=44, color="#2a78d6", zorder=3)
    for (ours_name, _, approx, score), c in zip(join, our, strict=True):
        label = ours_name.split("/")[-1] + (" *" if approx else "")
        ax.annotate(label, (score, c), fontsize=7, xytext=(4, 3), textcoords="offset points")
    z = np.polyfit(ext, our, 1)
    xs = np.linspace(min(ext), max(ext), 50)
    ax.plot(xs, np.polyval(z, xs), color="#898781", linewidth=1, linestyle="--", zorder=2)
    ax.set_facecolor("#fcfcfb")
    ax.set_xlabel(board, fontsize=10)
    ax.set_ylabel("our composite (fast, n=2)", fontsize=10)
    title = f"Our cohort vs {board}: Spearman {sp:.2f} (n={len(join)})"
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        str(OUT / f"CORRELATION_{score_col}.png"),
        dpi=160,
        facecolor="#fcfcfb",
        bbox_inches="tight",
    )


def main() -> None:
    ours = our_scores()
    boards = {
        "EQ-Bench Creative Writing v3 (Elo)": (
            parse_eqbench("/tmp/opencode/cw.js"),
            "elo_score",
        ),
        "EQ-Bench Longform Writing (0-100)": (
            parse_eqbench("/tmp/opencode/lf.js"),
            "overall_score_100",
        ),
    }
    lines = [
        "# External correlation — this cohort vs EQ-Bench creative-writing boards",
        "",
        "Rank correlations over manually verified name matches. Our side: fast-budget,",
        "five-task, n=2 development scores with one pinned judge; their side: LLM-judged",
        "boards with far more per-model effort. Approximate-variant matches (flagged *)",
        "are included; n is small, so treat these as orientation, not validation.",
        "",
    ]
    for board, (rows, score_col) in boards.items():
        join = []
        for ours_name, (cw_name, lf_name, approx) in MAPPING.items():
            ext_name = cw_name if "Creative Writing v3" in board else lf_name
            if ours_name in ours and ext_name and ext_name in rows:
                join.append((ours_name, ext_name, approx, float(rows[ext_name][score_col])))
        if len(join) < 6:
            lines += [f"## {board}: only {len(join)} matches — skipped", ""]
            continue
        our = [ours[j[0]]["composite"] for j in join]
        ext = [j[3] for j in join]
        sp, kt = spearman(our, ext), kendall(our, ext)
        lines += [f"## {board} (n={len(join)})", "", "n.b. * = approximate variant match", ""]
        lines += ["| our model | EQ-Bench name | our composite | their score |",
                  "|---|---|---|---|"]
        ordered = sorted(zip(join, our, strict=True), key=lambda t: -t[0][3])
        for (ours_name, ext_name, approx, score), c in ordered:
            flag = "*" if approx else ""
            lines.append(f"| {ours_name}{flag} | {ext_name} | {c:.2f} | {score:.1f} |")
        lines += ["", f"**Spearman rho = {sp:.2f}, Kendall tau = {kt:.2f}**"]
        lines += ["", "Per-task Spearman vs their score:"]
        for task in ("same_but_different", "shaggy_dog", "subversion", "camels_back"):
            tv = [ours[j[0]]["tasks"].get(task) for j in join]
            if all(v is not None and np.isfinite(v) for v in tv):
                lines.append(f"- {task}: {spearman(tv, ext):+.2f}")
        lines += ["", f"Chart: `CORRELATION_{score_col}.png`", ""]
        try:
            scatter(board, score_col, join, our, sp)
        except Exception as exc:
            lines.append(f"(chart skipped: {exc})")
    (OUT / "CORRELATION.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'CORRELATION.md'}")


if __name__ == "__main__":
    main()
