"""Protocol compatibility and uncertainty at the independent seed-unit level."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

PROVENANCE_FIELDS = (
    "protocol_version",
    "selected_tasks",
    "task_sizes",
    "fast",
    "judge_model",
    "judge_provider",
    "embed_model",
    "embed_provider",
    "generation_settings",
    "judge_settings",
    "protocol_fingerprint",
    # Writer provider identity is deliberately excluded from the cohort
    # signature: comparing models across API vendors under one pinned protocol
    # and judge is the benchmark's purpose. Providers stay recorded in run
    # metadata, in verified_provenance, and in chart/report labels.
)


def cohort_key(run: dict) -> str:
    metadata = run.get("metadata") or {}
    signature = {field: metadata.get(field) for field in PROVENANCE_FIELDS}
    if isinstance(signature["selected_tasks"], list) and all(
        isinstance(t, str) for t in signature["selected_tasks"]
    ):
        signature["selected_tasks"] = sorted(signature["selected_tasks"])
    signature["evaluation_complete"] = metadata.get("evaluation_complete")
    signature["scores"] = sorted(run.get("scores", {}))
    signature["weights"] = run.get("weights")
    # Unknown provenance cannot establish cross-model compatibility.
    if not verified_provenance(run):
        signature["unverified_model"] = run.get("model")
        signature["unverified_provider"] = run.get("provider")
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def verified_provenance(run: dict) -> bool:
    metadata = run.get("metadata") or {}
    if not isinstance(metadata, dict) or not all(f in metadata for f in PROVENANCE_FIELDS):
        return False
    if metadata.get("evaluation_complete") is not True or type(metadata.get("fast")) is not bool:
        return False
    required_names = (
        "protocol_version",
        "protocol_fingerprint",
        "judge_model",
        "judge_provider",
        "generation_provider",
    )
    if any(
        not isinstance(metadata.get(f), str) or not metadata.get(f, "").strip()
        for f in required_names
    ):
        return False
    if metadata.get("generation_provider") != run.get("provider"):
        return False
    if any(
        not isinstance(metadata[f], dict) or not metadata[f]
        for f in ("task_sizes", "generation_settings", "judge_settings")
    ):
        return False
    selected = metadata["selected_tasks"]
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(task, str) for task in selected)
        or len(set(selected)) != len(selected)
        or set(selected) != set(run.get("scores", {}))
    ):
        return False
    embedding_tasks = {"telephone", "diversity", "style_transfer", "odd_one_out"}
    if embedding_tasks.intersection(selected) and any(
        not isinstance(metadata[f], str) or not metadata[f].strip()
        for f in ("embed_model", "embed_provider")
    ):
        return False
    for role in ("generation", "judge", "embed"):
        endpoint = metadata.get(f"{role}_base_url")
        if endpoint is not None and (not isinstance(endpoint, str) or not endpoint.strip()):
            return False
        if metadata.get(f"{role}_provider") == "custom" and endpoint is None:
            return False
    weights = run.get("weights")
    return (
        isinstance(weights, dict)
        and bool(weights)
        and all(
            isinstance(v, int | float) and not isinstance(v, bool) and np.isfinite(v) and v >= 0
            for v in weights.values()
        )
        and sum(weights.get(task, 0) for task in selected) > 0
    )


def group_cohorts(runs: dict[str, list[dict]]) -> dict[str, dict[str, list[dict]]]:
    groups: dict = {}
    for model, model_runs in runs.items():
        for run in model_runs:
            groups.setdefault(cohort_key(run), {}).setdefault(model, []).append(run)
    return groups


def paired_difference(
    left: list[dict],
    right: list[dict],
    task: str,
    *,
    bootstrap_samples: int = 10000,
    random_seed: int = 0,
) -> dict:
    """Mean left-minus-right difference, resampling matched seeds, never story pairs.

    Repeated runs of the same seed are averaged before pairing. Missing seeds or
    non-finite task scores are excluded. Fewer than two units cannot yield a CI.
    """
    if any((run.get("metadata") or {}).get("evaluation_complete") is False for run in left + right):
        raise ValueError("Incomplete evaluations cannot enter paired comparisons")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if any(not verified_provenance(run) for run in left + right):
        raise ValueError("Paired comparisons require verified provenance")
    if len({cohort_key(run) for run in left + right}) > 1:
        raise ValueError("Paired comparisons require one compatible protocol cohort")

    def units(runs):
        values = defaultdict(list)
        for run in runs:
            seed = run.get("seed")
            score = run.get("scores", {}).get(task)
            if seed is not None and isinstance(score, int | float) and np.isfinite(score):
                values[seed].append(score)
        return {seed: float(np.mean(scores)) for seed, scores in values.items()}

    a, b = units(left), units(right)
    seeds = sorted(a.keys() & b.keys())
    differences = np.asarray([a[seed] - b[seed] for seed in seeds])
    result = {
        "task": task,
        "n_matched_seeds": len(seeds),
        "matched_seeds": seeds,
        "difference": float(differences.mean()) if seeds else None,
        "ci95": None,
    }
    if len(seeds) >= 2:
        rng = np.random.default_rng(random_seed)
        means = rng.choice(differences, size=(bootstrap_samples, len(seeds))).mean(axis=1)
        result["ci95"] = [float(v) for v in np.quantile(means, [0.025, 0.975])]
    return result
