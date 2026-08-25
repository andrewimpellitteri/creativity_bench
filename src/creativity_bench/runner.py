"""Orchestrates benchmark tasks and computes the composite creativity score."""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import data
from .client import Embedder, LLMClient
from .tasks import TASKS, TaskResult

SCHEMA_VERSION = 2

DEFAULT_WEIGHTS = {
    "free_association": 0.20,
    "telephone": 0.20,
    "camels_back": 0.20,
    "diversity": 0.20,
    "style_transfer": 0.20,
    "odd_one_out": 0.20,
}

# Task sizes: (full, fast)
_SIZES = {
    "n_words": (40, 10),
    "max_iter": (8, 3),
    "max_edits": (8, 3),
    "samples": (8, 4),
    "n_stories": (7, 2),
    "n_lists": (2, 1),
}


@dataclass
class RunResult:
    model: str
    provider: str
    task_results: dict[str, TaskResult]
    composite: float
    weights: dict[str, float]
    seed: int
    duration_seconds: float
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "provider": self.provider,
            "composite": self.composite,
            "scores": {name: result.score for name, result in self.task_results.items()},
            "weights": self.weights,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
            "tasks": {name: result.to_dict() for name, result in self.task_results.items()},
        }


def composite_score(task_results: dict[str, TaskResult], weights: dict[str, float]) -> float:
    active = {name: weights[name] for name in task_results if weights.get(name, 0) > 0}
    total_weight = sum(active.values())
    if total_weight == 0:
        return 0.0
    return sum(task_results[name].score * weight for name, weight in active.items()) / total_weight


def run_benchmark(
    client: LLMClient,
    judge_client: LLMClient,
    embedder: Embedder,
    *,
    tasks: list[str] | None = None,
    seed: int | None = None,
    fast: bool = False,
    verbose: bool = False,
    weights: dict[str, float] | None = None,
) -> RunResult:
    task_names = tasks or list(TASKS)
    unknown = set(task_names) - set(TASKS)
    if unknown:
        raise ValueError(
            f"Unknown tasks: {', '.join(sorted(unknown))}. Available: {', '.join(TASKS)}"
        )

    seed = seed if seed is not None else random.randrange(2**31)
    rng = random.Random(seed)
    weights = weights or DEFAULT_WEIGHTS
    size = {key: values[1] if fast else values[0] for key, values in _SIZES.items()}
    seed_text = rng.choice(data.STORY_PROMPTS)

    task_kwargs = {
        "free_association": dict(n_words=size["n_words"]),
        "telephone": dict(embedder=embedder, seed_text=seed_text, max_iter=size["max_iter"]),
        "camels_back": dict(
            judge_client=judge_client,
            seed_text=seed_text,
            edit_requests=data.EDIT_REQUESTS,
            max_edits=size["max_edits"],
            rng=rng,
        ),
        "diversity": dict(embedder=embedder, samples=size["samples"], rng=rng),
        "style_transfer": dict(
            embedder=embedder,
            stories=data.SAMPLE_STORIES[: size["n_stories"]],
            genres=data.GENRES,
            rng=rng,
        ),
        "odd_one_out": dict(embedder=embedder, n_lists=size["n_lists"], rng=rng),
    }

    started = time.monotonic()
    task_results: dict[str, TaskResult] = {}
    for name in task_names:
        print(f"\n=== {name} ===")
        result = TASKS[name](client, verbose=verbose, **task_kwargs[name])
        task_results[name] = result
        print(f"    score: {result.score:.3f}")

    duration = time.monotonic() - started
    return RunResult(
        model=client.model,
        provider=client.provider.name,
        task_results=task_results,
        composite=composite_score(task_results, weights),
        weights=weights,
        seed=seed,
        duration_seconds=duration,
        metadata={
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "judge_model": judge_client.model,
            "embed_model": embedder.model,
            "fast": fast,
            "generation_usage": vars(client.usage),
            "judge_usage": vars(judge_client.usage) if judge_client is not client else "shared",
            "embedding_usage": vars(embedder.usage),
        },
    )


def save_run(result: RunResult, runs_dir: str | Path = "runs") -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", result.model)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = runs_dir / f"{safe_model}_{stamp}_{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


def print_results(result: RunResult) -> None:
    print("\n============= Final Results =============")
    print(f"Model:     {result.model} ({result.provider})")
    print(f"Composite: {result.composite:.3f}")
    print("\nTask scores (all in [0, 1]):")
    for name, task_result in result.task_results.items():
        print(f"  {name:<20} {task_result.score:.3f}")
    usage = result.metadata["generation_usage"]
    print(
        f"\nTokens: {usage['prompt_tokens']:,} in / {usage['completion_tokens']:,} out "
        f"across {usage['requests']} requests, {result.duration_seconds:.0f}s"
    )
    print("=========================================")
