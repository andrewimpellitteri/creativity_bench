"""Orchestrates benchmark tasks and computes the composite creativity score."""

from __future__ import annotations

import datetime as dt
import hashlib
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
PROTOCOL_VERSION = "0.4-validity"

DEFAULT_WEIGHTS = {
    "free_association": 0.20,
    "telephone": 0.20,
    "camels_back": 0.20,
    "diversity": 0.20,
    "style_transfer": 0.20,
    "odd_one_out": 0.20,
    "subversion": 0.20,
    "shaggy_dog": 0.20,
    "same_but_different": 0.20,
}

# Task sizes: (full, fast)
_SIZES = {
    "n_words": (40, 10),
    "judges": (5, 3),
    "max_iter": (8, 3),
    "telephone_premises": (4, 2),
    "max_edits": (8, 3),
    "samples": (8, 4),
    "n_stories": (7, 2),
    "n_lists": (2, 1),
    "n_premises": (4, 1),
    "sub_runs": (3, 2),
    "distinct_premises": (6, 2),
    "distinct_attempts": (10, 3),
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
    embedder: Embedder | None,
    *,
    tasks: list[str] | None = None,
    seed: int | None = None,
    fast: bool = False,
    verbose: bool = False,
    weights: dict[str, float] | None = None,
) -> RunResult:
    task_names = list(TASKS) if tasks is None else list(tasks)
    if not task_names or len(task_names) != len(set(task_names)):
        raise ValueError("Select at least one task without duplicates")
    unknown = set(task_names) - set(TASKS)
    if unknown:
        raise ValueError(
            f"Unknown tasks: {', '.join(sorted(unknown))}. Available: {', '.join(TASKS)}"
        )

    embedding_tasks = {"telephone", "diversity", "style_transfer", "odd_one_out"}
    if embedder is None and embedding_tasks.intersection(task_names):
        raise ValueError("Selected tasks require an embedder")

    seed = seed if seed is not None else random.randrange(2**31)
    rng = random.Random(seed)
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    size = {key: values[1] if fast else values[0] for key, values in _SIZES.items()}
    seed_text = rng.choice(data.STORY_PROMPTS)

    # Independent streams keep task inputs stable across subsets and ordering.
    task_rngs = {name: random.Random(f"{seed}:{name}") for name in TASKS}
    task_kwargs = {
        "free_association": dict(n_words=size["n_words"]),
        "telephone": dict(
            embedder=embedder,
            premises=task_rngs["telephone"].sample(data.STORY_PROMPTS, size["telephone_premises"]),
            conditions=("deterministic", "stochastic"),
            max_iter=size["max_iter"],
        ),
        "camels_back": dict(
            judge_client=judge_client,
            seed_text=seed_text,
            edit_requests=data.EDIT_REQUESTS,
            max_edits=size["max_edits"],
            rng=rng,
        ),
        "diversity": dict(embedder=embedder, samples=size["samples"], rng=rng),
        "shaggy_dog": dict(judge_client=judge_client, k=size["judges"], rng=rng),
        "style_transfer": dict(
            judge_client=judge_client,
            embedder=embedder,
            stories=data.SAMPLE_STORIES[: size["n_stories"]],
            genres=data.GENRES,
            rng=rng,
        ),
        "odd_one_out": dict(
            embedder=embedder, judge_client=judge_client, n_lists=size["n_lists"], rng=rng
        ),
        "same_but_different": dict(
            judge_client=judge_client,
            premises=task_rngs["same_but_different"].sample(
                data.CREATIVE_PREMISES, size["distinct_premises"]
            ),
            attempts=size["distinct_attempts"],
        ),
        "subversion": dict(
            judge_client=judge_client,
            premises=data.STORY_PROMPTS[: size["n_premises"]],
            runs=size["sub_runs"],
        ),
    }

    usage_before = {
        role: dict(vars(obj.usage)) if obj is not None else {}
        for role, obj in (("generation", client), ("judge", judge_client), ("embedding", embedder))
    }
    log_offsets = {
        "generation": len(getattr(client, "request_log", [])),
        "judge": len(getattr(judge_client, "request_log", [])),
    }
    started = time.monotonic()
    task_results: dict[str, TaskResult] = {}
    for name in task_names:
        print(f"\n=== {name} ===")
        kwargs = dict(task_kwargs[name])
        if "rng" in kwargs:
            kwargs["rng"] = task_rngs[name]
        result = TASKS[name](client, verbose=verbose, **kwargs)
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
            "judge_provider": _provider_field(judge_client, "name"),
            "judge_base_url": _provider_field(judge_client, "base_url"),
            "embed_provider": _provider_field(embedder, "name"),
            "embed_base_url": _provider_field(embedder, "base_url"),
            "generation_provider": _provider_field(client, "name"),
            "generation_base_url": _provider_field(client, "base_url"),
            "generation_settings": _settings(client),
            "judge_settings": _settings(judge_client),
            "protocol_fingerprint": protocol_fingerprint(),
            "embed_model": embedder.model if embedder is not None else None,
            "fast": fast,
            "evaluation_complete": not any(
                result.metrics.get(key, 0)
                for result in task_results.values()
                for key in ("judge_unresolved", "unresolved_judgments", "generation_errors")
            ),
            "protocol_version": PROTOCOL_VERSION,
            "task_sizes": size,
            "telephone_conditions": ["deterministic", "stochastic"],
            "selected_tasks": task_names,
            "generation_usage": _usage_delta(client, usage_before["generation"]),
            "judge_usage": (
                _usage_delta(judge_client, usage_before["judge"])
                if judge_client is not client
                else "shared"
            ),
            "embedding_usage": _usage_delta(embedder, usage_before["embedding"]),
            "generation_requests": list(
                getattr(client, "request_log", [])[log_offsets["generation"] :]
            ),
            "judge_requests": (
                list(getattr(judge_client, "request_log", [])[log_offsets["judge"] :])
                if judge_client is not client
                else "shared"
            ),
        },
    )


def _provider_field(obj, name: str):
    return getattr(getattr(obj, "provider", None), name, None)


def _settings(client) -> dict:
    return {
        "sampling": "task-defined; see protocol source fingerprint and request log",
        "temperature_supported": getattr(client, "_temperature_supported", None),
        "empty_length_retry": "double token budget up to 16000",
        "max_retries": getattr(client, "max_retries", None),
    }


def _usage_delta(client, before: dict) -> dict:
    if client is None:
        return {}
    return {key: value - before.get(key, 0) for key, value in vars(client.usage).items()}


def protocol_fingerprint() -> str:
    """Hash scoring, prompts, corpus and request policy; independent of git state."""
    root = Path(__file__).parent
    paths = [
        root / name for name in ("runner.py", "data.py", "judge.py", "metrics.py", "client.py")
    ]
    paths += sorted((root / "tasks").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
    print(f"Composite: {result.composite:.3f} (exploratory)")
    print("\nTask scores (all in [0, 1]):")
    for name, task_result in result.task_results.items():
        print(f"  {name:<20} {task_result.score:.3f}")
    usage = result.metadata["generation_usage"]
    print(
        f"\nTokens: {usage['prompt_tokens']:,} in / {usage['completion_tokens']:,} out "
        f"across {usage['requests']} requests, {result.duration_seconds:.0f}s"
    )
    print("=========================================")
