"""Exploratory embedding diversity under repeated and varied prompts.

Gwern's "Don't Repeat Yourself" proposes controlled prompt randomness and
embedding volume: https://gwern.net/creative-benchmark#possible-tasks.
We retain varied-prompt distances, but score repeated responses to identical
prompts so prompt differences alone cannot earn diversity credit. Distances
and effective rank describe geometry, not validated creativity or story quality.
"""

from __future__ import annotations

import random

import numpy as np
from tqdm.auto import tqdm

from ..client import Embedder, LLMClient
from ..data import DIVERSITY_CONCEPTS
from ..metrics import pairwise_cosine_distances
from .base import TaskResult, clamp01


def _effective_rank(embeddings: np.ndarray) -> float:
    """Entropy effective rank of the normalized, centered cloud's covariance.

    This is exp(entropy(variance proportions)), not geometric volume. A cloud
    with no variation has rank zero. No semantic similarity threshold is used.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.where(norms == 0, 1.0, norms)
    # Subtract a reference first so identical rows center to exact zero.
    centered = normalized - normalized[0]
    centered -= centered.mean(axis=0)
    eigenvalues = np.linalg.svd(centered, compute_uv=False) ** 2
    total = float(eigenvalues.sum())
    if total == 0:
        return 0.0
    proportions = eigenvalues[eigenvalues > 0] / total
    return float(np.exp(-np.sum(proportions * np.log(proportions))))


def dont_repeat_yourself(
    client: LLMClient,
    embedder: Embedder,
    *,
    samples: int = 8,
    repeats_per_prompt: int = 2,
    template: str = "Write a short story (150-250 words) about {}.",
    rng: random.Random | None = None,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    if samples < 2:
        raise ValueError("Need at least 2 distinct prompts to measure diversity")
    if repeats_per_prompt < 2:
        raise ValueError("Need at least 2 repeats per prompt")
    rng = rng or random.Random()
    categories = list(DIVERSITY_CONCEPTS)
    available = {category: list(concepts) for category, concepts in DIVERSITY_CONCEPTS.items()}
    for concepts in available.values():
        rng.shuffle(concepts)
    # Construct all prompts before making calls, ensuring samples means distinct
    # input prompts even when a custom template erases the concept differences.
    prompts: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    while any(available.values()) and len(prompts) < samples:
        for category in categories:
            if not available[category]:
                continue
            concept = available[category].pop()
            prompt = template.format(concept)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append((category, concept, prompt))
            if len(prompts) == samples:
                break
    if len(prompts) < samples:
        raise ValueError(
            "Requested more distinct prompts than the template and concept pool provide"
        )

    stories: list[dict] = []
    for prompt_id, (category, concept, prompt) in enumerate(
        tqdm(prompts, desc="Diversity", leave=False)
    ):
        for repeat in range(repeats_per_prompt):
            text = client.generate(prompt, temperature=0.9, max_tokens=2000)
            stories.append(
                {
                    "prompt_id": prompt_id,
                    "repeat": repeat,
                    "prompt": prompt,
                    "category": category,
                    "concept": concept,
                    "text": text,
                }
            )
            if verbose:
                print(f"  prompt {prompt_id + 1}, repeat {repeat + 1}: {len(text.split())} words")

    embeddings = embedder.embed([story["text"] for story in stories])
    distances = pairwise_cosine_distances(embeddings)
    i, j = np.triu_indices(len(stories), k=1)
    prompt_ids = np.array([story["prompt_id"] for story in stories])
    within = prompt_ids[i] == prompt_ids[j]
    within_distance = float(np.mean(distances[within]))
    between_distance = float(np.mean(distances[~within]))

    return TaskResult(
        name="diversity",
        score=clamp01(within_distance / 2),
        metrics={
            "within_prompt_mean_distance": within_distance,
            "between_prompt_mean_distance": between_distance,
            "mean_pairwise_distance": float(np.mean(distances)),
            "min_pairwise_distance": float(np.min(distances)),
            "std_pairwise_distance": float(np.std(distances)),
            "effective_rank": _effective_rank(embeddings),
            "samples": samples,
            "repeats_per_prompt": repeats_per_prompt,
            "stories_generated": len(stories),
            "quality_validated": False,
            "score_exploratory": True,
        },
        details={"stories": stories},
    )
