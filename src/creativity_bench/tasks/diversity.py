"""Don't repeat yourself: how different are stories generated from similar prompts?

Generates one story per prompt (rotating through concept categories) and
scores the mean pairwise cosine distance between story embeddings. Higher
means the model spreads its outputs over a wider region of idea-space.
"""

from __future__ import annotations

import random

import numpy as np
from tqdm.auto import tqdm

from ..client import Embedder, LLMClient
from ..data import DIVERSITY_CONCEPTS
from ..metrics import pairwise_cosine_distances
from .base import TaskResult, clamp01


def dont_repeat_yourself(
    client: LLMClient,
    embedder: Embedder,
    *,
    samples: int = 8,
    template: str = "Write a short story (150-250 words) about {}.",
    rng: random.Random | None = None,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    if samples < 2:
        raise ValueError("Need at least 2 samples to measure diversity")
    rng = rng or random.Random()

    categories = list(DIVERSITY_CONCEPTS)
    stories: list[dict] = []

    for i in tqdm(range(samples), desc="Diversity", leave=False):
        category = categories[i % len(categories)]
        concept = rng.choice(DIVERSITY_CONCEPTS[category])
        prompt = template.format(concept)
        text = client.generate(prompt, temperature=0.9, max_tokens=2000)
        stories.append({"category": category, "concept": concept, "text": text})
        if verbose:
            print(f"  story {i + 1} ({category}: {concept}): {len(text.split())} words")

    embeddings = embedder.embed([story["text"] for story in stories])
    distances = pairwise_cosine_distances(embeddings)
    mean_distance = float(np.mean(distances))
    min_distance = float(np.min(distances))

    return TaskResult(
        name="diversity",
        score=clamp01(mean_distance),
        metrics={
            "mean_pairwise_distance": mean_distance,
            "min_pairwise_distance": min_distance,
            "std_pairwise_distance": float(np.std(distances)),
            "samples": samples,
        },
        details={"stories": stories},
    )
