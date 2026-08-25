"""Extreme style transfer: summarize a story, rewrite it in a different genre.

Score is the mean embedding distance between the original and the transferred
story (divergence). Fidelity to the summary is reported alongside so a model
can't score high by simply ignoring the plot.
"""

from __future__ import annotations

import random

import numpy as np
from tqdm.auto import tqdm

from ..client import Embedder, LLMClient
from ..metrics import cosine_similarity
from .base import TaskResult, clamp01


def style_transfer(
    client: LLMClient,
    embedder: Embedder,
    *,
    stories: list[dict],
    genres: list[str],
    rng: random.Random | None = None,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    rng = rng or random.Random()
    transfers: list[dict] = []
    divergences: list[float] = []
    fidelities: list[float] = []

    for story in tqdm(stories, desc="Style transfer", leave=False):
        original = story["text"]
        summary = client.generate(
            f"Summarize the following story in 2-3 sentences:\n\n{original}",
            temperature=0.3,
            max_tokens=2000,
        )
        candidates = [genre for genre in genres if genre != story["genre"]]
        target_genre = rng.choice(candidates)
        transferred = client.generate(
            f"Using only the summary below, write a new short story in the genre "
            f"'{target_genre}'.\n\nSUMMARY:\n{summary}",
            temperature=0.8,
            max_tokens=2000,
        )

        original_emb, summary_emb, transferred_emb = embedder.embed(
            [original, summary, transferred]
        )
        divergence = 1.0 - cosine_similarity(original_emb, transferred_emb)
        fidelity = cosine_similarity(summary_emb, transferred_emb)
        divergences.append(divergence)
        fidelities.append(fidelity)
        transfers.append(
            {
                "original_genre": story["genre"],
                "target_genre": target_genre,
                "summary": summary,
                "transferred": transferred,
                "divergence": divergence,
                "fidelity": fidelity,
            }
        )
        if verbose:
            print(
                f"  {story['genre']} -> {target_genre}: "
                f"divergence={divergence:.3f} fidelity={fidelity:.3f}"
            )

    return TaskResult(
        name="style_transfer",
        score=clamp01(float(np.mean(divergences))),
        metrics={
            "mean_divergence": float(np.mean(divergences)),
            "mean_fidelity": float(np.mean(fidelities)),
            "stories": len(stories),
        },
        details={"transfers": transfers},
    )
