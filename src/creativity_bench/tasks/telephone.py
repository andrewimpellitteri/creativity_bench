"""Telephone game: expand a summary into a story, re-summarize, repeat.

A model with no creative drive collapses to a fixed point: the summary stops
changing. Score is the fraction of iterations survived before consecutive
summaries become near-identical (semantically and lexically).
"""

from __future__ import annotations

from tqdm.auto import tqdm

from ..client import Embedder, LLMClient
from ..metrics import cosine_similarity, lexical_similarity
from .base import TaskResult, clamp01

SEMANTIC_CONVERGENCE = 0.95
LEXICAL_CONVERGENCE = 0.85


def telephone_game(
    client: LLMClient,
    embedder: Embedder,
    *,
    seed_text: str,
    max_iter: int = 8,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    if not seed_text.strip():
        raise ValueError("Seed text cannot be empty")

    summary = seed_text
    previous_embedding = embedder.embed_one(summary)
    semantic_sims: list[float] = []
    lexical_sims: list[float] = []
    survived = max_iter
    transcript: list[dict] = []

    for i in tqdm(range(max_iter), desc="Telephone game", leave=False):
        story = client.generate(
            f"Expand this summary into a detailed short story:\n\n{summary}",
            temperature=0.8,
            max_tokens=2000,
        )
        new_summary = client.generate(
            f"Summarize this story in one sentence:\n\n{story}",
            temperature=0.3,
            max_tokens=2000,
        )

        new_embedding = embedder.embed_one(new_summary)
        semantic_sim = cosine_similarity(previous_embedding, new_embedding)
        lexical_sim = lexical_similarity(new_summary, summary)
        semantic_sims.append(semantic_sim)
        lexical_sims.append(lexical_sim)
        transcript.append(
            {"summary": new_summary, "semantic_sim": semantic_sim, "lexical_sim": lexical_sim}
        )
        if verbose:
            print(f"  iter {i + 1}: sem={semantic_sim:.3f} lex={lexical_sim:.3f} :: {new_summary}")

        if semantic_sim >= SEMANTIC_CONVERGENCE and lexical_sim >= LEXICAL_CONVERGENCE:
            survived = i
            break

        summary = new_summary
        previous_embedding = new_embedding

    mean_drift = 1.0 - (sum(semantic_sims) / len(semantic_sims)) if semantic_sims else 0.0

    return TaskResult(
        name="telephone",
        score=clamp01(survived / max_iter),
        metrics={
            "iterations_survived": survived,
            "max_iterations": max_iter,
            "mean_semantic_drift": mean_drift,
            "semantic_similarities": semantic_sims,
            "lexical_similarities": lexical_sims,
        },
        details={"seed": seed_text, "transcript": transcript},
    )
