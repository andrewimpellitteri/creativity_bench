"""Telephone game: expand a summary into a story, re-summarize, repeat.

Gwern, "Telephone Game" (https://gwern.net/creative-benchmark#possible-tasks,
Iteration section): "starting with a seed prompt containing a summary to
expand, then summarize it, then prompt with the summary, and so on. The score
is the number of iterations until two successive expansions are the same
(higher = better)."

"The less creative and more mode-collapsed a model, the faster you would
expect it to hit a fixed point and repeat the same output."

Fixed-point detection fidelity:
- The comparison is between successive EXPANSIONS (the stories), not the
  intermediate summaries, per "two successive expansions are the same".
- Exact text match is preferred. Gwern expects this to suffice: "I expect that
  an exact text match would be enough given the flattened-logits of LLMs
  eliminates stochastic variation". Only as a fallback do we loosen to lexical
  (edit-distance / ROUGE-L) and embedding similarity thresholds.
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
    story = ""
    previous_embedding = None
    semantic_sims: list[float] = []
    lexical_sims: list[float] = []
    survived = max_iter
    transcript: list[dict] = []

    for i in tqdm(range(max_iter), desc="Telephone game", leave=False):
        new_story = client.generate(
            f"Expand this summary into a detailed short story:\n\n{summary}",
            temperature=0.8,
            max_tokens=2000,
        )
        new_summary = client.generate(
            f"Summarize this story in one sentence:\n\n{new_story}",
            temperature=0.3,
            max_tokens=2000,
        )

        # Fixed point iff two successive expansions are the same. Exact match
        # first (preferred); similarity thresholds are only a fallback.
        if story and (
            new_story == story or _near_identical(new_story, story, previous_embedding, embedder)
        ):
            survived = i
            transcript.append({"summary": new_summary, "exact_match": new_story == story})
            break

        new_embedding = embedder.embed_one(new_story)
        semantic_sim = (
            cosine_similarity(previous_embedding, new_embedding)
            if previous_embedding is not None
            else 1.0
        )
        lexical_sim = lexical_similarity(new_story, story) if story else 1.0
        semantic_sims.append(semantic_sim)
        lexical_sims.append(lexical_sim)
        transcript.append(
            {
                "summary": new_summary,
                "semantic_sim": semantic_sim,
                "lexical_sim": lexical_sim,
                "exact_match": new_story == story,
            }
        )
        if verbose:
            print(f"  iter {i + 1}: sem={semantic_sim:.3f} lex={lexical_sim:.3f} :: {new_summary}")

        summary = new_summary
        story = new_story
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


def _near_identical(a: str, b: str, b_embedding, embedder: Embedder) -> bool:
    """Fallback convergence check when the expansions differ verbatim.

    Gwern: "it might prove necessary to loosen it to an edit-distance or
    possibly similarity in a text embedding." We require BOTH lexical and
    embedding similarity to clear their thresholds so trivial rewording does
    not count as a fixed point."""
    return (
        cosine_similarity(b_embedding, embedder.embed_one(a)) >= SEMANTIC_CONVERGENCE
        and lexical_similarity(a, b) >= LEXICAL_CONVERGENCE
    )
