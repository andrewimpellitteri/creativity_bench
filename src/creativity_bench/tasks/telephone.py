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

Conditions and censoring (design audit: "eight rounds is a ceiling, not an
observed collapse time"):
- ``conditions`` separates deterministic (temperature 0) from stochastic
  (temperature 0.8) generation. Stochastic wording alone can prevent
  convergence indefinitely, so the two conditions are reported separately and
  must never be pooled into one number.
- With ``premises`` the task walks one chain per premise per condition and
  reports a censored survival curve per condition: at each round t, the
  fraction of premises not yet collapsed. Chains that never collapse within
  the budget are RIGHT-CENSORED (``collapse_iter=None``); they contribute the
  full budget to the score and count as surviving in the curve. The curve is
  therefore a lower bound on true survival, not an observed collapse time.
- The legacy single-premise, single-condition call (``seed_text`` only) is
  unchanged: same score, ``iterations_survived``, ``right_censored`` and
  similarity metrics as before.
"""

from __future__ import annotations

from collections.abc import Sequence

from tqdm.auto import tqdm

from ..client import Embedder, LLMClient
from ..metrics import cosine_similarity, lexical_similarity
from .base import TaskResult, clamp01

SEMANTIC_CONVERGENCE = 0.95
LEXICAL_CONVERGENCE = 0.85

# (expansion temperature, summarization temperature) per condition. The
# deterministic condition pins both calls to temperature 0.
CONDITION_TEMPERATURES: dict[str, tuple[float, float]] = {
    "deterministic": (0.0, 0.0),
    "stochastic": (0.8, 0.3),
}


def _walk_chain(
    client: LLMClient,
    embedder: Embedder,
    *,
    premise: str,
    max_iter: int,
    condition: str,
    verbose: bool,
) -> dict:
    """Play one telephone chain; return its collapse time (or censoring)."""
    expansion_temp, summary_temp = CONDITION_TEMPERATURES[condition]
    summary = premise
    story = ""
    previous_embedding = None
    semantic_sims: list[float] = []
    lexical_sims: list[float] = []
    collapse_iter: int | None = None
    transcript: list[dict] = []

    for i in tqdm(range(max_iter), desc=f"Telephone game ({condition})", leave=False):
        new_story = client.generate(
            f"Expand this summary into a detailed short story:\n\n{summary}",
            temperature=expansion_temp,
            max_tokens=2000,
        )
        new_summary = client.generate(
            f"Summarize this story in one sentence:\n\n{new_story}",
            temperature=summary_temp,
            max_tokens=2000,
        )

        # Fixed point iff two successive expansions are the same. Exact match
        # first (preferred); similarity thresholds are only a fallback.
        if story and (
            new_story == story or _near_identical(new_story, story, previous_embedding, embedder)
        ):
            collapse_iter = i
            transcript.append(
                {"story": new_story, "summary": new_summary, "exact_match": new_story == story}
            )
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
                "story": new_story,
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

    return {
        "premise": premise,
        "condition": condition,
        "collapse_iter": collapse_iter,
        "right_censored": collapse_iter is None,
        "semantic_sims": semantic_sims,
        "lexical_sims": lexical_sims,
        "transcript": transcript,
    }


def telephone_game(
    client: LLMClient,
    embedder: Embedder,
    *,
    seed_text: str = "",
    max_iter: int = 8,
    premises: list[str] | None = None,
    conditions: Sequence[str] = ("stochastic",),
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    """Score telephone chains, separating conditions and censoring at budget.

    Score = mean over chains of (collapse time / max_iter), with right-censored
    chains contributing the full budget. Per-condition survival curves are in
    metrics["survival_curves"]; per-chain outcomes in details["chains"].
    """
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    unknown = [c for c in conditions if c not in CONDITION_TEMPERATURES]
    if unknown:
        raise ValueError(
            f"Unknown conditions: {', '.join(unknown)}. "
            f"Available: {', '.join(CONDITION_TEMPERATURES)}"
        )
    if premises is None:
        if not seed_text.strip():
            raise ValueError("Seed text cannot be empty")
        premises = [seed_text]
    if not premises or any(not p.strip() for p in premises):
        raise ValueError("Premises must be nonempty strings")

    chains: list[dict] = []
    for condition in conditions:
        for premise in premises:
            chains.append(
                _walk_chain(
                    client,
                    embedder,
                    premise=premise,
                    max_iter=max_iter,
                    condition=condition,
                    verbose=verbose,
                )
            )

    # Right-censored chains contribute the full budget: the budget is a
    # ceiling, not an observed collapse time.
    scores = [
        clamp01((c["collapse_iter"] if c["collapse_iter"] is not None else max_iter) / max_iter)
        for c in chains
    ]
    score = sum(scores) / len(scores)

    # Survival curve per condition: at each t in 1..max_iter, the fraction of
    # premises not yet collapsed (censored chains count as surviving).
    survival_curves: dict[str, list[float]] = {}
    for condition in conditions:
        cond_chains = [c for c in chains if c["condition"] == condition]
        survival_curves[condition] = [
            sum(
                c["collapse_iter"] is None or c["collapse_iter"] > t
                for c in cond_chains
            ) / len(cond_chains)
            for t in range(1, max_iter + 1)
        ]

    metrics: dict = {
        "max_iterations": max_iter,
        "conditions": list(conditions),
        "n_premises": len(premises),
        "observed_collapses": sum(c["collapse_iter"] is not None for c in chains),
        "censored_chains": sum(c["right_censored"] for c in chains),
        "survival_curves": survival_curves,
    }
    details: dict = {
        "premises": list(premises),
        "chains": chains,
    }

    if len(chains) == 1:
        # Legacy single-chain contract: identical metrics/details keys.
        chain = chains[0]
        survived = chain["collapse_iter"] if chain["collapse_iter"] is not None else max_iter
        semantic_sims = chain["semantic_sims"]
        metrics.update(
            {
                "iterations_survived": survived,
                "right_censored": chain["right_censored"],
                "mean_semantic_drift": 1.0 - (sum(semantic_sims) / len(semantic_sims))
                if semantic_sims
                else 0.0,
                "semantic_similarities": semantic_sims,
                "lexical_similarities": chain["lexical_sims"],
            }
        )
        details["seed"] = premises[0]
        details["transcript"] = chain["transcript"]

    return TaskResult(name="telephone", score=score, metrics=metrics, details=details)


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
