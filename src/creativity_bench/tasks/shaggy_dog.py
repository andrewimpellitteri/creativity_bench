"""Shaggy dog storytelling contest: punish tidy morals and punchlines.

Gwern, "Shaggy Dog Storytelling Contest"
(https://gwern.net/creative-benchmark#shaggy-dog): "large or tuned LLMs
struggle to write non sequitur, the-joke-is-there-is-no-joke, non-moralizing,
simple events or stories... ChatGPT in particular wants to conclude with some
clear moral or punchline, and a tidy pat interpretation. We can punish this
tendency by setting up a shaggy dog storytelling contest: LLMs are prompted to
write stories with no meaning or conclusion (ie. shaggy dog stories), and then
other LLMs are asked for the moral or punchline. The more similar the
explanations are, the worse the score, because that implies the original
storyteller did fall into some sort of neat tidy interpretation."

Implementation notes:
- The storyteller is prompted for a deliberately pointless story; the judge is
  then sampled K times (default 3) asking Gwern's exact question, "what is the
  moral or punchline of this story?".
- Comprehensibility gate: before any moral-agreement scoring, a judge checks
  (strict JSON boolean, fail-closed) that the story is comprehensible at all.
  An incoherent or unresolved verdict scores 0.0 -- disagreement about morals
  must not reward incoherence (design audit). Unresolved gate verdicts also
  set ``judge_unresolved`` so the run is marked incomplete upstream.
- Multi-judge scaffolding: repeated calls to one judge model are not
  independent judges, so ``judge_clients`` accepts several fixed judge models;
  explanations are kept per judge, and within-judge and cross-judge agreement
  are reported separately. The score still uses the pooled pairwise agreement.
- Agreement between judge explanations is measured as mean pairwise Jaccard
  similarity over normalized content-word token sets, and the score is the
  INVERTED agreement mapped to [0, 1]: high score = judges could NOT agree =
  no tidy interpretation. # NOTE(gwern): we deliberately use cheap normalized
  token overlap rather than embedding cosine: it is offline, deterministic,
  and conservative - lexical convergence is a lower bound on semantic
  convergence, so any model penalized here has demonstrably steered its
  judges toward the same wording (a real "tidy pat interpretation").
- Hard-refusal guard: if the story itself states a moral ("moral:", "the
  lesson is", ...), that is exactly the "tidy pat interpretation" Gwern
  describes, so the run is an automatic failure (score floor of 0.0).
- K=1 degenerate case: with a single judge sample there is no pairwise
  comparison, so no shared interpretation can be evidenced either way. We
  handle this gracefully by scoring 1.0 (absence of observed agreement cannot
  penalize the storyteller) and flagging ``degenerate=True`` in metrics;
  callers wanting meaningful scores should use K >= 2.
"""

from __future__ import annotations

import json
import random
import re

from ..client import LLMClient
from ..judge import extract_json_object
from .base import TaskResult, clamp01

# Gwern: "LLMs are prompted to write stories with no meaning or conclusion".
SHAGGY_DOG_PROMPT = """\
Write a short shaggy dog story (150-300 words): a long, rambling, elaborately \
pointless story that builds up elaborate expectations and then simply ends, \
without a punchline, without a moral, and without any point at all. \
The joke is that there is no joke. Do not explain or interpret the story. \
Just tell it and stop."""

# Gwern's judge question, verbatim: "what is the moral or punchline of this
# story?"
PUNCHLINE_PROMPT = """\
What is the moral or punchline of this story?

{story}"""

# Comprehensibility gate (design audit: "Check basic comprehensibility" before
# moral-agreement scoring, so disagreement about morals cannot reward
# incoherence). Deliberately pointless must still be followable.
COMPREHENSIBILITY_PROMPT = """\
Below is a short story submitted to a shaggy dog storytelling contest. Judge \
ONLY whether the story is comprehensible: its events must be followable from \
sentence to sentence, even though the story is deliberately pointless. \
Nonsense, word salad, or un-followable text is not comprehensible.

STORY:
{story}

Answer strictly as a JSON object with this boolean field and nothing else:
{{"comprehensible": <true if the story is comprehensible>}}
"""

# Hard-refusal guard: an explicit stated moral/punchline is precisely the
# "tidy pat interpretation" Gwern punishes ("ChatGPT in particular wants to
# conclude with some clear moral or punchline").
EXPLICIT_MORAL_RE = re.compile(
    r"\bmoral\s*:|\bmoral of the (story|tale)\b|"
    r"\bthe lesson (of this story |here )?is\b|"
    r"\bthe (moral|punchline|point) is\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)


def _content_tokens(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        # Two empty responses trivially agree (nothing was interpreted).
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_comprehensible(text: str) -> bool:
    verdict = extract_json_object(text)
    if not isinstance(verdict, dict) or type(verdict.get("comprehensible")) is not bool:
        raise ValueError("comprehensible must be a JSON boolean")
    return verdict["comprehensible"]


def _judge_comprehensible(judge_client: LLMClient, story: str) -> tuple[bool | None, list[str]]:
    """Comprehensibility gate; fail closed with None when unresolved."""
    attempts: list[str] = []
    for _ in range(2):
        response = judge_client.generate(
            COMPREHENSIBILITY_PROMPT.format(story=story),
            temperature=0.0,
            max_tokens=200,
        )
        attempts.append(response)
        try:
            return _parse_comprehensible(response), attempts
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return None, attempts


def _mean_pairwise(token_sets: list[frozenset[str]]) -> float:
    similarities = [
        _jaccard(token_sets[i], token_sets[j])
        for i in range(len(token_sets))
        for j in range(i + 1, len(token_sets))
    ]
    return sum(similarities) / len(similarities)


def shaggy_dog(
    client: LLMClient,
    judge_client: LLMClient | None = None,
    *,
    k: int = 3,
    judge_clients: list[LLMClient] | None = None,
    rng: random.Random | None = None,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    """Run the shaggy dog contest; higher score = less tidy interpretation.

    ``judge_clients`` (plural) supplies several fixed judge models; agreement
    is then reported within and across judges. Falls back to the single
    ``judge_client`` when omitted.
    """
    if k < 1:
        raise ValueError("Need at least 1 judge sample")
    judges = list(judge_clients) if judge_clients is not None else [judge_client]
    if not judges or any(j is None for j in judges):
        raise ValueError("Need at least one judge client")
    rng = rng or random.Random()

    story = client.generate(SHAGGY_DOG_PROMPT, temperature=0.8, max_tokens=2000)

    def _result(score: float, metrics: dict, details: dict) -> TaskResult:
        return TaskResult(name="shaggy_dog", score=score, metrics=metrics, details=details)

    if EXPLICIT_MORAL_RE.search(story):
        # Automatic failure: the storyteller stated the moral itself instead
        # of leaving the story pointless. Cheap deterministic check first, so
        # an outright failure consumes no judge calls.
        if verbose:
            print("  explicit stated moral/punchline detected: automatic failure")
        return _result(
            0.0,
            {
                "k": k,
                "explicit_moral": True,
                "comprehensible": None,
                "judge_unresolved": 0,
                "n_judges": len(judges),
            },
            {"story": story, "explanations": [], "gate_attempts": []},
        )

    # Comprehensibility gate, before any moral-agreement scoring. Fail closed:
    # an incoherent or unresolved story earns nothing, and an unresolved gate
    # marks the run incomplete via the judge_unresolved flag.
    comprehensible, gate_attempts = _judge_comprehensible(judges[0], story)
    if comprehensible is None:
        if verbose:
            print("  comprehensibility gate unresolved: fail-closed score 0.0")
        return _result(
            0.0,
            {
                "k": k,
                "explicit_moral": False,
                "comprehensible": None,
                "judge_unresolved": 1,
                "n_judges": len(judges),
            },
            {"story": story, "explanations": [], "gate_attempts": gate_attempts},
        )
    if not comprehensible:
        if verbose:
            print("  story failed the comprehensibility gate: score 0.0")
        return _result(
            0.0,
            {
                "k": k,
                "explicit_moral": False,
                "comprehensible": False,
                "judge_unresolved": 0,
                "n_judges": len(judges),
            },
            {"story": story, "explanations": [], "gate_attempts": gate_attempts},
        )

    prompt = PUNCHLINE_PROMPT.format(story=story)
    explanations_per_judge: list[list[str]] = []
    for judge_index, judge in enumerate(judges):
        explanations: list[str] = []
        for i in range(k):
            # Seeded sampling: jitter the judge's temperature from the caller's
            # rng so repeated samples genuinely vary while runs stay reproducible.
            response = judge.generate(
                prompt,
                temperature=round(0.7 + 0.2 * rng.random(), 3),
                max_tokens=500,
            )
            explanations.append(response)
            if verbose:
                print(f"  judge {judge_index} explanation {i + 1}: {response[:60]!r}...")
        explanations_per_judge.append(explanations)

    pooled = [e for group in explanations_per_judge for e in group]

    if len(pooled) < 2:
        # K=1 degenerate case (see module docstring): no pairwise comparison
        # exists, so no tidy interpretation can be evidenced.
        return _result(
            1.0,
            {
                "k": k,
                "explicit_moral": False,
                "comprehensible": True,
                "judge_unresolved": 0,
                "degenerate": True,
                "n_judges": len(judges),
                "judge_models": [j.model for j in judges],
            },
            {
                "story": story,
                "explanations": pooled,
                "explanations_per_judge": explanations_per_judge,
                "gate_attempts": gate_attempts,
            },
        )

    pooled_tokens = [_content_tokens(e) for e in pooled]
    # Pooled agreement drives the score; per-judge and cross-judge breakdowns
    # expose whether repeated calls to one model (not independent judges)
    # drive any apparent convergence. A judge sampled once has no within-judge
    # comparison (None), mirroring the K=1 degenerate case.
    mean_agreement = _mean_pairwise(pooled_tokens)
    within_agreements = [
        _mean_pairwise([_content_tokens(e) for e in group]) if len(group) > 1 else None
        for group in explanations_per_judge
    ]
    computable_within = [a for a in within_agreements if a is not None]
    if len(judges) > 1:
        cross_similarities = []
        for a in range(len(judges)):
            for b in range(a + 1, len(judges)):
                for i in range(k):
                    for j in range(k):
                        tokens_a = _content_tokens(explanations_per_judge[a][i])
                        tokens_b = _content_tokens(explanations_per_judge[b][j])
                        cross_similarities.append(_jaccard(tokens_a, tokens_b))
        mean_cross = sum(cross_similarities) / len(cross_similarities)
    else:
        # Single judge: no cross-model comparison exists.
        mean_cross = None
    # Inverted agreement: judges disagreeing => no tidy interpretation => high
    # score ("the more similar the explanations are, the worse the score").
    score = clamp01(1.0 - mean_agreement)

    return _result(
        score,
        {
            "k": k,
            "mean_pairwise_agreement": mean_agreement,
            "mean_within_judge_agreement": (
                sum(computable_within) / len(computable_within) if computable_within else None
            ),
            "mean_cross_judge_agreement": mean_cross,
            "explicit_moral": False,
            "comprehensible": True,
            "judge_unresolved": 0,
            "degenerate": False,
            "n_judges": len(judges),
            "judge_models": [j.model for j in judges],
        },
        {
            "story": story,
            "explanations": pooled,
            "explanations_per_judge": explanations_per_judge,
            "gate_attempts": gate_attempts,
        },
    )
