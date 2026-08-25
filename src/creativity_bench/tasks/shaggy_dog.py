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

import random
import re

from ..client import LLMClient
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


def shaggy_dog(
    client: LLMClient,
    judge_client: LLMClient,
    *,
    k: int = 3,
    rng: random.Random | None = None,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    """Run the shaggy dog contest; higher score = less tidy interpretation."""
    if k < 1:
        raise ValueError("Need at least 1 judge sample")
    rng = rng or random.Random()

    story = client.generate(SHAGGY_DOG_PROMPT, temperature=0.8, max_tokens=2000)

    if EXPLICIT_MORAL_RE.search(story):
        # Automatic failure: the storyteller stated the moral itself instead
        # of leaving the story pointless.
        if verbose:
            print("  explicit stated moral/punchline detected: automatic failure")
        return TaskResult(
            name="shaggy_dog",
            score=0.0,
            metrics={"k": k, "explicit_moral": True},
            details={"story": story, "explanations": []},
        )

    prompt = PUNCHLINE_PROMPT.format(story=story)
    explanations: list[str] = []
    for i in range(k):
        # Seeded sampling: jitter the judge's temperature from the caller's
        # rng so repeated samples genuinely vary while runs stay reproducible.
        response = judge_client.generate(
            prompt,
            temperature=round(0.7 + 0.2 * rng.random(), 3),
            max_tokens=500,
        )
        explanations.append(response)
        if verbose:
            print(f"  explanation {i + 1}: {response[:60]!r}...")

    if len(explanations) < 2:
        # K=1 degenerate case (see module docstring): no pairwise comparison
        # exists, so no tidy interpretation can be evidenced.
        return TaskResult(
            name="shaggy_dog",
            score=1.0,
            metrics={"k": k, "explicit_moral": False, "degenerate": True},
            details={"story": story, "explanations": explanations},
        )

    token_sets = [_content_tokens(e) for e in explanations]
    similarities = [
        _jaccard(token_sets[i], token_sets[j])
        for i in range(len(token_sets))
        for j in range(i + 1, len(token_sets))
    ]
    mean_agreement = sum(similarities) / len(similarities)
    # Inverted agreement: judges disagreeing => no tidy interpretation => high
    # score ("the more similar the explanations are, the worse the score").
    score = clamp01(1.0 - mean_agreement)

    return TaskResult(
        name="shaggy_dog",
        score=score,
        metrics={
            "k": k,
            "mean_pairwise_agreement": mean_agreement,
            "explicit_moral": False,
            "degenerate": False,
        },
        details={"story": story, "explanations": explanations},
    )
