"""Subversion: write "the opposite" of a generated story.

Gwern, "Subversion" (https://gwern.net/creative-benchmark#subversion,
Difference & Negation section): "after a seed story prompt, ask an LLM to
write 'the opposite' of the generated story, which subverts the first one.
For all the possible pairs, have an LLM judge classify by whether the stories
are 'opposite'."

Implementation notes:
- Each of the n runs per premise generates a fresh story from the seed premise
  (data.STORY_PROMPTS premises / data.SAMPLE_STORIES seed stories) and then
  its subversion: a new story which inverts the original's genre, tone,
  outcome, and themes while remaining coherent itself.
- With n runs per premise there are n^2 story/subversion pairings ("all the
  possible pairs"). The diagonal pairs (story_i vs its own subversion_i) are
  *within-pairs* and should be classified as opposite; the off-diagonal pairs
  (story_i vs subversion_j, i != j) are *cross-pairs* -- a story against
  someone else's subversion -- and should NOT be.
- The cross-pairs act as a discrimination check in the same spirit as the
  within-vs-cross contrast elsewhere in this benchmark: they catch the two
  degenerate outcomes. A model whose "opposites" are just trivially different
  stories fails the within-pairs; a lazy judge that answers "opposite" to
  everything is exposed by the cross-pair false positives.

Score (in [0, 1], higher = better):

    score = clamp01(tpr - fpr)

where tpr is the fraction of within-pairs judged opposite (sensitivity) and
fpr the fraction of cross-pairs judged opposite (false-positive rate). This is
Youden's J statistic: the raw within-pair hit rate alone would reward judges
that always answer "opposite", so we subtract the false-positive rate as a
floor. A perfectly discriminative writer/judge scores 1.0; a judge that flags
every pairing scores 0.0; undiscriminated behavior hovers near 0. With a
single run per premise there are no cross-pairs, so the score falls back to
the raw within-pair hit rate (documented limitation).
"""

from __future__ import annotations

import itertools
import json
import re

from tqdm.auto import tqdm

from ..client import LLMClient
from .base import TaskResult, clamp01

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

SUBVERSION_PROMPT = """\
Below is a short story. Write a NEW short story that is "the opposite" of it: \
a story which subverts the first one by inverting its genre, tone, themes, \
and especially its outcome, while still being a coherent story in its own \
right. Return only the new story.

STORY:
{story}
"""

SUBVERSION_JUDGE_PROMPT = """\
You are evaluating whether two short stories are opposites of each other: not \
merely different, but one subverting and inverting the other (its genre, \
tone, themes, or outcome turned around).

STORY A:
{story_a}

STORY B:
{story_b}

Answer strictly as a JSON object with this boolean field and nothing else:
{{"opposite": <true if STORY B reads as "the opposite" of STORY A>}}
"""


def _parse_opposite(text: str) -> bool:
    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object in judge response: {text!r}")
    return bool(json.loads(match.group())["opposite"])


def judge_opposite(
    judge_client: LLMClient,
    story_a: str,
    story_b: str,
) -> bool:
    last_error: Exception | None = None
    for _ in range(2):
        response = judge_client.generate(
            SUBVERSION_JUDGE_PROMPT.format(story_a=story_a, story_b=story_b),
            temperature=0.0,
            max_tokens=2000,
        )
        try:
            return _parse_opposite(response)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_error = e
    raise RuntimeError(f"Judge returned unparseable verdicts twice: {last_error}")


def subversion(
    client: LLMClient,
    judge_client: LLMClient,
    *,
    premises: list[str],
    runs: int = 3,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    if runs < 1:
        raise ValueError("runs must be >= 1")

    stories: list[list[str]] = []
    subverted: list[list[str]] = []
    for premise in tqdm(premises, desc="Subversion", leave=False):
        premise_stories = []
        premise_subversions = []
        for _ in range(runs):
            # Fresh story from the seed premise each run...
            story = client.generate(
                f"Write a short story (200-300 words) based on this premise:\n\n{premise}",
                temperature=0.8,
                max_tokens=2000,
            )
            # ...then its subversion ("write 'the opposite' ... which
            # subverts the first one").
            opposite = client.generate(
                SUBVERSION_PROMPT.format(story=story),
                temperature=0.8,
                max_tokens=2000,
            )
            premise_stories.append(story)
            premise_subversions.append(opposite)
        stories.append(premise_stories)
        subverted.append(premise_subversions)

    # Judge all the possible pairs: n^2 story/subversion combos per premise.
    # i == j are within-pairs (expected opposite); i != j are cross-pairs
    # (expected NOT opposite -- the false-positive discrimination check).
    within_hits = 0
    within_total = 0
    cross_false_positives = 0
    cross_total = 0
    pairs: list[dict] = []
    for p_idx, (premise_stories, premise_subversions) in enumerate(
        zip(stories, subverted, strict=True)
    ):
        for i, j in itertools.product(range(runs), repeat=2):
            is_within = i == j
            opposite = judge_opposite(judge_client, premise_stories[i], premise_subversions[j])
            pairs.append(
                {
                    "premise": p_idx,
                    "i": i,
                    "j": j,
                    "within": is_within,
                    "opposite": opposite,
                }
            )
            if is_within:
                within_total += 1
                within_hits += opposite
            else:
                cross_total += 1
                cross_false_positives += opposite
        if verbose:
            print(f"  premise {p_idx}: judged {runs * runs} pairs")

    tpr = within_hits / within_total if within_total else 0.0
    # Youden's J: sensitivity minus false-positive rate (see module docstring).
    # Without cross-pairs (runs == 1) fall back to the raw within-pair rate.
    score = clamp01(tpr - (cross_false_positives / cross_total)) if cross_total else clamp01(tpr)

    return TaskResult(
        name="subversion",
        score=score,
        metrics={
            "within_opposite_rate": tpr,
            "cross_opposite_rate": cross_false_positives / cross_total if cross_total else None,
            "within_pairs": within_total,
            "cross_pairs": cross_total,
            "premises": len(premises),
            "runs_per_premise": runs,
        },
        details={"pairs": pairs},
    )
