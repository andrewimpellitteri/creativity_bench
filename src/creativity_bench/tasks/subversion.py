"""Subversion: write "the opposite" of a generated story.

Gwern, "Subversion" (https://gwern.net/creative-benchmark#subversion,
Difference & Negation section): "after a seed story prompt, ask an LLM to
write 'the opposite' of the generated story, which subverts the first one.
For all the possible pairs, have an LLM judge classify by whether the stories
are 'opposite'."

Implementation notes:
- Each of the n runs per premise generates a fresh story from the seed premise
  (data.STORY_PROMPTS premises / data.SAMPLE_STORIES seed stories) and then
  its subversion. "Opposite" is made EXPLICIT: run i must invert exactly one
  named dimension (INVERSION_DIMENSIONS[i % len(...)], cycling outcome, tone,
  genre in a fixed order shared by every model) while PRESERVING the
  original's setting, characters and stated facts. The dimension is named in
  both the writer prompt and the judge prompt, and recorded on every judged
  pair.
- With n runs per premise there are n^2 story/subversion pairings ("all the
  possible pairs"). The diagonal pairs (story_i vs its own subversion_i) are
  *within-pairs* and should be classified as opposite; the off-diagonal pairs
  (story_i vs subversion_j, i != j) are *cross-pairs* -- a story against
  someone else's subversion -- and should NOT be.
- The cross-pairs are the MATCHED NEGATIVES and act as a discrimination check
  in the same spirit as the within-vs-cross contrast elsewhere in this
  benchmark: they catch the two degenerate outcomes. A model whose
  "opposites" are just trivially different stories fails the within-pairs; a
  lazy judge that answers "opposite" to everything is exposed by the
  cross-pair false positives. Residual ambiguity is audited, not hidden:
  cross-pairs sharing the same declared dimension may legitimately read as
  opposites, which is exactly why sensitivity and specificity are reported
  separately below.

Score (in [0, 1], higher = better):

    score = clamp01(tpr - fpr)

where tpr is the fraction of within-pairs judged opposite (sensitivity) and
fpr the fraction of cross-pairs judged opposite (false-positive rate; 1 - fpr
is specificity). This is Youden's J statistic: the raw within-pair hit rate
alone would reward judges that always answer "opposite", so we subtract the
false-positive rate as a floor. A perfectly discriminative writer/judge scores
1.0; a judge that flags every pairing scores 0.0; undiscriminated behavior
hovers near 0. With a single run per premise there are no cross-pairs, so the
score falls back to the raw within-pair hit rate (documented limitation).
"""

from __future__ import annotations

import itertools
import json
import re

from tqdm.auto import tqdm

from ..client import LLMClient
from .base import TaskResult, clamp01

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# The named inversion dimensions, cycled in a fixed order so every model and
# run faces the same assignment for a given run index.
INVERSION_DIMENSIONS: tuple[str, ...] = ("outcome", "tone", "genre")

SUBVERSION_PROMPT = """\
Below is a short story. Write a NEW short story that is "the opposite" of it \
in exactly one dimension: {dimension}. Subvert the original by turning its \
{dimension} around while PRESERVING its setting, characters, and stated \
facts, and while remaining a coherent story in its own right. Do not invert \
anything other than the named dimension. Return only the new story.

STORY:
{story}
"""

SUBVERSION_JUDGE_PROMPT = """\
You are evaluating whether STORY B subverts STORY A by inverting exactly one \
of its dimensions: the {dimension}. Not merely different: the named \
dimension must be turned around, while A's setting, characters and stated \
facts are preserved.

STORY A:
{story_a}

STORY B:
{story_b}

Answer strictly as a JSON object with this boolean field and nothing else:
{{"opposite": <true if STORY B inverts STORY A's {dimension} while \
preserving the rest>}}
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
    dimension: str = INVERSION_DIMENSIONS[0],
) -> bool:
    last_error: Exception | None = None
    for _ in range(2):
        response = judge_client.generate(
            SUBVERSION_JUDGE_PROMPT.format(story_a=story_a, story_b=story_b, dimension=dimension),
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
    dimensions: list[list[str]] = []
    for premise in tqdm(premises, desc="Subversion", leave=False):
        premise_stories = []
        premise_subversions = []
        premise_dimensions = []
        for run_index in range(runs):
            # Fresh story from the seed premise each run...
            story = client.generate(
                f"Write a short story (200-300 words) based on this premise:\n\n{premise}",
                temperature=0.8,
                max_tokens=2000,
            )
            # ...then its subversion, inverting exactly the named dimension
            # (the assignment cycles in a fixed order shared by all models).
            dimension = INVERSION_DIMENSIONS[run_index % len(INVERSION_DIMENSIONS)]
            opposite = client.generate(
                SUBVERSION_PROMPT.format(story=story, dimension=dimension),
                temperature=0.8,
                max_tokens=2000,
            )
            premise_stories.append(story)
            premise_subversions.append(opposite)
            premise_dimensions.append(dimension)
        stories.append(premise_stories)
        subverted.append(premise_subversions)
        dimensions.append(premise_dimensions)

    # Judge all the possible pairs: n^2 story/subversion combos per premise.
    # i == j are within-pairs (expected opposite); i != j are cross-pairs
    # (expected NOT opposite -- the matched-negative discrimination check).
    # Each pair is judged on the dimension its subversion was written to
    # invert; the dimension is recorded for auditing ambiguous negatives.
    within_hits = 0
    within_total = 0
    cross_false_positives = 0
    cross_total = 0
    pairs: list[dict] = []
    for p_idx, (premise_stories, premise_subversions, premise_dimensions) in enumerate(
        zip(stories, subverted, dimensions, strict=True)
    ):
        for i, j in itertools.product(range(runs), repeat=2):
            is_within = i == j
            dimension = premise_dimensions[j]
            opposite = judge_opposite(
                judge_client, premise_stories[i], premise_subversions[j], dimension
            )
            pairs.append(
                {
                    "premise": p_idx,
                    "i": i,
                    "j": j,
                    "within": is_within,
                    "dimension": dimension,
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
            "sensitivity": tpr,
            "specificity": (1.0 - cross_false_positives / cross_total) if cross_total else None,
            "within_pairs": within_total,
            "cross_pairs": cross_total,
            "premises": len(premises),
            "runs_per_premise": runs,
            "inversion_dimensions": list(INVERSION_DIMENSIONS),
        },
        details={"pairs": pairs},
    )
