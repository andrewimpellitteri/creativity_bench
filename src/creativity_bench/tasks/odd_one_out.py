"""Odd one out: escape the anchor of same-category example items.

Gwern, "Odd One Out" (https://gwern.net/creative-benchmark#possible-tasks):
"present a list of example items sharing a theme; the model must generate the
maximally-different item that still qualifies as a member."

Implementation notes:
- Seed lists are themed inventories (dog breeds, jazz musicians, ...) of
  familiar items: exactly the anchor a mode-collapsed model gravitates to.
- # NOTE(gwern): scoring is "via embedding distance from the list items"
  ("document whether you use mean or min cosine distance and why"). We use
  the MINIMUM cosine distance from the candidate to any example, not the
  mean: the task demands distance from ALL examples, and a candidate that
  paraphrases even one anchor exhibits precisely the anchoring this task
  exists to detect -- a mean would let the remaining examples dilute that
  failure. The mean distance is reported as an auxiliary metric.
- Normalization: cosine distance spans [0, 2] (antiparallel vectors), so each
  item's minimum distance is halved and clamped into [0, 1]; higher = more
  different. Real embedding similarities rarely drop below 0, so scores
  concentrate in the upper half of the interval; comparisons are only valid
  within one fixed embedding space.
- # NOTE(gwern): the spec's optional gate -- check "that the item still
  qualifies as a member" with a judge, where "judge rejects -> that item
  scores 0" -- is implemented via ``judge_client`` returning a JSON verdict
  in the style of judge.py's EditVerdict parsing. Unparseable judge output
  counts as qualified (fail-open) and is recorded in the ``judge_unparseable``
  metric rather than crashing the run.
"""

from __future__ import annotations

import json
import random
import re

import numpy as np
from tqdm.auto import tqdm

from ..client import Embedder, LLMClient
from ..metrics import cosine_similarity
from .base import TaskResult, clamp01

SEED_LISTS: list[tuple[str, list[str]]] = [
    (
        "dog breeds",
        [
            "Labrador Retriever",
            "Poodle",
            "Beagle",
            "German Shepherd",
            "Golden Retriever",
            "Bulldog",
            "Rottweiler",
            "Dachshund",
        ],
    ),
    (
        "kitchen tools",
        [
            "whisk",
            "spatula",
            "ladle",
            "tongs",
            "colander",
            "grater",
            "peeler",
            "rolling pin",
        ],
    ),
    (
        "jazz musicians",
        [
            "Miles Davis",
            "John Coltrane",
            "Charlie Parker",
            "Thelonious Monk",
            "Duke Ellington",
            "Bill Evans",
            "Louis Armstrong",
            "Dave Brubeck",
        ],
    ),
    (
        "European capitals",
        ["Paris", "Berlin", "Madrid", "Rome", "Vienna", "Warsaw", "Athens", "Lisbon"],
    ),
    (
        "trees",
        ["oak", "maple", "pine", "birch", "willow", "elm", "spruce", "cedar"],
    ),
]

ITEM_PROMPT = """\
Here are example items from the category "{theme}":

{items}

Name the single item that is as different as possible from every example
above -- as unfamiliar, unusual, and distant from them in every respect as
you can make it -- while still unquestionably belonging to the category
"{theme}".

Answer with only the item itself, no explanation.
"""

QUALIFICATION_JUDGE_PROMPT = """\
You are evaluating an answer to a creativity task.

The task: given example items from the category "{theme}", name the single
item that is as different as possible from the examples while still
unquestionably belonging to that category.

EXAMPLE ITEMS:
{items}

CANDIDATE ANSWER:
{candidate}

Answer strictly as a JSON object with one boolean field and nothing else:
{{"qualifies": <true if the candidate still clearly belongs to the category "{theme}">}}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clean_item(text: str) -> str:
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    return first_line.strip().strip("\"'`").rstrip(".").strip()


def _parse_qualification(text: str) -> bool:
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object in judge response: {text!r}")
    return bool(json.loads(match.group())["qualifies"])


def _judge_qualifies(
    judge_client: LLMClient, theme: str, items: list[str], candidate: str
) -> tuple[bool, bool]:
    """Returns (qualified, unparseable); unparseable verdicts fail open."""
    prompt = QUALIFICATION_JUDGE_PROMPT.format(
        theme=theme,
        items="\n".join(f"- {item}" for item in items),
        candidate=candidate,
    )
    for _ in range(2):
        response = judge_client.generate(prompt, temperature=0.0, max_tokens=2000)
        try:
            return _parse_qualification(response), False
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return True, True


def odd_one_out(
    client: LLMClient,
    *,
    embedder: Embedder,
    judge_client: LLMClient | None = None,
    n_lists: int = 2,
    lists: list[tuple[str, list[str]]] | None = None,
    rng: random.Random | None = None,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    if n_lists < 1:
        raise ValueError("Need at least 1 seed list")
    rng = rng or random.Random()
    if lists is None:
        lists = rng.sample(SEED_LISTS, min(n_lists, len(SEED_LISTS)))
    if not lists:
        raise ValueError("Need at least 1 seed list")

    records: list[dict] = []
    item_scores: list[float] = []

    for theme, items in tqdm(lists, desc="Odd one out", leave=False):
        prompt = ITEM_PROMPT.format(theme=theme, items="\n".join(f"- {item}" for item in items))
        raw = client.generate(prompt, temperature=0.9, max_tokens=2000)
        candidate = _clean_item(raw)

        embeddings = embedder.embed([*items, candidate])
        distances = np.asarray(
            [1.0 - cosine_similarity(embeddings[-1], seed) for seed in embeddings[:-1]]
        )
        min_distance = float(np.min(distances))
        mean_distance = float(np.mean(distances))
        # Linear normalization: cosine distance spans [0, 2], divide by 2.
        item_score = clamp01(min_distance / 2)

        qualified, unparseable = True, False
        if judge_client is not None:
            qualified, unparseable = _judge_qualifies(judge_client, theme, items, candidate)
        if not qualified:
            item_score = 0.0
        if verbose:
            status = "qualified" if qualified else "rejected"
            print(
                f"  {theme}: {candidate!r} min_distance={min_distance:.3f} "
                f"score={item_score:.3f} ({status})"
            )

        item_scores.append(item_score)
        records.append(
            {
                "theme": theme,
                "items": items,
                "candidate": candidate,
                "min_distance": min_distance,
                "mean_distance": mean_distance,
                "qualified": qualified,
                "judge_unparseable": unparseable,
                "score": item_score,
            }
        )

    return TaskResult(
        name="odd_one_out",
        score=clamp01(float(np.mean(item_scores))),
        metrics={
            "mean_min_distance": float(np.mean([record["min_distance"] for record in records])),
            "mean_distance": float(np.mean([record["mean_distance"] for record in records])),
            "n_lists": len(records),
            "judge_used": judge_client is not None,
            "judge_rejected": sum(not record["qualified"] for record in records),
            "judge_unparseable": sum(record["judge_unparseable"] for record in records),
        },
        details={"lists": records},
    )
