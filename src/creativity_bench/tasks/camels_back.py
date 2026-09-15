"""Camel's back: how many rounds of stacked edits can a story absorb?

Gwern, "Camel's Back" (https://gwern.net/creative-benchmark#possible-tasks,
Iteration section): "a stress test in which we simply repeatedly ask an LLM to
edit a sample in randomized arbitrary ways (drawing on a big list of possible
ways to modify a sample, like 'make it rhyme' or 'add more cowbell' or
'rewrite as noir detective mystery' or 'translate into Japanese'), until the
sample stops changing (like Telephone) because the LLM has given up, or the
edit fails or the quality is low."

Implementation notes:
- The edit schedule is PRECOMPUTED before the run starts from the caller's
  seeded rng, so every model evaluated with the same seed faces the identical
  sequence of edit bundles (previously bundles were drawn per round while the
  run was underway, and random bundles could contradict one another, e.g.
  "make it more concise" alongside "add more descriptive details").
- Bundle compatibility: contradictory edit pairs (see CONFLICTING_EDITS) are
  never placed in the same round. If no compatible bundle of the wanted size
  can be drawn after repeated attempts, the round falls back to a single edit,
  which cannot self-conflict.
- Constraint semantics (the design audit's persist-or-replace ambiguity):
  constraints are REPLACED each round. The story carries forward, but only the
  current bundle is enforced and judged; earlier rounds' instructions are
  explicitly released in the edit prompt. Persistent-constraint variants remain
  future work.
- The starting story is still generated per run from seed_text; sharing fixed
  starting stories across models is not implemented here.
- "The difficulty can be ramped up by asking for multiple edits
  simultaneously": each round applies a 1-3 bundle of edits from the schedule.
- The run stops on any of: the sample stopping changing (fixed point, checked
  here), the judge failing the round (edit not applied / incoherent / quality
  below OK, see judge.py). "The final sample can be additionally scored for
  quality": the last verdict's quality flag is surfaced as a metric. Failed
  rounds stay in details["rounds"] with a "failed" flag for auditing.
"""

from __future__ import annotations

import itertools
import random

from tqdm.auto import tqdm

from ..client import LLMClient
from ..judge import judge_edit
from .base import TaskResult, clamp01

# Unordered edit-pattern pairs that contradict one another within a single
# round: compression versus expansion, and language/style demands that cannot
# coexist. Matching is case-insensitive on either substring order, so the
# table also covers custom edit lists built from the same vocabulary.
CONFLICTING_EDITS: tuple[tuple[str, str], ...] = (
    ("make it more concise", "add"),
    ("make it more concise", "more descriptive details"),
    ("make it more concise", "more emotional depth"),
    ("translate it into japanese", "make it rhyme"),
    ("change the tone to be more serious", "make it more humorous"),
)


def _edits_conflict(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return any(
        (pattern_a in a and pattern_b in b) or (pattern_b in a and pattern_a in b)
        for pattern_a, pattern_b in CONFLICTING_EDITS
    )


def build_edit_schedule(
    edit_requests: list[str],
    rounds: int,
    rng: random.Random,
    *,
    min_size: int = 1,
    max_size: int = 3,
    draw_attempts: int = 100,
) -> list[list[str]]:
    """Precompute the full per-round edit bundles before any model call.

    The schedule depends only on the seeded rng and the edit pool, so every
    model sees the same bundles under the same seed ("shared ... edit
    schedules" from the design audit). Each bundle avoids the contradictory
    pairs in CONFLICTING_EDITS; a round that cannot draw a compatible bundle
    of the wanted size falls back to a single edit (never self-conflicting).
    """
    if rounds < 1 or not edit_requests:
        raise ValueError("Need positive rounds and at least one edit request")
    max_size = max(1, min(max_size, len(edit_requests)))
    min_size = max(1, min(min_size, max_size))
    schedule: list[list[str]] = []
    for _ in range(rounds):
        bundle: list[str] | None = None
        for _ in range(draw_attempts):
            candidate = rng.sample(edit_requests, rng.randint(min_size, max_size))
            if all(not _edits_conflict(a, b) for a, b in itertools.combinations(candidate, 2)):
                bundle = candidate
                break
        schedule.append(bundle if bundle is not None else [rng.choice(edit_requests)])
    return schedule


def camels_back(
    client: LLMClient,
    judge_client: LLMClient,
    *,
    seed_text: str,
    edit_requests: list[str],
    max_edits: int = 8,
    rng: random.Random | None = None,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    if max_edits < 1 or not edit_requests:
        raise ValueError("Need positive max_edits and at least one edit request")
    rng = rng or random.Random()
    # The whole schedule exists before the first model call: the sequence does
    # not depend on how earlier rounds went, and identical seeds give every
    # model identical edit sequences.
    schedule = build_edit_schedule(edit_requests, max_edits, rng)
    story = client.generate(
        f"Write a short story (200-300 words) based on this premise:\n\n{seed_text}",
        temperature=0.8,
        max_tokens=2000,
    )

    rounds_survived = 0
    rounds: list[dict] = []
    stopped_changing = False
    final_quality_ok: bool | None = None

    for i, current_edits in enumerate(tqdm(schedule, desc="Camel's back", leave=False)):
        edit_prompt = (
            "Modify this story according to the instructions below. "
            "These instructions REPLACE all earlier instructions: apply them to "
            "the story as it stands now; edits requested in earlier rounds are "
            "no longer in force and only this round's instructions are judged. "
            "Return only the modified story.\n\n"
            f"STORY:\n{story}\n\nINSTRUCTIONS:\n" + "\n".join(f"- {edit}" for edit in current_edits)
        )
        modified = client.generate(edit_prompt, temperature=0.8, max_tokens=2000)

        verdict = judge_edit(judge_client, story, modified, current_edits)
        final_quality_ok = verdict.quality_maintained
        rounds.append(
            {
                "round_index": i,
                "original": story,
                "modified": modified,
                "edits": current_edits,
                "coherent": verdict.coherent,
                "edits_applied": verdict.edits_applied,
                "quality_maintained": verdict.quality_maintained,
                "unchanged": modified == story,
                "failed": not verdict.passed,
            }
        )
        if verbose:
            print(f"  round {i + 1}: edits={current_edits} passed={verdict.passed}")

        if modified == story:
            # Fixed point: the sample stopped changing because the LLM has
            # given up ("until the sample stops changing (like Telephone)").
            stopped_changing = True
            break
        if not verdict.passed:
            break

        rounds_survived += 1
        story = modified
        final_quality_ok = verdict.quality_maintained

    failed_round = next((r["round_index"] for r in rounds if r["failed"]), None)
    return TaskResult(
        name="camels_back",
        score=clamp01(rounds_survived / max_edits),
        metrics={
            "rounds_survived": rounds_survived,
            "max_rounds": max_edits,
            "right_censored": rounds_survived == max_edits,
            "stopped_changing": stopped_changing,
            "final_quality_ok": final_quality_ok,
            "failed_round_index": failed_round,
            "constraints": "replaced_each_round",
        },
        details={"seed": seed_text, "schedule": schedule, "rounds": rounds},
    )
