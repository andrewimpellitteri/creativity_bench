"""Camel's back: how many rounds of stacked edits can a story absorb?

Gwern, "Camel's Back" (https://gwern.net/creative-benchmark#possible-tasks,
Iteration section): "a stress test in which we simply repeatedly ask an LLM to
edit a sample in randomized arbitrary ways (drawing on a big list of possible
ways to modify a sample, like 'make it rhyme' or 'add more cowbell' or
'rewrite as noir detective mystery' or 'translate into Japanese'), until the
sample stops changing (like Telephone) because the LLM has given up, or the
edit fails or the quality is low."

Implementation notes:
- Edit requests are drawn randomly each round from data.EDIT_REQUESTS, which
  includes Gwern's example edits verbatim.
- "The difficulty can be ramped up by asking for multiple edits
  simultaneously": each round applies a random 1-3 bundle of edits.
- The run stops on any of: the sample stopping changing (fixed point, checked
  here), the judge failing the round (edit not applied / incoherent / quality
  below OK, see judge.py). "The final sample can be additionally scored for
  quality": the last verdict's quality flag is surfaced as a metric.
"""

from __future__ import annotations

import random

from tqdm.auto import tqdm

from ..client import LLMClient
from ..judge import judge_edit
from .base import TaskResult, clamp01


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
    rng = rng or random.Random()
    story = client.generate(
        f"Write a short story (200-300 words) based on this premise:\n\n{seed_text}",
        temperature=0.8,
        max_tokens=2000,
    )

    rounds_survived = 0
    rounds: list[dict] = []
    stopped_changing = False
    final_quality_ok: bool | None = None

    for i in tqdm(range(max_edits), desc="Camel's back", leave=False):
        # Ramp difficulty: multiple simultaneous random edit requests.
        current_edits = rng.sample(edit_requests, rng.randint(1, 3))
        edit_prompt = (
            "Modify this story according to the instructions below. "
            "Return only the modified story.\n\n"
            f"STORY:\n{story}\n\nINSTRUCTIONS:\n" + "\n".join(f"- {edit}" for edit in current_edits)
        )
        modified = client.generate(edit_prompt, temperature=0.8, max_tokens=2000)

        verdict = judge_edit(judge_client, story, modified, current_edits)
        rounds.append(
            {
                "edits": current_edits,
                "coherent": verdict.coherent,
                "edits_applied": verdict.edits_applied,
                "quality_maintained": verdict.quality_maintained,
                "unchanged": modified == story,
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

    return TaskResult(
        name="camels_back",
        score=clamp01(rounds_survived / max_edits),
        metrics={
            "rounds_survived": rounds_survived,
            "max_rounds": max_edits,
            "stopped_changing": stopped_changing,
            "final_quality_ok": final_quality_ok,
        },
        details={"seed": seed_text, "rounds": rounds},
    )
