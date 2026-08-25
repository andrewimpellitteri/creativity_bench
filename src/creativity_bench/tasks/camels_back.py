"""Camel's back: how many rounds of stacked edits can a story absorb?

Each round applies 1-3 random edit requests; an LLM judge (ideally a fixed
model, separate from the one under test) checks that the story stayed coherent
and the edits were actually applied. Score is the fraction of rounds survived.
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

    for i in tqdm(range(max_edits), desc="Camel's back", leave=False):
        current_edits = rng.sample(edit_requests, rng.randint(1, 3))
        edit_prompt = (
            "Modify this story according to the instructions below. "
            "Return only the modified story.\n\n"
            f"STORY:\n{story}\n\nINSTRUCTIONS:\n"
            + "\n".join(f"- {edit}" for edit in current_edits)
        )
        modified = client.generate(edit_prompt, temperature=0.8, max_tokens=2000)

        verdict = judge_edit(judge_client, story, modified, current_edits)
        rounds.append(
            {
                "edits": current_edits,
                "coherent": verdict.coherent,
                "edits_applied": verdict.edits_applied,
                "quality_maintained": verdict.quality_maintained,
            }
        )
        if verbose:
            print(f"  round {i + 1}: edits={current_edits} passed={verdict.passed}")

        if not verdict.passed:
            break
        rounds_survived += 1
        story = modified

    return TaskResult(
        name="camels_back",
        score=clamp01(rounds_survived / max_edits),
        metrics={"rounds_survived": rounds_survived, "max_rounds": max_edits},
        details={"seed": seed_text, "rounds": rounds},
    )
