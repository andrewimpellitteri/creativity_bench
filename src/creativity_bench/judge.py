"""LLM judge that returns structured boolean verdicts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .client import LLMClient

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

EDIT_JUDGE_PROMPT = """\
You are evaluating an edit made to a short story.

ORIGINAL STORY:
{original}

MODIFIED STORY:
{modified}

REQUESTED EDITS:
{edits}

Answer strictly as a JSON object with these three boolean fields and nothing else:
{{"coherent": <true if the modified story is still coherent and logical>,
 "edits_applied": <true if every requested edit was applied>,
 "quality_maintained": <true if the writing quality is at least as good as the original>}}
"""


@dataclass
class EditVerdict:
    coherent: bool
    edits_applied: bool
    quality_maintained: bool

    @property
    def passed(self) -> bool:
        return self.coherent and self.edits_applied


def _parse_verdict(text: str) -> EditVerdict:
    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object in judge response: {text!r}")
    payload = json.loads(match.group())
    return EditVerdict(
        coherent=bool(payload["coherent"]),
        edits_applied=bool(payload["edits_applied"]),
        quality_maintained=bool(payload["quality_maintained"]),
    )


def judge_edit(
    judge_client: LLMClient,
    original: str,
    modified: str,
    edits: list[str],
) -> EditVerdict:
    prompt = EDIT_JUDGE_PROMPT.format(
        original=original,
        modified=modified,
        edits="\n".join(f"- {edit}" for edit in edits),
    )
    last_error: Exception | None = None
    for _ in range(2):
        response = judge_client.generate(prompt, temperature=0.0, max_tokens=2000)
        try:
            return _parse_verdict(response)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_error = e
    raise RuntimeError(f"Judge returned unparseable verdicts twice: {last_error}")
