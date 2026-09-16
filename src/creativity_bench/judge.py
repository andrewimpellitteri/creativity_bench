"""LLM judge that returns structured boolean verdicts.

Gwern, "Camel's Back" (https://gwern.net/creative-benchmark#possible-tasks,
Iteration section): each round can be checked "by calling a judge LLM to ask
questions like, 'is the quality at least OK?' and 'here is the edit request:
"add more cowbell"; and the before/after; was the edit correct?'"

The EDIT_JUDGE_PROMPT below mirrors both questions: it shows the judge the
before/after plus the edit request(s), and asks whether quality stayed at
least OK ("quality_maintained") and whether the requested edits were actually
applied ("edits_applied").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .client import LLMClient

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """Return the last balanced JSON object in a judge response.

    A greedy ``{.*}`` span is wrong on the two things judges actually do:
    restate the schema before answering ("Recall the schema {...}. My answer:
    {...}") makes the span cover both objects and fail to parse, and a trailing
    note after the answer does the same. Scanning for balanced braces and taking
    the LAST parseable object handles both, because the answer comes after the
    preamble. Quoted braces and escapes are respected so a brace inside story
    text cannot end the scan early.

    A parse failure here means an unresolved judgment, which costs the whole run
    its completeness flag, so being generous about surrounding prose is a
    correctness matter, not a convenience.
    """
    candidates: list[dict] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except ValueError:
                    continue
                if isinstance(value, dict):
                    candidates.append(value)
    if not candidates:
        raise ValueError(f"No JSON object in judge response: {text!r}")
    return candidates[-1]


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
 "edits_applied": <true if every requested edit was applied correctly>,
 "quality_maintained": <true if the writing quality is at least OK>}}
"""


@dataclass
class EditVerdict:
    coherent: bool
    edits_applied: bool
    quality_maintained: bool

    @property
    def passed(self) -> bool:
        # Gwern's stop condition: the run ends when "the edit fails or the
        # quality is low", so a low-quality result fails the round even if
        # the letter of the request was carried out.
        return self.coherent and self.edits_applied and self.quality_maintained


def _parse_verdict(text: str) -> EditVerdict:
    payload = extract_json_object(text)
    fields = ("coherent", "edits_applied", "quality_maintained")
    if not isinstance(payload, dict) or any(type(payload[k]) is not bool for k in fields):
        raise ValueError("Judge verdict fields must be JSON booleans")
    return EditVerdict(**{k: payload[k] for k in fields})


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
