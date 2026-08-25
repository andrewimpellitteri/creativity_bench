"""Free association: how long can the model produce words without repeating itself?

The model sees its full word history (as chat turns), so a repetition is a
genuine failure of novelty rather than an artifact of a memoryless prompt.
Score is the fraction of unique words. A Chao1 richness estimate is reported
as a rough proxy for the size of the model's accessible vocabulary.
"""

from __future__ import annotations

import re
from collections import Counter

from tqdm.auto import tqdm

from ..client import LLMClient
from .base import TaskResult, clamp01

SYSTEM_PROMPT = (
    "You are playing a free-association game. Each turn, reply with exactly one "
    "English word that comes to mind next. Never repeat a word you have already "
    "said. Reply with the single word only: no punctuation, no explanation."
)

WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")


def _extract_word(response: str) -> str | None:
    match = WORD_RE.search(response.lower())
    return match.group() if match else None


def _chao1(frequencies: Counter) -> float:
    observed = len(frequencies)
    singletons = sum(1 for count in frequencies.values() if count == 1)
    doubletons = sum(1 for count in frequencies.values() if count == 2)
    if doubletons == 0:
        return observed + singletons * (singletons - 1) / 2
    return observed + singletons**2 / (2 * doubletons)


def free_association(
    client: LLMClient,
    *,
    n_words: int = 40,
    verbose: bool = False,
    **_: object,
) -> TaskResult:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Begin. Say your first word."},
    ]
    words: list[str] = []
    frequencies: Counter = Counter()
    first_repeat_index: int | None = None

    for i in tqdm(range(n_words), desc="Free association", leave=False):
        response = client.chat(messages, temperature=1.0, max_tokens=2000)
        word = _extract_word(response)
        if word is None:
            continue
        if word in frequencies and first_repeat_index is None:
            first_repeat_index = i
        frequencies[word] += 1
        words.append(word)
        if verbose:
            print(f"  word {i + 1}: {word}")
        messages.append({"role": "assistant", "content": word})
        messages.append({"role": "user", "content": "Next word."})

    total = len(words)
    unique = len(frequencies)
    unique_ratio = unique / total if total else 0.0

    return TaskResult(
        name="free_association",
        score=clamp01(unique_ratio),
        metrics={
            "words_generated": total,
            "unique_words": unique,
            "unique_ratio": unique_ratio,
            "first_repeat_index": first_repeat_index,
            "chao1_estimate": _chao1(frequencies) if total else 0.0,
        },
        details={"words": words},
    )
