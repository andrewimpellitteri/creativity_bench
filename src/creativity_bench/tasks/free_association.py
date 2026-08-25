"""Free association task.

Gwern, "Free Association" (https://gwern.net/creative-benchmark#free-association):
"prompt models to 'freely associate' lists of words or numbers---'just say
whatever word next comes to mind'. Score the overall unique number of words,
and time to first repetition."

Scoring fidelity notes:
- The PRIMARY scores are the raw unique-word count and the time to first
  repetition, exactly as specified. The unique-word count is deliberately NOT
  normalized by the number of LLM calls: Gwern warns that "Trying to normalize
  the unique word count by number of LLM calls is probably a bad idea, because
  users can make many calls if that is worthwhile, and we don't want to
  conflate calls & total vocabulary".
- The sanctioned normalization is the unseen-species treatment: "one can try to
  treat it as an unseen species problem, and return the estimated total." We
  report the Chao1 estimated total vocabulary alongside the raw count.

The TaskResult.score slot must lie in [0, 1] for the composite (see base.py),
so the headline score here is the time-to-first-repetition fraction
(first_repeat_index / n_words, 1.0 if no repetition within the window). The
raw unique-word count and Chao1 estimate are carried in ``metrics`` verbatim.
"""

from __future__ import annotations

import re
from collections import Counter

from tqdm.auto import tqdm

from ..client import LLMClient
from .base import TaskResult

# Gwern: models are prompted to "freely associate ... 'just say whatever word
# next comes to mind'". We deliberately do NOT instruct the model to avoid
# repeating itself: the benchmark measures time to first repetition, and an
# anti-repetition instruction would suppress the very failure mode under test.
SYSTEM_PROMPT = (
    "Freely associate. Each turn, just say whatever word next comes to mind. "
    "Reply with a single English word only: no punctuation, no explanation."
)

WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")


def _extract_word(response: str) -> str | None:
    match = WORD_RE.search(response.lower())
    return match.group() if match else None


def _chao1(frequencies: Counter) -> float:
    """Chao1 estimator of total vocabulary size (unseen-species problem).

    Gwern sanctions this as the normalization for free association: treat each
    emitted word as a 'species' capture and return the estimated total."""
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
    # "time to first repetition": index of the first word already said.
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
    unique = len(frequencies)  # raw unique-word count: reported unnormalized

    # Headline score: time to first repetition (in [0, 1]); 1.0 = no repeat.
    score = 1.0 if first_repeat_index is None else first_repeat_index / n_words

    return TaskResult(
        name="free_association",
        score=score,
        metrics={
            "words_generated": total,
            "unique_words": unique,
            "first_repeat_index": first_repeat_index,
            "chao1_estimate": _chao1(frequencies) if total else 0.0,
        },
        details={"words": words},
    )
