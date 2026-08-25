"""Text-similarity metrics used by the benchmark tasks.

Implemented directly (no external NLP deps): the strings compared here are
one-to-two-sentence summaries, so O(n*m) dynamic programming is plenty fast.

Gwern (https://gwern.net/creative-benchmark#possible-tasks, "Ranking/Distance
Metrics") names embeddings as a core primitive: "Embeddings provide direct
quantitative distance measurements between pairs of points." The cosine
helpers below implement that; the lexical helpers back the Telephone Game's
edit-distance fallback.
"""

from __future__ import annotations

import re

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Condensed upper-triangle cosine distances for a (n, d) embedding matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.where(norms == 0, 1.0, norms)
    similarity = normalized @ normalized.T
    i, j = np.triu_indices(len(embeddings), k=1)
    return 1.0 - similarity[i, j]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1: longest common subsequence over word tokens."""
    hyp, ref = _tokenize(hypothesis), _tokenize(reference)
    if not hyp or not ref:
        return 0.0
    # LCS length via DP over two rows
    previous = [0] * (len(ref) + 1)
    for h in hyp:
        current = [0] * (len(ref) + 1)
        for j, r in enumerate(ref, start=1):
            current[j] = previous[j - 1] + 1 if h == r else max(previous[j], current[j - 1])
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def damerau_levenshtein_similarity(a: str, b: str) -> float:
    """1 - normalized optimal-string-alignment distance over characters.

    Gwern's Telephone Game spec (https://gwern.net/creative-benchmark#possible-tasks)
    names edit distance as the sanctioned fallback when exact text match is
    too strict for fixed-point detection; this and lexical_similarity below
    serve that role."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    rows, cols = len(a) + 1, len(b) + 1
    dist = np.zeros((rows, cols), dtype=np.int32)
    dist[:, 0] = np.arange(rows)
    dist[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dist[i, j] = min(
                dist[i - 1, j] + 1,
                dist[i, j - 1] + 1,
                dist[i - 1, j - 1] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                dist[i, j] = min(dist[i, j], dist[i - 2, j - 2] + 1)
    return 1.0 - dist[-1, -1] / max(len(a), len(b))


def lexical_similarity(a: str, b: str) -> float:
    """Mean of ROUGE-L F1 and Damerau-Levenshtein similarity, in [0, 1]."""
    return (rouge_l_f1(a, b) + damerau_levenshtein_similarity(a, b)) / 2
