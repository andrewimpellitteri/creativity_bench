import numpy as np
import pytest

from creativity_bench.metrics import (
    cosine_similarity,
    damerau_levenshtein_similarity,
    lexical_similarity,
    pairwise_cosine_distances,
    rouge_l_f1,
)


def test_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0


def test_pairwise_cosine_distances():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    distances = pairwise_cosine_distances(embeddings)
    assert distances == pytest.approx([1.0, 0.0, 1.0])


def test_rouge_l_identical():
    assert rouge_l_f1("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_rouge_l_disjoint():
    assert rouge_l_f1("alpha beta", "gamma delta") == 0.0


def test_rouge_l_partial():
    # LCS("the cat sat down", "the dog sat down") = 3; P = R = 3/4
    assert rouge_l_f1("the cat sat down", "the dog sat down") == pytest.approx(0.75)


def test_rouge_l_empty():
    assert rouge_l_f1("", "anything") == 0.0


def test_damerau_identical():
    assert damerau_levenshtein_similarity("abc", "abc") == pytest.approx(1.0)


def test_damerau_transposition_counts_once():
    # One transposition in a 4-char string -> distance 1 -> similarity 0.75
    assert damerau_levenshtein_similarity("abcd", "abdc") == pytest.approx(0.75)


def test_damerau_empty_strings():
    assert damerau_levenshtein_similarity("", "") == 1.0
    assert damerau_levenshtein_similarity("a", "") == 0.0


def test_lexical_similarity_bounds():
    assert lexical_similarity("same text here", "same text here") == pytest.approx(1.0)
    assert 0.0 <= lexical_similarity("aaa bbb", "zzz qqq xyz") <= 1.0
