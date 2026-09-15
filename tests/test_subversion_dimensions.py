"""Offline tests for Subversion's explicit inversion dimension.

Design audit: "'Opposite' is ambiguous... Specify the dimension to invert and
facts to preserve; audit matched negatives and report sensitivity and
specificity separately."
"""

from __future__ import annotations

import re

import pytest
from conftest import FakeClient

from creativity_bench.tasks.subversion import INVERSION_DIMENSIONS, subversion


def make_dimensioned_writer():
    counter = {"n": 0}

    def responder(messages):
        prompt = messages[-1]["content"]
        if "opposite" in prompt:
            return f"subverted-{counter['n']}"
        counter["n"] += 1
        return f"original-{counter['n']}"

    return FakeClient(responder)


def make_dimension_judge():
    def responder(messages):
        prompt = messages[-1]["content"]
        story = re.search(r"original-(\d+)", prompt)
        sub = re.search(r"subverted-(\d+)", prompt)
        verdict = story.group(1) == sub.group(1)
        return f'{{"opposite": {"true" if verdict else "false"}}}'

    return FakeClient(responder)


def test_subversion_prompt_names_the_dimension_and_preserved_facts():
    prompts: list[str] = []

    def responder(messages):
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        if "opposite" in prompt:
            return "sub"
        return "original"

    subversion(
        FakeClient(responder),
        FakeClient(lambda _: '{"opposite": true}'),
        premises=["premise"],
        runs=3,
    )
    subversion_prompts = [p for p in prompts if "opposite" in p]
    assert len(subversion_prompts) == 3
    for prompt, dimension in zip(subversion_prompts, INVERSION_DIMENSIONS, strict=True):
        assert dimension in prompt
        assert "PRESERVING" in prompt


def test_subversion_dimensions_rotate_deterministically():
    result = subversion(
        make_dimensioned_writer(), make_dimension_judge(), premises=["p"], runs=4
    )
    dims = [pair["dimension"] for pair in result.details["pairs"] if pair["within"]]
    assert dims == [INVERSION_DIMENSIONS[i % len(INVERSION_DIMENSIONS)] for i in range(4)]


def test_subversion_reports_sensitivity_and_specificity():
    result = subversion(
        make_dimensioned_writer(), make_dimension_judge(), premises=["a", "b"], runs=2
    )
    assert result.metrics["sensitivity"] == pytest.approx(1.0)
    assert result.metrics["specificity"] == pytest.approx(1.0)
    assert result.metrics["within_opposite_rate"] == pytest.approx(result.metrics["sensitivity"])
