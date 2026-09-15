import random

import numpy as np
import pytest
from conftest import FakeClient, FakeEmbedder

from creativity_bench.tasks.diversity import _effective_rank, dont_repeat_yourself


def test_prompt_variation_alone_cannot_earn_score():
    result = dont_repeat_yourself(
        FakeClient(lambda m: m[-1]["content"]),
        FakeEmbedder(),
        samples=4,
        repeats_per_prompt=3,
        rng=random.Random(2),
    )
    assert result.score == pytest.approx(0, abs=1e-15)
    assert result.metrics["between_prompt_mean_distance"] > 0
    assert result.metrics["mean_pairwise_distance"] > 0
    assert result.metrics["stories_generated"] == 12
    stories = result.details["stories"]
    assert len({s["prompt"] for s in stories}) == 4
    for prompt_id in range(4):
        group = [s for s in stories if s["prompt_id"] == prompt_id]
        assert [s["repeat"] for s in group] == [0, 1, 2]
        assert len({s["text"] for s in group}) == 1


def test_orthogonal_cloud_effective_rank_is_centered_dimension():
    assert _effective_rank(np.eye(4)) == pytest.approx(3)
    assert _effective_rank(np.repeat([[1.0, 2.0, 3.0]], 9, axis=0)) == 0
    assert _effective_rank(np.array([[1.0, 0.0], [-1.0, 0.0]])) == pytest.approx(1)
    assert _effective_rank(np.zeros((3, 4))) == 0


@pytest.mark.parametrize(
    "options", [dict(repeats_per_prompt=1), dict(samples=1000), dict(template="identical prompt")]
)
def test_bad_design_fails_before_generation(options):
    client = FakeClient(lambda _: "unused")
    with pytest.raises(ValueError):
        dont_repeat_yourself(client, FakeEmbedder(), **options)
    assert not client.calls
