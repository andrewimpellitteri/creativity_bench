import json

import pytest
from conftest import FakeClient

from creativity_bench.tasks.same_but_different import evaluate_candidate, same_but_different


def verdict(**updates):
    result = dict(
        premise_adherent=True,
        comprehensible=True,
        plot_distinct=True,
        evidence="The conflict and resolution satisfy the rubric.",
        summary="A courier sacrifices her cargo to save a stranded rival.",
    )
    result.update(updates)
    return json.dumps(result)


def test_fixed_budget_and_gates():
    stories = iter(["original", "unrelated", "nonsense", "renamed plot", "original", "new"])
    judgments = iter(
        [
            verdict(),
            verdict(premise_adherent=False),
            verdict(comprehensible=False),
            verdict(plot_distinct=False),
            verdict(),
        ]
    )
    client = FakeClient(lambda _: next(stories))
    judge = FakeClient(lambda _: next(judgments))
    result = same_but_different(client, judge, premises=["courier"], attempts=6)
    assert result.score == pytest.approx(2 / 6)
    assert result.metrics["distinct_count"] == 2
    assert len(client.calls) == 6
    assert len(judge.calls) == 5
    record = result.details["premises"][0]
    assert record["acceptance_curve"] == [1, 1, 1, 1, 1, 2]
    assert record["transcript"][4]["rejection_reasons"] == ["exact_duplicate"]
    # Generator receives only summaries, judge receives full accepted stories.
    assert json.loads(client.calls[1][-1]["content"])["excluded_plots"]
    assert json.loads(judge.calls[1][-1]["content"])["accepted_stories"] == ["original"]


@pytest.mark.parametrize(
    "bad",
    [
        "not JSON",
        verdict(plot_distinct="true"),
        verdict(comprehensible=1),
        verdict(evidence=""),
        "[]",
    ],
)
def test_malformed_judgments_are_bounded_and_unresolved(bad):
    client = FakeClient(lambda _: "story")
    judge = FakeClient(lambda _: bad)
    result = same_but_different(client, judge, premises=["premise"], attempts=2)
    assert len(client.calls) == 2
    assert len(judge.calls) == 6
    assert result.score == 0
    assert result.metrics["unresolved_judgments"] == 2
    assert len(result.details["premises"][0]["transcript"][0]["judge_responses"]) == 3


def test_parser_retry_recovers_and_helper_matches_schema():
    outputs = iter(["bad", verdict()])
    judge = FakeClient(lambda _: next(outputs))
    evaluation = evaluate_candidate(judge, premise="p", candidate="c", accepted_stories=[])
    assert evaluation["status"] == "ok"
    assert len(evaluation["judge_responses"]) == 2


def test_generation_failure_consumes_attempt_and_continues():
    def respond(_):
        if len(client.calls) == 1:
            raise RuntimeError("offline")
        return "story"

    client = FakeClient(respond)
    result = same_but_different(client, FakeClient(lambda _: verdict()), premises=["p"], attempts=2)
    assert result.score == 0.5
    assert result.metrics["generation_errors"] == 1


def test_premises_have_independent_exclusion_histories():
    client = FakeClient(lambda _: "story")
    result = same_but_different(
        client, FakeClient(lambda _: verdict()), premises=["p", "q"], attempts=1
    )
    assert result.metrics["distinct_count"] == 2
    assert all(json.loads(c[-1]["content"])["excluded_plots"] == [] for c in client.calls)


@pytest.mark.parametrize("attempts", [0, -1, 21, True, 1.5])
def test_invalid_budget(attempts):
    with pytest.raises(ValueError):
        same_but_different(None, None, premises=["p"], attempts=attempts)
