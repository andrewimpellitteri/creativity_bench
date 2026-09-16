"""Judge-swap sensitivity: re-judge saved Same But Different transcripts.

Re-judges every originally judged attempt of saved runs with an alternative
judge, reusing the production prompt and parsing unchanged via
``creativity_bench.tasks.same_but_different.evaluate_candidate`` and the
original accepted-story context, then reports per-dimension agreement, flips,
and unresolved judgments. No stories are regenerated.

Judge-swap measures judge sensitivity, not story quality. Agreement with the
original judge is development evidence, not human agreement or validity.

Usage (live, later):

    uv run python scripts/rescore_judge_swap.py \
        --run results/extended-20260915/suite_runs/deepseek-flash_20260915-171544_d0dcb7.json \
        --judge-model deepseek-flash --judge-provider deepseek \
        --out results/judge-swap/deepseek-flash_swap.json

Alternative judge on the GLM coding endpoint:

    uv run python scripts/rescore_judge_swap.py \
        --run results/extended-20260915/suite_runs/glm-5.3-flash_20260915-194725_c903b3.json \
        --judge-model glm-5.3-flash --judge-provider zai-coding \
        --out results/judge-swap/glm-5.3-flash_swap.json

Credentials come from the environment or a local .env (same loading rule as
scripts/live_pilot.py) and are never printed or written to the report.
Offline tests use fake clients: uv run pytest tests/test_judge_swap.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

from creativity_bench.tasks.same_but_different import evaluate_candidate

DIMENSIONS = ("premise_adherent", "comprehensible", "plot_distinct")
NOT_SENT_TO_JUDGE = frozenset(
    {"generation_error", "empty_story", "story_too_long", "exact_duplicate"}
)
REPORT_SCHEMA = "judge-swap-v1"
NOTICE = "Judge-swap measures judge sensitivity, not story quality."


class RunSchemaError(ValueError):
    """Raised when a run file does not match the saved transcript schema."""


def _load_local_key() -> None:
    path = Path(__file__).resolve().parent / "live_pilot.py"
    spec = importlib.util.spec_from_file_location("rescore_live_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_local_key()


def build_judge(provider_name: str, model: str):
    """Build the alternative judge exactly like scripts/live_pilot.py does."""
    from creativity_bench.client import LLMClient, resolve_provider

    _load_local_key()
    provider = resolve_provider(provider_name)
    return LLMClient(provider=provider, model=model, max_retries=2, timeout=90)


def _load_sbd_premises(path: Path) -> tuple[dict, list]:
    try:
        run = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RunSchemaError(f"unreadable or invalid JSON ({exc})") from None
    if not isinstance(run, dict):
        raise RunSchemaError("top level is not a JSON object")
    tasks = run.get("tasks")
    if not isinstance(tasks, dict) or not isinstance(tasks.get("same_but_different"), dict):
        raise RunSchemaError("no saved same_but_different task result")
    details = tasks["same_but_different"].get("details")
    premises = details.get("premises") if isinstance(details, dict) else None
    if not isinstance(premises, list) or not premises:
        raise RunSchemaError("no saved premises with transcripts")
    for index, record in enumerate(premises):
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("premise"), str)
            or not isinstance(record.get("transcript"), list)
        ):
            raise RunSchemaError(f"premise record {index} has no premise/transcript")
    return run, premises


def _bump(counts: dict, key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _rescore_attempt(judge_client, *, run_id, premise_index, premise, entry, context) -> tuple:
    evaluation = evaluate_candidate(
        judge_client,
        premise=premise,
        candidate=entry["story"],
        accepted_stories=list(context),
    )
    original = entry.get("verdict")
    record = {
        "attempt_id": f"{run_id}:{premise_index}:{entry.get('attempt')}",
        "premise_index": premise_index,
        "attempt": entry.get("attempt"),
        "story": entry["story"],
        "accepted_stories_context": list(context),
        "original_verdict": original,
        "original_judge_responses": list(entry.get("judge_responses") or []),
        "original_accepted": entry.get("accepted"),
        "swapped_verdict": evaluation["verdict"],
        "swapped_judge_responses": list(evaluation["judge_responses"]),
        "swapped_status": evaluation["status"],
        "swapped_error": evaluation.get("judge_error"),
    }
    return record, original, evaluation["verdict"]


def rescore_run(path: Path, judge_client, *, limit: int | None = None) -> dict:
    """Re-judge one saved run; raises RunSchemaError on schema mismatch."""
    run, premises = _load_sbd_premises(Path(path))
    metadata = run.get("metadata") or {}
    task_details = run["tasks"]["same_but_different"].get("details") or {}
    run_report = {
        "run_id": Path(path).stem,
        "run_path": str(path),
        "model": run.get("model"),
        "provider": run.get("provider"),
        "seed": run.get("seed"),
        "source_protocol_fingerprint": metadata.get("protocol_fingerprint"),
        "source_judge_model": metadata.get("judge_model")
        or task_details.get("judge_model"),
        "attempted": 0,
        "skipped_attempts": {},
        "unresolved_original": 0,
        "unresolved_swapped": 0,
        "context_warnings": [],
        "agreement": {dim: {"agree": 0, "comparable": 0, "rate": None} for dim in DIMENSIONS},
        "flips": {dim: [] for dim in DIMENSIONS},
        "attempts": [],
    }
    seen = 0
    for premise_index, record in enumerate(premises):
        premise = record["premise"]
        transcript = record["transcript"]
        context: list[str] = []
        for entry in transcript:
            story = entry.get("story")
            reasons = set(entry.get("rejection_reasons") or [])
            if not isinstance(story, str) or not story.strip():
                _bump(run_report["skipped_attempts"], "no_story")
            elif reasons & NOT_SENT_TO_JUDGE:
                _bump(run_report["skipped_attempts"], "not_sent_to_judge")
            elif limit is not None and seen >= limit:
                _bump(run_report["skipped_attempts"], "limit")
            else:
                seen += 1
                attempt_record, original, swapped = _rescore_attempt(
                    judge_client,
                    run_id=run_report["run_id"],
                    premise_index=premise_index,
                    premise=premise,
                    entry=entry,
                    context=context,
                )
                comparable = original is not None and swapped is not None
                attempt_record["comparable"] = comparable
                attempt_record["agreement"] = {
                    dim: (original[dim] == swapped[dim]) if comparable else None
                    for dim in DIMENSIONS
                }
                if comparable:
                    for dim in DIMENSIONS:
                        cell = run_report["agreement"][dim]
                        cell["comparable"] += 1
                        if original[dim] == swapped[dim]:
                            cell["agree"] += 1
                        else:
                            run_report["flips"][dim].append(
                                {
                                    "attempt_id": attempt_record["attempt_id"],
                                    "original": original[dim],
                                    "swapped": swapped[dim],
                                    "original_evidence": original.get("evidence"),
                                    "swapped_evidence": swapped.get("evidence"),
                                }
                            )
                if original is None:
                    run_report["unresolved_original"] += 1
                if swapped is None:
                    run_report["unresolved_swapped"] += 1
                run_report["attempted"] += 1
                run_report["attempts"].append(attempt_record)
            if entry.get("accepted") and isinstance(story, str) and story.strip():
                context.append(story)
        saved = record.get("accepted_stories")
        if isinstance(saved, list):
            saved_stories = [a.get("story") for a in saved if isinstance(a, dict)]
            if saved_stories != context:
                run_report["context_warnings"].append(premise_index)
    for cell in run_report["agreement"].values():
        if cell["comparable"]:
            cell["rate"] = cell["agree"] / cell["comparable"]
    return run_report


def _totals(runs: list[dict], skipped_attempts: dict) -> dict:
    totals = {
        "attempts_rejudged": sum(r["attempted"] for r in runs),
        "unresolved_original": sum(r["unresolved_original"] for r in runs),
        "unresolved_swapped": sum(r["unresolved_swapped"] for r in runs),
        "skipped_attempts": skipped_attempts,
        "agreement": {},
        "flips": {},
    }
    for dim in DIMENSIONS:
        comparable = sum(r["agreement"][dim]["comparable"] for r in runs)
        agree = sum(r["agreement"][dim]["agree"] for r in runs)
        totals["agreement"][dim] = {
            "agree": agree,
            "comparable": comparable,
            "rate": agree / comparable if comparable else None,
        }
        totals["flips"][dim] = sum(len(r["flips"][dim]) for r in runs)
    return totals


def rescore_runs(run_paths, judge_client, *, limit: int | None = None) -> dict:
    """Re-judge multiple runs, skipping schema mismatches with a clear message."""
    runs, skipped = [], []
    skipped_attempts: dict = {}
    for path in run_paths:
        path = Path(path)
        try:
            runs.append(rescore_run(path, judge_client, limit=limit))
        except RunSchemaError as exc:
            print(f"Skipping {path}: schema mismatch ({exc})", file=sys.stderr)
            skipped.append({"path": str(path), "reason": str(exc)})
    for run in runs:
        for key, count in run["skipped_attempts"].items():
            skipped_attempts[key] = skipped_attempts.get(key, 0) + count
    fingerprints = sorted(
        {r["source_protocol_fingerprint"] for r in runs if r["source_protocol_fingerprint"]}
    )
    return {
        "tool": "rescore_judge_swap",
        "schema": REPORT_SCHEMA,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notice": NOTICE,
        "alternative_judge": {
            "model": getattr(judge_client, "model", None),
            "provider": getattr(getattr(judge_client, "provider", None), "name", None),
            "base_url": getattr(getattr(judge_client, "provider", None), "base_url", None),
        },
        "source_protocol_fingerprints": fingerprints,
        "runs": runs,
        "skipped_runs": skipped,
        "totals": _totals(runs, skipped_attempts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-judge saved Same But Different transcripts with an alternative judge."
    )
    parser.add_argument("--run", nargs="+", type=Path, required=True, metavar="PATH")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-provider", default="deepseek")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    judge = build_judge(args.judge_provider, args.judge_model)
    report = rescore_runs(args.run, judge, limit=args.limit)
    if not report["runs"]:
        print("No usable run files; nothing to report", file=sys.stderr)
        return 1
    payload = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
