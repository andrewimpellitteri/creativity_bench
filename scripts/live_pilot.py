"""Bounded live pilot; credentials remain local and are never written to artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from creativity_bench.client import LLMClient, resolve_provider


def load_local_key() -> None:
    # Read just the authorized provider key; do not execute a shell dotenv file.
    path = Path(".env")
    if os.environ.get("DEEPSEEK_API_KEY") or not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.removeprefix("export ").partition("=")
        if sep and key.strip() == "DEEPSEEK_API_KEY":
            os.environ[key.strip()] = value.strip().strip('"\x27')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["models", "validate", "run"])
    parser.add_argument("--judge")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/pilot-20260915"))
    args = parser.parse_args()
    load_local_key()
    provider = resolve_provider("deepseek")
    if args.action == "models":
        client = LLMClient(provider=provider, model="discovery", max_retries=0, timeout=30)
        print(json.dumps([m.id for m in client._client.models.list().data]))
        return
    if not args.judge:
        parser.error("--judge is required")
    args.out.mkdir(parents=True, exist_ok=True)
    judge = LLMClient(provider=provider, model=args.judge, max_retries=2, timeout=90)
    from creativity_bench.runner import protocol_fingerprint, run_benchmark, save_run

    if args.action == "validate":
        from creativity_bench.calibration import load_controls, validate_judge

        result = validate_judge(judge, load_controls())
        result["protocol_fingerprint"] = protocol_fingerprint()
        result["usage"] = dict(vars(judge.usage))
        result["requests"] = list(judge.request_log)
        path = args.out / "judge_validation.json"
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result["summaries"], indent=2), flush=True)
        print(f"Saved {path}", flush=True)
        return
    if not args.models:
        parser.error("--models is required for run")
    validation = args.out / "judge_validation.json"
    if not validation.exists():
        parser.error("Run judge validation first")
    v = json.loads(validation.read_text())
    if v["judge_model"] != args.judge or v["protocol_fingerprint"] != protocol_fingerprint():
        parser.error("Validation judge/protocol does not match this pilot")
    # Stop on any unresolved or incorrect development control, not a validated
    # acceptance threshold. Review failures before spending a larger pilot budget.
    if any(
        d["resolution_rate"] != 1 or d["accuracy_resolved"] != 1
        for d in v["summaries"]["development"].values()
    ):
        parser.error("Judge did not pass all development controls; inspect validation first")
    plan = {
        "models": args.models,
        "judge": args.judge,
        "provider": "deepseek",
        "seeds": args.seeds,
        "fast": args.fast,
        "protocol_fingerprint": protocol_fingerprint(),
        "note": "Development pilot; no independent human validation. Judge self-bias possible.",
    }
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))
    for seed in args.seeds:
        for model in args.models:
            client = LLMClient(provider=provider, model=model, max_retries=2, timeout=90)
            print(f"Starting {model}, seed {seed}", flush=True)
            result = run_benchmark(
                client,
                judge,
                None,
                tasks=["same_but_different"],
                seed=seed,
                fast=args.fast,
                verbose=True,
            )
            path = save_run(result, args.out / "runs")
            print(f"Saved {path}", flush=True)
            if not result.metadata["evaluation_complete"]:
                raise RuntimeError("Incomplete run saved; stopping pilot for inspection")


if __name__ == "__main__":
    main()
