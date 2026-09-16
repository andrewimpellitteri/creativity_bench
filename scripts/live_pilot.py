"""Bounded live pilot; credentials remain local and are never written to artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from creativity_bench.calibration import (
    DEFAULT_GATE,
    all_default_controls,
    gate_failures,
    gate_names,
    load_controls,
)
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


def controls_sha256(payload) -> str:
    """Hash a control set exactly as calibration.py hashes it while validating."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validated_gates(validation: dict) -> list[str]:
    """The judge gates a saved validation actually ran."""
    if validation.get("kind") == "judge_control_validation_suite":
        return list(validation["gates"])
    return [validation.get("gate", DEFAULT_GATE)]


def expected_controls_sha(validation: dict) -> str:
    """The bundled-control hash a saved validation must carry to gate this pilot.

    validate_gates() hashes the ``{gate: controls}`` mapping it validated;
    validate_judge() hashes one gate's control list. Hashing the bundled controls
    the same way the saved validation hashed its own keeps the check meaningful
    for either shape, and still rejects a validation run against controls that
    have changed since.
    """
    if validation.get("kind") == "judge_control_validation_suite":
        return controls_sha256(all_default_controls())
    return controls_sha256(load_controls(gate=validation.get("gate", DEFAULT_GATE)))


def validation_gate_error(
    validation: dict, *, judge: str, fingerprint: str, controls_sha: str
) -> str | None:
    """Return a rejection reason when a saved validation cannot gate this pilot.

    calibration.py is outside the protocol fingerprint, so the validation's
    controls hash must be checked explicitly: a validation produced against a
    different control set would otherwise pass this gate. Every registered gate
    must be covered, and every one of them must pass, not only Same But
    Different.
    """
    if validation["judge_model"] != judge:
        return "Validation judge does not match this pilot"
    if validation["protocol_fingerprint"] != fingerprint:
        return "Validation protocol does not match this pilot"
    if validation.get("controls_sha256") != controls_sha:
        return "Validation controls do not match this pilot's control set"
    missing = [gate for gate in gate_names() if gate not in validated_gates(validation)]
    if missing:
        return f"Validation does not cover judge gate(s): {', '.join(missing)}"
    # Stop on any unresolved or incorrect development control, on any gate. This
    # is a stop-and-inspect rule, not a validated acceptance threshold: review
    # the failures before spending a larger pilot budget.
    failures = gate_failures(validation)
    if failures:
        return "Judge did not pass all development controls; inspect first: " + "; ".join(failures)
    return None


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
        from creativity_bench.calibration import validate_gates

        # Every registered gate, one judge call per control (two or three when a
        # response fails schema validation). The pilot blocks on any of them.
        result = validate_gates(judge)
        result["protocol_fingerprint"] = protocol_fingerprint()
        result["usage"] = dict(vars(judge.usage))
        result["requests"] = list(judge.request_log)
        path = args.out / "judge_validation.json"
        path.write_text(json.dumps(result, indent=2))
        summaries = {name: gate["summaries"] for name, gate in result["gates"].items()}
        print(json.dumps(summaries, indent=2), flush=True)
        for reason in gate_failures(result):
            print(f"blocker: {reason}", flush=True)
        print(f"Saved {path}", flush=True)
        return
    if not args.models:
        parser.error("--models is required for run")
    validation = args.out / "judge_validation.json"
    if not validation.exists():
        parser.error("Run judge validation first")
    v = json.loads(validation.read_text())
    error = validation_gate_error(
        v,
        judge=args.judge,
        fingerprint=protocol_fingerprint(),
        controls_sha=expected_controls_sha(v),
    )
    if error:
        parser.error(error)
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
