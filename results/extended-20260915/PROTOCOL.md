# Extended cohort protocol (pre-registered before inspecting outcomes)

Recorded before any run of this cohort. Written from the plan, not the results.

## Cohort identity

- Protocol: 0.4-validity plus runner wiring commit `3108b00` (telephone premises and
  conditions, shaggy dog judge panel parameter). The exact `protocol_fingerprint()`
  is stored in `judge_validation.json`, `plan.json`, and every run JSON. This is a
  different fingerprint from `results/pilot-20260915` (pre-wiring) and from all
  legacy/0.3 cohorts. Do not pool across cohorts.
- Provider: DeepSeek. Requested model identifiers: `deepseek-flash`,
  `deepseek-v4-pro`. Fixed judge and summarizer: `deepseek-v4-pro`. V4 Pro judges
  its own outputs in one arm; self-preference bias remains a known limitation.

## Part A: five-task suite, fast sizes, paired seeds

- Tasks: `free_association, camels_back, shaggy_dog, subversion, same_but_different`.
  The four embedding tasks (telephone, diversity, style_transfer, odd_one_out) are
  excluded because no embedding credentials are configured in this environment.
  This cohort is therefore NOT comparable to full nine-task suites, and reports must
  keep it in its own group.
- Budget: `--fast` task sizes; seeds 0 and 1 per model (`--seed 0 --n 2`), paired
  between writers. Four suite runs total.
- Judge gate: re-run `live_pilot.py validate` first because the controls now include
  nine development controls (eight prior plus `injection_paraphrase`) and the
  fingerprint changed. Proceed only if every labeled dimension resolves and matches;
  stop on any incomplete run.

## Part B: Same But Different extension, fast budget, fresh seeds

- `same_but_different` only; seeds 2 and 3, both models, fast budget (two premises,
  three scheduled attempts each). Twenty-four further attempts.
- The SBD task module and its kwargs are unchanged between the pilot-20260915
  fingerprint and this one, and task RNG streams derive from `seed:task`, so seeds
  0-3 sample identical premises across both cohorts. Analyses may present them as
  one SBD series only while labeling the fingerprint split per run.

## Measurements

Per-task scores with the new validity metrics surfaced: Camel's Back schedule
determinism and failed-round audit, Shaggy Dog comprehensibility gate and judge
agreement, Subversion named inversion dimension with sensitivity/specificity, SBD
acceptance curves, distinct valid counts, unresolved judgments, truncations,
per-run token usage, and wall time. No general creativity ranking is inferred from
these smoke budgets.

## Limitations

Fast budgets only; two models; self-judging arm; no embedding tasks; no human
annotations yet (the blinded review packet in `results/human-review-20260915` is
prepared but unlabeled). Results are development data for plumbing, early failure
modes, and cost estimation.
