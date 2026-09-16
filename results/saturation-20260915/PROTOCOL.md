# Saturation cohort protocol (pre-registered before inspecting outcomes)

Recorded before any run of this cohort.

- Purpose: measure Same But Different acceptance **saturation** at the full
  attempt budget. The fast budget (two premises, three attempts) ended near
  ceiling for every model tested (23/24 and 47/48 distinct-valid across seeds),
  so the interesting measurement starts beyond attempt three.
- Task: `same_but_different`, full sizes: six sampled premises, ten scheduled
  attempts per premise (sixty attempts per run). Rejected stories consume
  attempts; there is no retry-until-success.
- Models: `deepseek-flash` and `deepseek-v4-pro` (paired, seed 0). Fixed judge
  and summarizer: `deepseek-v4-pro`. Judge gate: the bundled nine development
  controls must resolve and match at 100% before any spend.
- Cohort: new fingerprint-scoped directory `results/saturation-20260915/`.
  Not pooled with the fast-budget suites, the pre-wiring pilot, or legacy runs.
- Measurements: cumulative distinct-valid stories vs attempt number
  (acceptance curves), per-premise saturation points, rejection evidence
  categories, unresolved judgments (none expected; any marks the run
  incomplete), truncations, token usage and wall time per attempt index.
- No general creativity ranking is inferred from two models at one seed.

## Known limitations

One seed; two models from one provider family; the judge shares a family with
one writer arm (self-preference threat, undetectable here); development-control
gate is author-labeled, not human-annotated.
