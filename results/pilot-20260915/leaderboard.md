# Creativity Bench — Leaderboard

Generated 2026-09-15 from 4 runs of 2 models in `results/pilot-20260915/runs/`.
Excluded 0 incomplete evaluations: unresolved judgments or generation errors yield audit lower bounds, not comparable creativity scores.
Task profiles are primary. The composite is exploratory: its weighting has not been validated as a measure of creativity. Models are listed alphabetically.
Different protocol cohorts are not directly comparable; no cross-cohort ranking is made.

## Cohort 1 — verified provenance

Protocol: `0.4-validity`; fast: `True`; judge: `deepseek-v4-pro`.

| Model | Same but different | Free association | Telephone game | Camel's back | Diversity | Style transfer | Odd one out | Subversion | Shaggy dog | Exploratory composite ± SD | n runs | Seeds | Judge | Runs from |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `deepseek-flash` ⚡ | 1.000 | — | — | — | — | — | — | — | — | 1.000 ± 0.000 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-15 |
| `deepseek-v4-pro` ⚡ | 0.917 | — | — | — | — | — | — | — | — | 0.917 ± 0.083 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-15 |

### Paired task differences

Differences are left minus right; percentile bootstrap 95% intervals resample matched seeds. Duplicate seed runs are averaged first. Intervals are descriptive, without multiple-comparison correction.

| Left - right | Task | Matched seeds | Difference | 95% interval |
|---|---|---:|---:|---|
| `deepseek-flash` - `deepseek-v4-pro` | same_but_different | 2 | 0.083 | [0.000, 0.167] |

## Notes

- ⚡ denotes fast task budgets. Cohorts split by protocol, tasks, weights, budgets, judge, embedding and generation settings.
- Legacy or incomplete provenance cannot establish compatibility; these runs are shown separately by model and are excluded from paired inference.
- Profile means weight saved runs equally; paired differences weight matched seeds equally after averaging duplicates, so their differences may differ.
- SD describes variation across saved runs, not uncertainty from independent samples. Pairwise story distances are dependent and are never bootstrap units.
- Judge-dependent scores inherit the judge model's biases.
- At least one model graded its own outputs (see Judge); interpret judge-dependent scores with caution.
