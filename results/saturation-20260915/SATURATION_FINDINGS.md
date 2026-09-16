# Saturation cohort findings — 2026-09-16

Pre-registered protocol in `PROTOCOL.md`; judge gate passed 9/9 before any spend.
Two runs: six premises, ten scheduled attempts each (60 attempts per model),
seed 0, judge `deepseek-v4-pro`.

## Headline: no saturation at ten attempts

Acceptance curves are close to linear through attempt 10 for both models:

- `deepseek-flash`: 55/60 accepted (0.92), 65 min, two premises hit 10/10.
- `deepseek-v4-pro`: 51/60 accepted (0.85), 145 min, best premise 9/10.

Per-premise final acceptance: flash 9, 9, 8, 10, 9, 10; v4-pro 8, 9, 8, 9, 9, 8.
Every curve's slope between attempts 8-10 remains near the attempt-1 slope; the
fast budget's near-ceiling scores reflected the three-attempt cap, not model
skill. Attempt count, not premise variety, was the binding constraint.

## Interpretation (development data, one seed)

- The fixed budget does not yet bound the models' distinct-plot supply. A pilot
  intending to observe saturation needs a larger budget (20+) or a harder
  distinctness bar; otherwise it measures the budget, not the model.
- v4-pro's rejections cluster on `plot_distinct` (repeats of its own earlier
  causal plots) in later attempts - the intended failure mode, now visible.
- Cost: ~1.1 and ~2.4 s/attempt-second scale; v4-pro costs ~2.2x flash in wall
  time for 4 fewer accepted plots (not a ranking: one seed, shared judge family).

## Known limitations

Single seed; single provider family; judge shares a family with the v4-pro arm;
premises are six samples from a 20-premise corpus. Judge-swap sensitivity on
these transcripts should run before any cross-judge claims.
