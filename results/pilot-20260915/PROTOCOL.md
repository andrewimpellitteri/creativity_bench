# Development pilot protocol

This document records the plan before inspecting pilot outcomes.

- Task: Same But Different, current source fingerprint saved with validation and runs.
- Provider: DeepSeek. Authenticated model discovery returned `deepseek-flash` and
  `deepseek-v4-pro`; use those exact requested identifiers.
- Writers: both discovered models. Fixed judge/summarizer: `deepseek-v4-pro`.
- Seeds: 0 and 1, paired between writers. Fast budget: two premises per seed,
  three scheduled attempts per premise. Total: 24 story attempts.
- First run the eight bundled development controls using the production judge.
  Proceed only if all labeled dimensions resolve and match their proposed labels.
  This is a conservative operational gate, not a validated statistical threshold.
- Save the judge's raw responses, token usage and request finish reasons.
- Stop on incomplete runs; preserve failure artifacts. Do not silently drop failed
  attempts or select the better result from repeated runs.
- Primary measurements: accepted distinct counts and per-attempt acceptance curves.
  Report task acceptance fractions, validity, unresolved judgments, truncations,
  and matched-seed differences. Do not infer a general creativity ranking.
- Do not pool these smoke budgets with full-budget or earlier-protocol results.
- Make a blinded review packet for independent human review. Hide model identity,
  judge verdicts, rationales, and proposed control labels from reviewers.

## Known limitations

Eight public author-labeled controls are development data. They do not establish
human agreement or held-out validity. Two seeds and six attempts per run provide
only a plumbing and early failure-mode pilot, not a powered model comparison.
V4 Pro judges itself in one arm; self-preference and shared-family bias remain.
Model names are provider aliases; retain returned model identifiers where available.

Independent human review has not yet occurred. No reviewers will be contacted or
represented as having provided labels by this automated work.
