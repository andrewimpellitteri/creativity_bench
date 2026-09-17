# Extended cohort findings — 2026-09-15

Development pilot results. Two seeds, two models, fast budgets. These numbers
describe plumbing behavior and early failure modes; they do not rank the models
on creativity and must not be quoted as such.

## What ran

- **Judge gate**: `deepseek-v4-pro` on all nine development controls (eight prior
  plus the new `injection_paraphrase`): 100% resolution and 100% accuracy on every
  dimension. Saved with protocol fingerprint, raw responses, and usage.
- **Suite** (Part A): `free_association, camels_back, shaggy_dog, subversion,
  same_but_different` at fast sizes, seeds 0 and 1 per model, paired. Fixed judge
  and summarizer: `deepseek-v4-pro`. Four runs, all `evaluation_complete`, zero
  unresolved judgments, zero generation errors.
- **SBD extension** (Part B): seeds 2 and 3, both models, fast budget (two
  premises × three scheduled attempts). Four runs, all complete.
- **Blinded review packet**: 33 items (24 pilot attempts + 9 controls), shuffled
  under seed 42, identity leak-scanned clean, exported to
  `results/human-review-20260915/`. Unlabeled — no reviewers yet.

## Suite results (mean over seeds 0, 1)

| model | composite | free_assoc | camels_back | shaggy_dog | subversion | SBD | wall/run | tokens gen+judge (k) |
|---|---|---|---|---|---|---|---|---|
| deepseek-flash | 0.91 | 1.00 | 1.00 | 0.63 | 1.00 | 0.92 | 6–7 min | ≈107 |
| deepseek-v4-pro | 0.93 | 1.00 | 1.00 | 0.65 | 1.00 | 1.00 | 10–14 min | ≈180 |

Per-run detail: 25 generation requests and 17 judge requests per suite run;
`REPORT.md` has the leaderboard view (single verified cohort).

## Same But Different series (fast budget, 6 attempts/run, seeds 0–3)

| model | seed 0 | seed 1 | seed 2 | seed 3 | total distinct valid |
|---|---|---|---|---|---|
| deepseek-flash | 6/6 | 6/6 | 6/6 | 5/6 | 23/24 |
| deepseek-v4-pro | 5/6 | 6/6 | 6/6 | 6/6 | 23/24 |

Every rejection was a genuine repeat (same causal plot, renamed surface), judged
against full prior stories with saved evidence. All 48 attempts were
premise-adherent and comprehensible.

## Reading

1. **Fast budgets saturate.** Free association, Camel's Back and Subversion score
   1.00 for both models at fast sizes: these sizes do not discriminate these
   models. Meaningful profiles need full sizes (or harder constraints).
2. **Shaggy Dog discriminates at this budget** (0.63 vs 0.65, with the new
   fail-closed comprehensibility gate active). It is currently the most
   informative task per token spent, but its agreement-based score still rewards
   overlapping judgments; multi-judge scaffolding is wired but unexercised.
3. **SBD is near ceiling at 3 attempts.** The interesting saturation regime starts
   beyond attempt 3; the full 10-attempt budget is the next measurement.
4. **Validity infrastructure held in production.** Nine controls gate every run;
   unresolved judgments would have marked runs incomplete; none occurred.
   Truncations: none observed at the configured limits.

## Threats to validity

- V4 Pro judged its own arm (self-preference undetectable at this resolution).
- Author-labeled development controls, not human annotations; the blinded packet
  exists but is unlabeled.
- Suite cohort excludes the four embedding tasks (no embedding credentials), so
  this is a five-task profile, not the nine-task suite.
- Two seeds; wide uncertainty; premises are a small corpus sample.

## Next steps, in order

1. Full-budget SBD (6 premises × 10 attempts) for saturation curves and cost
   extrapolation before any larger panel.
2. Human labeling of the blinded packet (two-plus independent reviewers) to
   measure judge agreement on plot distinctness and validity.
3. Embedding credentials → run the four embedding tasks → first nine-task
   profile cohort (kept separate from this one).
4. Judge-swap sensitivity: re-judge saved SBD transcripts with `deepseek-flash`
   as judge (transcripts are saved; no regeneration cost).
5. Multi-judge Shaggy Dog panel once a second judge provider is configured.

## Cohort map (do not pool)

| cohort | fingerprint | contents |
|---|---|---|
| legacy runs/ + results/runs | pre-0.4 | deepseek-chat, gpt-4o-mini, gpt-5-mini full suites |
| results/pilot-20260915 | 0.4-validity pre-wiring | SBD only, seeds 0–1 |
| results/extended-20260915 | 0.4-validity post-wiring | 5-task suite, 12 models (below) + SBD seeds 2–3 |

## Addendum 2: full roster — 23 models, n=2 everywhere (2026-09-17)

The cohort now holds 46 complete runs of 23 models (all n=2, seeds 0+1, zero
unresolved judgments, zero incomplete evaluations): the original twelve, seven
OpenRouter expansion models, a GLM seed-1 backfill, and four more families
(grok-4.20, gemini-3-flash-preview, command-a, phi-4). Judge-swap sensitivity
for the whole cohort's SBD attempts is in `../judge-swap/SUMMARY.md`
(99-100% agreement on validity dimensions, 86-91% on plot distinctness, one
alternative judge 33% unresolved and excluded fail-closed).

Composite ranking (exploratory, unvalidated weighting): deepseek-v4-pro 0.93
and glm-5.3-flash 0.93 lead; glm-4.5-air 0.91 and deepseek-flash 0.91 next;
kimi-k3 0.88 best OpenRouter debut; phi-4 0.43 and gemini-2.5-flash-lite 0.58
trail. Per-task failure signatures are the interesting signal: gpt-5-mini
collapses on Camel's Back (0.17), gemini-2.5-flash-lite and qwen3.7-flash on
Subversion (0.00/0.25), grok-4.20 on free association (0.65) and SBD (0.33),
while shaggy dog stays the only task with a smooth 0.6-0.8 spread. The judge
model ranks first overall: self-judging bias remains a live caveat, mitigated
by the swap evidence but not eliminated. Serving path matters: DeepSeek V4
Flash via OpenRouter scores 0.50 on free association vs 1.00 direct.

Charts: `BENCHMARK_GRAPH.png` (original twelve), `GRAPH_EXPANSION.png` (the
eleven later additions plus both deepseek anchors). Full table: `REPORT.md`.
`scripts/run_multimodel.sh` is now the single idempotent driver for the whole
roster (skips every saved model/seed pair; reruns only fill gaps).

## Addendum 1: multi-model expansion (same cohort, same day)

After the initial four deepseek/GLM runs, the cohort grew to **12 models**
(20 runs) by adding four GLM variants via the z.ai coding endpoint and six
OpenRouter value models (Gemini 2.5 Flash, GPT-4o-mini, Claude Haiku 4.5,
Mistral Small 3.2, Llama-4-Scout, Kimi K2.5), all judged by the same
`deepseek-v4-pro`. The cohort signature was deliberately relaxed so one pinned
protocol spans API vendors; provider identity stays in metadata and provenance
verification. Chart: `BENCHMARK_GRAPH.png`; table: `REPORT.md`.

Composite ranking (exploratory, unvalidated weighting; n runs in parens):

glm-5.3-flash 0.95 (1) > deepseek-v4-pro 0.93 (2) > deepseek-flash 0.91 (2) >
glm-4.5-air 0.90 (1) > claude-haiku-4.5 0.87 (2) > glm-4.6 0.83 (1) =
kimi-k2.5 0.83 (2) > glm-5-turbo 0.79 (1) > gemini-2.5-flash 0.76 (2) >
mistral-small-3.2 0.72 (2) > gpt-4o-mini 0.71 (2) > llama-4-scout 0.68 (2).

Observations: shaggy dog remains the discriminating task (0.50–0.81 spread);
free association saturates at 1.00 for every model; Camel's Back and Subversion
now show real failures at fast sizes (kimi 0.50 camel's back; glm-4.6/glm-5-turbo
0.50 subversion) that the earlier two-model pilot did not surface. Same but
different spreads 0.33–1.00 and tracks the composite better than any other task.
GLM models ran with n=1 (seed 0 only; slower endpoint); an idempotent seed-1
pass can fill them later via `scripts/run_multimodel.sh`. One mistral seed-0
run was quarantined under `incomplete/` after a transport failure and re-run
clean rather than being silently dropped.
