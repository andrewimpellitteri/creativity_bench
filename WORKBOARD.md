# Workboard — what is open, in what order

Status date: 2026-09-16. Protocol on main: `0.5-coverage`.
Companion docs: [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) (why the measurements
are shaped this way), [README.md](README.md) (how to run them).

Board conventions: **P0** blocks a credible published claim; **P1** is the next
real experiment; **P2** is engineering that removes friction or cost; **P3** is
new surface area. Each item names its *done test* — the observable that closes
it — because "improve the judge" is not a task.

---

## Just landed (2026-09-16)

The three benchmark jobs that had been open since the 2026-08-25 fleet session
(T1, T3, T5 on the fleet board at `../.coordination/CLAIMS.md`, which lives
outside this git repo) are implemented, wired and tested offline:

| Task | Score | Gate | Live evidence |
|---|---|---|---|
| `this_and_that` | Angular interpolation excess vs. an unrelated baseline story | judge: draws on A, draws on B, comprehensible | **none yet** |
| `this_and_that_not` | Share of the pair's angular headroom away from the designated bad example | judge: draws on the good example, comprehensible | **none yet** |
| `copycat` (LLM-uta) | Chance-corrected blinded opening↔continuation matching accuracy | continuation non-empty, not a restatement, comprehensible | **none yet** |
| `quilting` | Validity-gated mean of distinct-recipe rate and story embedding diversity | offline fragment-use check + judge: comprehensible, integrated | **none yet** |

Suite: tests pass offline and ruff is clean under the version pinned in
`.pre-commit-config.yaml` and `ci.yml` (0.16.8 since P2.1 below). A review of
the landing commit found defects in all three; see "Open defects" below — the
scores are not yet trustworthy even offline.

**Consequence you must not miss:** `protocol_fingerprint()` hashes every task
source file, so *all* runs produced from now on land in a different cohort from
`results/extended-20260915/` and `results/saturation-20260915/`, even for the
unchanged five-task subset in `scripts/run_multimodel.sh`. Report and chart code
will refuse to pool them, which is the correct behavior — but it means the
published cohort cannot be extended, only re-run.

---

## P0.0 — defects found reviewing the landing commit (fix before any live run)

A blind review of `a5a1ee7` reproduced each of these against the real modules.
They are P0 because every one of them makes a score mean something other than
what the README says it means. Ordered by severity.

1. **Quilting is non-monotonic: failing a run can double the score.**
   `quilting.py` — the degenerate branch returns `validity_rate`, a different
   scale from the normal branch. Two valid runs with an identical recipe and an
   identical story score **0.25**; the same model scores **0.50** if one run
   emits no `FRAGMENTS:`/`STORY:` markers, because that leaves one valid run and
   trips the degenerate branch. Breaking your own output format is the cheapest
   way to raise this score, and `evaluation_complete` stays `True` throughout.
2. **This & That: copying example A verbatim scores 1.0.** Angular excess is 0
   everywhere on the geodesic *including its endpoints*, so a copy-paste of one
   example is indistinguishable from a true midpoint blend. Only the judge gate
   stands in the way, and that gate has no labeled controls yet (P0.2) — a
   verbatim copy is literally the first missing control. Needs a balance term,
   e.g. `|d(a,c) − d(b,c)| / d(a,b)`, which is 1 at an endpoint and 0 at the
   midpoint, reported and folded into the score.
3. **This & That: a near-zero baseline silently zeroes a whole pair.**
   `baseline_excess` is a *difference* of angular distances and the corpus is
   seven stories in one generated voice, so tiny values are likely, not exotic.
   A reproduced baseline excess of 1.27e-4 sat above the 1e-6 guard, so nothing
   was flagged, and candidates with real excess of 0.005–0.026 all scored
   **0.000**. An uninformative pair must be excluded from the mean and counted,
   never scored 0 — scoring it 0 produces exactly the "looks like low
   creativity" number the design doc forbids.
4. **Quilting's `selection_diversity` has an uncorrected floor of `1/valid`.**
   A totally mode-collapsed model scores 0.5 at the fast size and 0.25 at full,
   so the two sizes are not comparable and the floor is large at shipped sizes.
   Copycat chance-corrects; this should too: `(unique − 1) / (valid − 1)`.
5. **Greedy `{.*}` JSON extraction breaks on ordinary judge prose** — in all
   three tasks. `Recall the schema {...}. My answer: {...}` spans both objects
   and fails to parse. Two failures mark the observation unresolved, which marks
   the **whole run** incomplete, so one chatty judge nullifies a suite run. Both
   retries also send an identical prompt at temperature 0, so the retry is a
   duplicate charge rather than an independent draw (Same But Different already
   sends a repair nudge; these do not).
6. **Copycat's restatement gate is effectively inert.** The 0.8 threshold on
   `lexical_similarity` measures **0.224** for a continuation that quotes the
   opening verbatim and then adds 200 words — i.e. the realistic failure passes
   the gate *and* hands the matcher a verbatim key. The README/workboard claim
   "not a restatement" overstates what it catches.
7. **Copycat may be measuring entity reuse, not voice.** Nothing separates voice
   from subject matter: a model that writes every continuation in its house
   voice but keeps "Augusta" and "INCIDENT REPORT 44-C" scores 1.0. Expect
   saturation at 1.0 for every competent model. Needs an entity-masked matcher
   or a topic-only ablation arm as a false-positive baseline before the number
   discriminates anything.
8. **Quilting's parser rejects plausible real output.** A one-line listing, a
   fragment wrapped across two lines, or `**Story**` without a colon all score
   **0** rather than being recorded as a formatting failure — scoring a format
   problem as low creativity. `malformed_response` needs its own metric and a
   place in the report.
9. **Test gaps that let these through.** No test pins the *direction and
   magnitude* of the excess formula (both existing fixtures are symmetric, so a
   sign flip passes); no test places the candidate at an example; no test asserts
   that adding a valid quilting run cannot lower the score; and no runner-level
   test checks that the new tasks' unresolved metrics flip
   `evaluation_complete` — rename `unresolved_judgments` in any of the three and
   the suite still passes while runs silently look complete.

*Done test for this block:* each defect has a regression test that fails against
the current implementation, and the README's score descriptions match what the
code actually rewards.

---

## P0 — blocks any published claim

### P0.1 Independent human labels on the blinded packet
The packet is exported and leak-scanned at `results/human-review-20260915/`
(`review.html`, identities in `private_key.json`, which must never be shared).
Nothing has been labeled by a person. Everything called "validated" in this repo
currently means "the judge agreed with labels this repo wrote".
*Done test:* ≥2 independent annotators' labels stored outside the packet,
annotator identities and disagreement preserved, and `validate-judge` accuracy
reported against them as a held-out split.

### P0.2 Control sets for the three new judge gates
`validate-judge` exercises only the Same But Different evaluator. The new gates
(`draws_on_a`/`draws_on_b`, opening↔continuation matching, `integrated`) ship
with **no** labeled controls, so their failure modes are unmeasured.
Minimum controls per gate: a copy of one example (should fail "draws on both"),
a generic story mentioning neither, a genuine blend; a continuation swapped onto
the wrong opening, a continuation in the model's house voice, a faithful
pastiche; a story that appends fragments as a list, one that quotes them but is
nonsense, one that weaves them in.
*Done test:* controls added to the `validate-judge` corpus with a split label,
resolution rate and accuracy reported per gate, and a gate whose resolution rate
is below 1.0 blocks the pilot exactly as the Same But Different gate does.

### P0.3 Evaluator self-preference and judge-swap sensitivity
`deepseek-v4-pro` graded its own runs in every pilot so far; the new matching
judge in Copycat has the same exposure, and a matcher may key on topic rather
than voice. Transcripts are saved, so re-judging costs nothing in generation.
*Done test:* `scripts/rescore_judge_swap.py` extended to the new tasks and run
with ≥2 alternative judges; per-task rank correlation and disagreement rate
reported; any task whose ranking flips with the judge is marked unreliable in
the README table.

---

## P1 — the next experiments

### P1.1 First live pilot of the three new tasks
No model has ever produced a single response for `this_and_that`, `copycat` or
`quilting`. Everything known about them comes from fake clients.
```bash
uv run creativity-bench run --model MODEL --judge-model JUDGE \
    --tasks this_and_that,this_and_that_not,copycat,quilting --fast --seed 0 \
    --runs-dir runs/pilot-0.5
```
Do **not** expect `gallery` to work on that run: `gallery.py` raises unless the
run contains `same_but_different` (see P2.9). Read the run JSON directly, or add
`same_but_different` to the task list, until that is fixed.
*Done test:* two models × two seeds saved, transcripts read by hand, and a short
`results/pilot-0.5-*/FINDINGS.md` recording what the scores actually reflected —
especially whether This & That's per-pair baseline is stable and whether Copycat's
matcher used voice or subject matter.

### P1.2 Re-establish a comparable multi-model cohort under 0.5
The 12-model cohort is frozen at the old fingerprint (see the consequence note
above). Decide and record: keep the five-task cheap suite as-is for continuity,
or add `copycat` to it (it needs no embedder, so it is cheap) and re-run.
*Done test:* a new `results/<cohort>/PROTOCOL.md` pre-registering the roster and
sizes, the cohort run, and `REPORT.md` + chart regenerated from a single verified
cohort.

### P1.3 Saturation and budget choice for the new tasks
Fast sizes are guesses: 1 pair, 3 openings, 2 quilting runs. Quilting's
distinct-recipe rate saturates at 1.0 with few runs, and Copycat's chance
correction is coarse at k=3.
*Done test:* acceptance/diversity curves across budgets (pairs 1→6, k 3→6,
runs 2→8) for at least two models, and `_SIZES` updated from the observed curves
rather than from intuition.

### P1.4 Finish the 0.4 research backlog
Carried over from BENCHMARK_DESIGN.md, unchanged: exercise the multi-judge
Shaggy Dog panel with genuinely different judge models (the runner still passes
`judge_clients=[judge_client]`, i.e. one model sampled K times); calibrate
Telephone convergence thresholds on labeled near-duplicate pairs; run
full-budget Same But Different beyond the saturation cohort if budgets change.

---

## P2 — engineering that removes friction

| # | Item | Why it hurts now | Done test |
|---|---|---|---|
| P2.1 | ~~CI lint red on `main`~~ **DONE 2026-09-16** | Both `.github/workflows/ci.yml` and `.pre-commit-config.yaml` now pin ruff 0.16.8 (the pre-commit hook id `ruff` is a legacy alias at that tag and is now `ruff-check`), and the 8 files that had simply never been formatted are formatted. Correction to the original diagnosis: those 8 files failed `format --check` under 0.12.5 **as well**, so this was un-run formatting, not version drift — pinning alone would have left CI red either way. Note that ruff ≥0.16 also formats Markdown, so doc edits are now in the formatter's scope. | Done: `ruff check` and `format --check` clean at 0.16.8, 264 tests pass. |
| P2.2 | **`odd_one_out` scores 0 without a judge** | With `judge_client=None` every item silently scores 0 (`odd_one_out.py:214`), which reads as "the model failed" rather than "no gate was configured". The three new tasks raise instead. | `odd_one_out` raises `ValueError` when no judge is supplied, or explicitly returns an unresolved/incomplete result; test added. |
| P2.3 | **No `--weights` flag** | `run_benchmark` accepts `weights`, but the CLI cannot reach it, so the composite is fixed at equal weights with no way to test alternatives. (Backlog #14 from the 2026-08-25 mailbox.) | `--weights task=value,...` parsed, validated against `TASKS`, echoed into run metadata, test added. |
| P2.4 | **No resumable runs** | A long multi-task run that dies at task 9 of 12 loses everything; `scripts/run_multimodel.sh` is idempotent only at whole-(model, seed) granularity. (Backlog #15.) | Partial task results written incrementally under the run id and reused on re-invocation, with the resume recorded in metadata. |
| P2.5 | **No cost accounting** | Tokens and requests are recorded per run; money is not, deliberately ("prices depend on your providers"). Budget planning for the 0.5 cohort is therefore guesswork. (Backlog #16.) | Optional `--price-table FILE` (per-provider $/Mtok) that annotates the report, clearly labeled as user-supplied, never inferred. |
| P2.6 | **Results tree has three run directories** | `runs/`, `results/runs/` and `results/*/suite_runs/` all hold run JSON, plus `runs/archive_5task/`. Which directory a command should point at is folklore. | One documented layout (or a short `results/README.md` explaining each), and `scripts/*.sh` updated to match. |
| P2.7 | ~~Coordination board stale~~ **DONE 2026-09-16** | The fleet board lives in the parent directory (`../.coordination/`), outside this repo, which is why nothing in git tracks it. T1/T3/T5 rows are closed against the landing commit and the mailbox records the session. | Done. |
| P2.9 | **`gallery` only understands Same But Different** | `gallery.py:44` raises unless the run has a `same_but_different` key, so the transcripts, judge attempts and label permutations the new tasks save so carefully have no inspection surface. "Save enough evidence to re-judge offline" is only half true without a reader. | `gallery` renders per-task sections for every task that saves transcripts, or fails with a message naming what it can render. |
| P2.10 | **Raw metrics never reach the report** | `report.task_diagnostics` covers telephone, subversion and shaggy_dog only. `mean_summed_cosine_distance` (the metric Gwern's spec actually names), `matching_accuracy`, `validity_rate`, `unique_recipes` and `degenerate` exist only in run JSON, so REPORT.md shows a score column with none of the evidence that qualifies it. | Diagnostics extended to every task, with degenerate/low-validity runs visibly marked in the report and chart. |
| P2.11 | **The protocol fingerprint is byte-sensitive** | It hashes source bytes, so a formatting-only commit (P2.1 reformatted `tasks/telephone.py`) starts a new cohort with identical behavior. Correct-but-blunt: it splits cohorts that are genuinely comparable. | Decide deliberately: either accept it and document that cohorts are per-commit, or hash normalized source (e.g. AST dump) and say why that is still safe. |
| P2.8 | **`main` is unpushed** | `main` has been ahead of `origin/main` since before this session (the saturation cohort was never published) and this session added more. (Backlog #19.) | Pushed, and a tag cut if the protocol bump is meant to be a release. |

---

## P3 — tasks from the post that are still unimplemented

Implemented: Free Association, Telephone, Camel's Back, Same But Different,
Don't Repeat Yourself, Extreme Style Transfer, This & That, This & That—But Not
Like That, Copycat (LLM-uta), Quilting, Odd One Out, Subversion, Shaggy Dog.

| Task | Shape | Blocker / note |
|---|---|---|
| Rubric Writing | Explicit tone/POV/locale/form parameters, systematically varied; measure divergence across versions | Needs a parameter grid and a constraint-satisfaction gate per parameter, or it measures nothing but topic |
| Fermi Problem Contest | Curated problems with knowable answers; score the volume of valid-but-varied solution paths | Needs a curated answer corpus with tolerance bounds |
| Worldbuilding / Fanfic Fantasizing | Story plus enumerated unsaid elements; list length as a breadth proxy | List length is trivially gameable; needs a distinctness/validity gate before it means anything |
| Thematic Apperception Test (+ charades variant) | Stories from images or cross-modal descriptions | Needs image input; the client is text-only today |
| Star Chameleon, Exquisite Corpse, Copycat: Truesight, Style Laboratory, multi-agent Free Association | Two or more models interacting, Shapley attribution, ensemble logits | Needs a multi-writer runner (the runner takes exactly one writer client) and, for the logit variant, a provider exposing logprobs — a genuine architecture change, not a task file |

---

## P4 — validity questions no amount of code closes

Carried from BENCHMARK_DESIGN.md and restated here so the board is honest: no
result in this repo shows that these task profiles predict blinded human
assessments of useful creative writing. Until P0.1 and P0.3 are done, every
number here is a property of a pipeline, not of creativity. The composite in
particular remains an exploratory weighted mean of incomparable scales, and
0.5-coverage composites cannot be compared with 0.4-validity composites at all.
