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
(T1, T3, T5 in `.coordination/CLAIMS.md`) are implemented, wired and tested
offline:

| Task | Score | Gate | Live evidence |
|---|---|---|---|
| `this_and_that` | Angular interpolation excess vs. an unrelated baseline story | judge: draws on A, draws on B, comprehensible | **none yet** |
| `this_and_that_not` | Share of the pair's angular headroom away from the designated bad example | judge: draws on the good example, comprehensible | **none yet** |
| `copycat` (LLM-uta) | Chance-corrected blinded opening↔continuation matching accuracy | continuation non-empty, not a restatement, comprehensible | **none yet** |
| `quilting` | Validity-gated mean of distinct-recipe rate and story embedding diversity | offline fragment-use check + judge: comprehensible, integrated | **none yet** |

Suite: 264 tests pass offline; `ruff check` clean under the pinned 0.12.5.

**Consequence you must not miss:** `protocol_fingerprint()` hashes every task
source file, so *all* runs produced from now on land in a different cohort from
`results/extended-20260915/` and `results/saturation-20260915/`, even for the
unchanged five-task subset in `scripts/run_multimodel.sh`. Report and chart code
will refuse to pool them, which is the correct behavior — but it means the
published cohort cannot be extended, only re-run.

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
    --tasks this_and_that,copycat,quilting --fast --seed 0 \
    --runs-dir runs/pilot-0.5
uv run creativity-bench gallery --run runs/pilot-0.5/RUN.json --out results/gallery-0.5.html
```
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
| P2.1 | **CI lint is red on `main`** | `.github/workflows/ci.yml` uses `astral-sh/ruff-action@v3` with no version, so CI runs the newest ruff (0.16.8), while `.pre-commit-config.yaml` pins 0.12.5. Under 0.16.8, 8 pre-existing files would be reformatted: `scripts/live_pilot.py`, `scripts/rescore_judge_swap.py`, `src/creativity_bench/report.py`, `src/creativity_bench/tasks/telephone.py`, `tests/test_judge_swap.py`, `tests/test_live_pilot_gate.py`, `tests/test_runner_wiring.py`, `tests/test_subversion_dimensions.py`. | One ruff version pinned in both places, `ruff format` run once across the repo, CI green. |
| P2.2 | **`odd_one_out` scores 0 without a judge** | With `judge_client=None` every item silently scores 0 (`odd_one_out.py:214`), which reads as "the model failed" rather than "no gate was configured". The three new tasks raise instead. | `odd_one_out` raises `ValueError` when no judge is supplied, or explicitly returns an unresolved/incomplete result; test added. |
| P2.3 | **No `--weights` flag** | `run_benchmark` accepts `weights`, but the CLI cannot reach it, so the composite is fixed at equal weights with no way to test alternatives. (Backlog #14 from the 2026-08-25 mailbox.) | `--weights task=value,...` parsed, validated against `TASKS`, echoed into run metadata, test added. |
| P2.4 | **No resumable runs** | A long multi-task run that dies at task 9 of 12 loses everything; `scripts/run_multimodel.sh` is idempotent only at whole-(model, seed) granularity. (Backlog #15.) | Partial task results written incrementally under the run id and reused on re-invocation, with the resume recorded in metadata. |
| P2.5 | **No cost accounting** | Tokens and requests are recorded per run; money is not, deliberately ("prices depend on your providers"). Budget planning for the 0.5 cohort is therefore guesswork. (Backlog #16.) | Optional `--price-table FILE` (per-provider $/Mtok) that annotates the report, clearly labeled as user-supplied, never inferred. |
| P2.6 | **Results tree has three run directories** | `runs/`, `results/runs/` and `results/*/suite_runs/` all hold run JSON, plus `runs/archive_5task/`. Which directory a command should point at is folklore. | One documented layout (or a short `results/README.md` explaining each), and `scripts/*.sh` updated to match. |
| P2.7 | **Coordination board is stale** | `.coordination/CLAIMS.md` still lists T1/T3/T5 as `pending` with "re-dispatch fresh", which is now wrong. | Rows closed with the landing commit, and the mailbox noting the fleet session is finished. |
| P2.8 | **3 unpushed commits on `main`** | `main` is ahead of `origin/main` by 3; the saturation cohort is not published. (Backlog #19.) | Pushed, and a tag cut if the protocol bump is meant to be a release. |

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
