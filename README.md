# LLM Creativity Benchmark

[![CI](https://github.com/andrewimpellitteri/creativity_bench/actions/workflows/ci.yml/badge.svg)](https://github.com/andrewimpellitteri/creativity_bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An evaluation suite for measuring the creative capabilities of large language models, based on [Gwern's creative-benchmark proposals](https://gwern.net/creative-benchmark).

Works with any OpenAI-compatible API: OpenAI, DeepSeek, z.ai (GLM), OpenRouter, or a custom endpoint.

## Benchmark status

This is an exploratory suite, not a validated measure of general creativity.
See [the design audit and implementation roadmap](BENCHMARK_DESIGN.md) for scoring
failure modes, proposed controls, and the next experiment, and
[the workboard](WORKBOARD.md) for what is open right now and in what order. Compare task profiles
and inspect outputs before interpreting the composite.

## The tasks

Every task produces a descriptive score in **[0, 1]**. The composite remains an
exploratory weighted mean, not a calibrated creativity scale. Protocol `0.5-coverage`
adds the tasks that were still missing from Gwern's single-model list
(This & That, This & That—But Not Like That, Copycat, Quilting) without
changing how the `0.4-validity` tasks
score; per-task numbers stay methodologically comparable across the two, but
composites do not, because the roster they average over changed. Earlier
protocols changed scoring itself and are not comparable at all.

| Task | What it measures | Score |
|------|------------------|-------|
| **Same But Different** | Sustained plot diversity under a fixed premise | Accepted distinct, valid stories / scheduled attempts; full acceptance curves and judgments saved |
| **Free association** | Spontaneous association with full word history | Fraction of budget before first repetition or invalid response; raw vocabulary diagnostics retained |
| **Telephone game** | Creative drift: expand a summary into a story, re-summarize, repeat | Fraction of iterations before successive stories converge |
| **Camel's back** | Coherence under stacked edits: apply 1–3 random edits per round, an LLM judge verifies coherence | Fraction of edit rounds survived |
| **Diversity (DRY)** | Variation across repeated identical prompts | Within-prompt mean cosine distance / 2; between-prompt distance and effective rank are diagnostics |
| **Style transfer** | Genre transformation: summarize a story, rewrite it in a different genre | Cosine distance / 2, gated on plot preservation, target genre and comprehensibility |
| **This & that** | Blending two unlike examples into one story that is like both | Angular interpolation excess against an unrelated baseline story, gated on a judge confirming the story draws on both examples |
| **This & that—but not like that** | Negation: follow a good example while ending up further from a designated bad example than the good one already is | Share of the pair's remaining angular headroom away from the bad example, gated on a judge confirming the story still draws on the good example and is comprehensible |
| **Copycat** (LLM-uta) | Holding a borrowed voice instead of collapsing to a house style | Chance-corrected accuracy of a blinded judge matching each continuation back to its opening |
| **Quilting** | Recipe variety: pick fragments from a shuffled pile, then use them | Validity-gated mean of distinct-subset rate and story embedding diversity across runs |
| **Odd one out** | Anti-anchoring: given themed example items, name the most different item that still belongs to the category | Mean per-list minimum embedding distance to the examples (cosine [0, 2] halved into [0, 1]); the runner requires membership judgments; non-members earn zero and unresolved judgments mark the run incomplete |
| **Subversion** | Negation: write "the opposite" of a generated story; a judge classifies every story/subversion pair as opposite or not | Within-pair hit rate minus cross-pair false-positive rate (Youden's J) |
| **Shaggy dog** | Non-moralizing: write a deliberately pointless story, judges then name its moral | Inverted judge agreement (divergent morals score high; stated moral fails outright) |

## Setup

Requires Python ≥ 3.10.

```bash
uv sync          # or: pip install -e .
```

Set the API key for whichever provider you use:

| Provider | Flag | Key env var | Base URL |
|----------|------|-------------|----------|
| OpenAI | `--provider openai` | `OPENAI_API_KEY` | api.openai.com |
| DeepSeek | `--provider deepseek` | `DEEPSEEK_API_KEY` | api.deepseek.com |
| z.ai (API credit) | `--provider zai` | `ZAI_API_KEY` | api.z.ai/api/paas/v4 |
| z.ai (GLM Coding Plan) | `--provider zai-coding` | `ZAI_API_KEY` | api.z.ai/api/coding/paas/v4 |
| OpenRouter | `--provider openrouter` | `OPENROUTER_API_KEY` | openrouter.ai/api/v1 |
| Anything else | `--provider custom --base-url URL` | `LLM_API_KEY` | your URL |

Embeddings default to OpenAI `text-embedding-3-small` (very cheap), so `OPENAI_API_KEY` is needed for the embedding-based tasks even when benchmarking a GLM model. Override with `--embed-provider` / `--embed-model`.

## Usage

```bash
# Cheap smoke run first
uv run creativity-bench run --model gpt-5-mini --fast

# Full run against OpenAI
uv run creativity-bench run --model gpt-5-mini --n 3 --seed 0

# Benchmark GLM on the coding plan, judged by a fixed OpenAI model
export ZAI_API_KEY=... OPENAI_API_KEY=...
uv run creativity-bench run --provider zai-coding --model glm-4.6 \
    --judge-model gpt-5-mini --judge-provider openai --n 3 --seed 0

# Only some tasks
uv run creativity-bench run --model gpt-5-mini --tasks diversity,style_transfer

# Free OpenRouter models (e.g. stealth/ox-alpha) cost $0
uv run creativity-bench run --provider openrouter --model stealth/ox-alpha --n 3 --seed 0

# Plot all saved runs
uv run creativity-bench viz

# Write a markdown leaderboard (and chart) from saved runs
uv run creativity-bench report --runs-dir runs --out results/leaderboard.md \
    --chart results/model_comparison.png
```

Results are written to `runs/*.json` with full transcripts, per-task metrics, token usage, and the seed for reproducibility.

### Start with the focused pilot

```bash
# Same But Different uses no embedding service. Choose and pin your judge.
uv run creativity-bench run --model MODEL --judge-model JUDGE \
    --tasks same_but_different --fast --seed 0 --runs-dir runs/pilot

# Inspect a saved run's stories, acceptance curves and judgment evidence.
uv run creativity-bench gallery --run runs/pilot/RUN.json --out results/gallery.html

# Exercise the actual plot judge on labeled development controls.
uv run creativity-bench validate-judge --judge-model JUDGE \
    --out results/judge_validation.json

# Larger pilot: 6 of 20 public premises, 10 attempts each, per seed.
uv run creativity-bench run --model MODEL --judge-model JUDGE \
    --tasks same_but_different --n 5 --seed 0 --runs-dir runs/pilot-full

uv run creativity-bench report --runs-dir runs/pilot-full --out results/profile.md
```

Default controls include copies, paraphrases, name substitutions, a distinct
valid plot, unrelated prose, nonsense, a premise violation and an instruction
attack. Their labels are proposed development labels, **not independent human
validation**. Supply a JSON array through `validate-judge --controls FILE` for
reviewed labels and a held-out split; see [the design](BENCHMARK_DESIGN.md).
The validation command makes real judge API calls.

### Comparing models fairly

- Pin judge and embedder identities, providers, and sampling policies. The runner
  fingerprints scoring code, prompts and data, and logs actual generation settings
  and finish reasons. API outputs are not deterministic merely because a benchmark
  seed is fixed. Backend aliases can also change without changing their name.
- Reports split incompatible protocols, task subsets, budgets and judge settings
  into separate cohorts. Legacy provenance is labeled unverified. Charts require
  one verified cohort. Runs with unresolved judgments or generation errors are
  excluded from comparative reports, while their audit data remains saved.
- Compare models on matching seeds. Reports bootstrap paired **seed-level** task
  differences, averaging duplicate runs of a seed before pairing. Intervals need
  at least two matched seeds and are descriptive; small samples remain weak evidence.
- Inspect task profiles and stories. Embedding geometry, judge acceptance and
  creative achievement are different quantities. Several legacy tasks remain
  exploratory (see the task audit).

### Running the multi-model cohort

`scripts/run_multimodel.sh` runs the five non-embedding tasks at fast sizes for
the 12-model roster (DeepSeek, z.ai GLM variants, OpenRouter value models) on
paired seeds 0–1, with the judge fixed to the validated `deepseek-v4-pro` for
every writer. It sources `.env` and is idempotent: any (model, seed) pair
already saved in the runs directory is skipped, so re-running fills in missing
seeds without duplicating work. OpenRouter models are skipped unless
`OPENROUTER_API_KEY` is set.

```bash
scripts/run_multimodel.sh
uv run creativity-bench report --runs-dir results/extended-20260915/suite_runs \
    --out results/extended-20260915/REPORT.md \
    --chart results/extended-20260915/BENCHMARK_GRAPH.png
```

The cohort lives in [`results/extended-20260915/`](results/extended-20260915/):
`PROTOCOL.md` pre-registers the run, `FINDINGS.md` and `REPORT.md` summarize it,
and `BENCHMARK_GRAPH.png` charts the single verified cohort (a same-basename
`.svg` vector copy is written alongside every PNG for publications). Incomplete
runs are quarantined into `results/*/incomplete/` instead of deleted, so their
audit data survives while comparative reports exclude them.

Two gates apply before any pilot: `validate-judge` must resolve and correctly
label all nine development controls (a failure or unresolved verdict blocks the
run), and the blinded review packet from
`scripts/export_human_review.py` — a shuffled, leak-scanned export where story
identities stay in a `private_key.json` that must never be shared — awaits
independent human labels.

## Results

Development results from this repo's own pilots — a demonstration of the
tooling, not a validated measurement.

### Same But Different pilot (protocol `0.4-validity`)

Matched fast pilot (2 premises × 3 attempts, seeds 0–1) with the judge pinned
to `deepseek-v4-pro`, which passed all 8 development controls (100% resolution
and accuracy on premise adherence, comprehensibility and plot distinctness).
The judge also graded its own runs, so self-preference bias is possible and
unquantified.

| Model | Seed 0 | Seed 1 | Mean |
|---|---|---|---|
| `deepseek-flash` | 1.000 | 1.000 | 1.000 |
| `deepseek-v4-pro` | 0.833 | 1.000 | 0.917 |

`deepseek-v4-pro`'s single rejection is legible in the saved evidence: the
story was adherent and comprehensible but reused an accepted story's
resolution arc (a family keepsake standing in for the whole home) with new
surface details. Stories, verdicts and per-premise acceptance curves are in
[`results/pilot-20260915/`](results/pilot-20260915/leaderboard.md), with an
inspectable gallery per run
([example](results/pilot-20260915/gallery/deepseek-v4-pro_20260915-161839_a4a04a.html)).

### Legacy 8-task profiles (pre-protocol audit)

The full legacy suite ran once per model on the pre-0.4 scoring code
(`deepseek-chat` 0.567, `gpt-4o-mini` 0.535, `gpt-5-mini` 0.625 exploratory
composites, shared `gpt-5-mini` judge). Provenance is unverified under
protocol 0.4, so the report refuses to pool or chart them: each profile is
shown separately in
[`results/legacy-leaderboard.md`](results/legacy-leaderboard.md).

## Cost and budgets

`--fast` is a smoke test, not comparable to full runs. Same But Different uses
6 generation attempts in fast mode (2 premises × 3) and 60 in full mode
(6 × 10), plus up to three judge requests per eligible attempt. The tasks added
in `0.5-coverage` are comparatively cheap: This & That is 1 generation + up to 2
judge calls per pair (fast 1 pair, full 3), Copycat is k generations + up to 2k
judge calls (fast k=3, full k=5), and Quilting is 1 generation + up to 2 judge
calls per run (fast 2 runs, full 4). Copycat needs no embedder; This & That and
Quilting embed 4 and `valid_runs` texts respectively. Exact duplicates
skip judging. Diversity now generates two stories per prompt. Prices depend on
your providers; no cost estimate is inferred from token counts. Saved usage is
per run, with actual token budgets and finish reasons recorded for generation
and judgment calls. Client retries may add requests and expand empty reasoning
responses up to 16,000 tokens; this policy is part of protocol provenance.

## Development

```bash
uv run pytest
```

The test suite runs entirely offline against fake clients.

## Contributing

PRs and issues welcome. The thirteen tasks above cover Gwern's iteration,
style-flexibility and difference/negation groups. Still unimplemented from the
post: Rubric Writing, the Thematic Apperception Test, the Fermi Problem
Contest, Worldbuilding / Fanfic Fantasizing, and the whole multi-agent family (Star Chameleon, Exquisite Corpse, Copycat: Truesight,
Style Laboratory, multi-agent Free Association). See
[the workboard](WORKBOARD.md) for those and for the open validation work.
Results from more models are appreciated.

## License

MIT
