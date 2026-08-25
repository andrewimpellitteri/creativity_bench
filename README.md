# LLM Creativity Benchmark

An evaluation suite for measuring the creative capabilities of large language models, based on [Gwern's creative-benchmark proposals](https://gwern.net/creative-benchmark).

Works with any OpenAI-compatible API: OpenAI, z.ai (GLM), OpenRouter, or a custom endpoint.

## The tasks

Every task produces a score in **[0, 1]**; the composite is a weighted mean (equal weights by default).

| Task | What it measures | Score |
|------|------------------|-------|
| **Free association** | Novelty under memory: the model sees its full word history and must never repeat itself | Fraction of unique words (Chao1 vocabulary estimate reported) |
| **Telephone game** | Creative drift: expand a summary into a story, re-summarize, repeat | Fraction of iterations before summaries collapse to a fixed point |
| **Camel's back** | Coherence under stacked edits: apply 1–3 random edits per round, an LLM judge verifies coherence | Fraction of edit rounds survived |
| **Diversity (DRY)** | Spread of outputs across similar prompts | Mean pairwise embedding distance between generated stories |
| **Style transfer** | Genre transformation: summarize a story, rewrite it in a different genre | Mean embedding divergence from the original (fidelity to the summary reported alongside) |
| **Odd one out** | Anti-anchoring: given themed example items, name the most different item that still belongs to the category | Mean per-list minimum embedding distance to the examples (cosine [0, 2] halved into [0, 1]); an optional LLM judge zeroes non-members |
| **Subversion** | Negation: write "the opposite" of a generated story; a judge classifies every story/subversion pair as opposite or not | Within-pair hit rate minus cross-pair false-positive rate (Youden's J) |

## Setup

Requires Python ≥ 3.10.

```bash
uv sync          # or: pip install -e .
```

Set the API key for whichever provider you use:

| Provider | Flag | Key env var | Base URL |
|----------|------|-------------|----------|
| OpenAI | `--provider openai` | `OPENAI_API_KEY` | api.openai.com |
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
```

Results are written to `runs/*.json` with full transcripts, per-task metrics, token usage, and the seed for reproducibility.

### Comparing models fairly

- **Pin the judge**: pass the same `--judge-model` for every model you compare, otherwise each model grades its own camel's-back edits.
- **Pin the embedder**: keep `--embed-model` identical across runs; embedding-based scores are only comparable within one embedding space.
- **Repeat runs**: use `--n 3` (or more) and a fixed `--seed`; the viz shows standard deviation as error bars.

## Cost

A full run is roughly 100–150 generation requests plus ~40 small embedding calls. With a mini-tier model that is a few cents per run; `--fast` cuts task sizes by ~3× for smoke testing. On OpenRouter, models that are not known to be free (no `:free` suffix, not `stealth/ox-alpha` or `openrouter/free`) trigger a warning since they will spend credits.

## Development

```bash
uv run pytest
```

The test suite runs entirely offline against fake clients.

## Contributing

PRs and issues welcome — the remaining benchmarks from Gwern's post are open, and results from more models are appreciated.

## License

MIT
