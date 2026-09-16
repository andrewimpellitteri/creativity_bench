# Creativity Bench — Leaderboard

Generated 2026-09-15 from 20 runs of 12 models in `results/extended-20260915/suite_runs/`.
Excluded 0 incomplete evaluations: unresolved judgments or generation errors yield audit lower bounds, not comparable creativity scores.
Task profiles are primary. The composite is exploratory: its weighting has not been validated as a measure of creativity. Models are listed alphabetically.
Different protocol cohorts are not directly comparable; no cross-cohort ranking is made.

## Cohort 1 — verified provenance

Protocol: `0.4-validity`; fast: `True`; judge: `deepseek-v4-pro`.

| Model | Same but different | Free association | Telephone game | Camel's back | Diversity | Style transfer | Odd one out | Subversion | Shaggy dog | Exploratory composite ± SD | n runs | Seeds | Judge | Runs from |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `anthropic/claude-haiku-4.5` ⚡ | 0.667 | 1.000 | — | 1.000 | — | — | — | 1.000 | 0.669 | 0.867 ± 0.047 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-16 |
| `deepseek-flash` ⚡ | 0.917 | 1.000 | — | 1.000 | — | — | — | 1.000 | 0.628 | 0.909 ± 0.019 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-15 |
| `deepseek-v4-pro` ⚡ | 1.000 | 1.000 | — | 1.000 | — | — | — | 1.000 | 0.655 | 0.931 ± 0.004 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-15 |
| `glm-4.5-air` ⚡ | 0.833 | 1.000 | — | 1.000 | — | — | — | 1.000 | 0.669 | 0.900 ± 0.000 | 1 | 0 | deepseek-v4-pro | 2026-09-16 |
| `glm-4.6` ⚡ | 0.833 | 1.000 | — | 1.000 | — | — | — | 0.500 | 0.809 | 0.829 ± 0.000 | 1 | 0 | deepseek-v4-pro | 2026-09-15 |
| `glm-5-turbo` ⚡ | 0.667 | 1.000 | — | 1.000 | — | — | — | 0.500 | 0.773 | 0.788 ± 0.000 | 1 | 0 | deepseek-v4-pro | 2026-09-16 |
| `glm-5.3-flash` ⚡ | 1.000 | 1.000 | — | 1.000 | — | — | — | 1.000 | 0.752 | 0.950 ± 0.000 | 1 | 0 | deepseek-v4-pro | 2026-09-15 |
| `google/gemini-2.5-flash` ⚡ | 0.500 | 1.000 | — | 0.833 | — | — | — | 0.750 | 0.742 | 0.765 ± 0.030 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-16 |
| `meta-llama/llama-4-scout` ⚡ | 0.417 | 1.000 | — | 0.500 | — | — | — | 0.750 | 0.752 | 0.684 ± 0.138 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-16 |
| `mistralai/mistral-small-3.2-24b-instruct` ⚡ | 0.500 | 0.850 | — | 0.833 | — | — | — | 0.750 | 0.663 | 0.719 ± 0.006 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-16 |
| `moonshotai/kimi-k2.5` ⚡ | 0.917 | 1.000 | — | 0.500 | — | — | — | 1.000 | 0.715 | 0.826 ± 0.125 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-16 |
| `openai/gpt-4o-mini` ⚡ | 0.333 | 1.000 | — | 0.833 | — | — | — | 0.750 | 0.657 | 0.715 ± 0.096 | 2 | 0, 1 | deepseek-v4-pro | 2026-09-16 |

### Paired task differences

Differences are left minus right; percentile bootstrap 95% intervals resample matched seeds. Duplicate seed runs are averaged first. Intervals are descriptive, without multiple-comparison correction.

| Left - right | Task | Matched seeds | Difference | 95% interval |
|---|---|---:|---:|---|
| `anthropic/claude-haiku-4.5` - `deepseek-flash` | same_but_different | 2 | -0.250 | [-0.500, 0.000] |
| `anthropic/claude-haiku-4.5` - `deepseek-flash` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `deepseek-flash` | camels_back | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `deepseek-flash` | subversion | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `deepseek-flash` | shaggy_dog | 2 | 0.041 | [-0.038, 0.121] |
| `anthropic/claude-haiku-4.5` - `deepseek-v4-pro` | same_but_different | 2 | -0.333 | [-0.500, -0.167] |
| `anthropic/claude-haiku-4.5` - `deepseek-v4-pro` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `deepseek-v4-pro` | camels_back | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `deepseek-v4-pro` | subversion | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `deepseek-v4-pro` | shaggy_dog | 2 | 0.015 | [-0.070, 0.099] |
| `anthropic/claude-haiku-4.5` - `glm-4.5-air` | same_but_different | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-4.5-air` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-4.5-air` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-4.5-air` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-4.5-air` | shaggy_dog | 1 | 0.067 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-4.6` | same_but_different | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-4.6` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-4.6` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-4.6` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-4.6` | shaggy_dog | 1 | -0.074 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5-turbo` | same_but_different | 1 | 0.167 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5-turbo` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5-turbo` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5-turbo` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5-turbo` | shaggy_dog | 1 | -0.037 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5.3-flash` | same_but_different | 1 | -0.167 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5.3-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5.3-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5.3-flash` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `glm-5.3-flash` | shaggy_dog | 1 | -0.016 | — (need ≥2 matched seeds) |
| `anthropic/claude-haiku-4.5` - `google/gemini-2.5-flash` | same_but_different | 2 | 0.167 | [0.000, 0.333] |
| `anthropic/claude-haiku-4.5` - `google/gemini-2.5-flash` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `google/gemini-2.5-flash` | camels_back | 2 | 0.167 | [0.000, 0.333] |
| `anthropic/claude-haiku-4.5` - `google/gemini-2.5-flash` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `anthropic/claude-haiku-4.5` - `google/gemini-2.5-flash` | shaggy_dog | 2 | -0.072 | [-0.203, 0.058] |
| `anthropic/claude-haiku-4.5` - `meta-llama/llama-4-scout` | same_but_different | 2 | 0.250 | [0.000, 0.500] |
| `anthropic/claude-haiku-4.5` - `meta-llama/llama-4-scout` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `meta-llama/llama-4-scout` | camels_back | 2 | 0.500 | [0.000, 1.000] |
| `anthropic/claude-haiku-4.5` - `meta-llama/llama-4-scout` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `anthropic/claude-haiku-4.5` - `meta-llama/llama-4-scout` | shaggy_dog | 2 | -0.083 | [-0.126, -0.040] |
| `anthropic/claude-haiku-4.5` - `mistralai/mistral-small-3.2-24b-instruct` | same_but_different | 2 | 0.167 | [0.000, 0.333] |
| `anthropic/claude-haiku-4.5` - `mistralai/mistral-small-3.2-24b-instruct` | free_association | 2 | 0.150 | [0.000, 0.300] |
| `anthropic/claude-haiku-4.5` - `mistralai/mistral-small-3.2-24b-instruct` | camels_back | 2 | 0.167 | [0.000, 0.333] |
| `anthropic/claude-haiku-4.5` - `mistralai/mistral-small-3.2-24b-instruct` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `anthropic/claude-haiku-4.5` - `mistralai/mistral-small-3.2-24b-instruct` | shaggy_dog | 2 | 0.006 | [-0.098, 0.111] |
| `anthropic/claude-haiku-4.5` - `moonshotai/kimi-k2.5` | same_but_different | 2 | -0.250 | [-0.333, -0.167] |
| `anthropic/claude-haiku-4.5` - `moonshotai/kimi-k2.5` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `moonshotai/kimi-k2.5` | camels_back | 2 | 0.500 | [0.000, 1.000] |
| `anthropic/claude-haiku-4.5` - `moonshotai/kimi-k2.5` | subversion | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `moonshotai/kimi-k2.5` | shaggy_dog | 2 | -0.045 | [-0.070, -0.021] |
| `anthropic/claude-haiku-4.5` - `openai/gpt-4o-mini` | same_but_different | 2 | 0.333 | [0.167, 0.500] |
| `anthropic/claude-haiku-4.5` - `openai/gpt-4o-mini` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `anthropic/claude-haiku-4.5` - `openai/gpt-4o-mini` | camels_back | 2 | 0.167 | [0.000, 0.333] |
| `anthropic/claude-haiku-4.5` - `openai/gpt-4o-mini` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `anthropic/claude-haiku-4.5` - `openai/gpt-4o-mini` | shaggy_dog | 2 | 0.012 | [0.009, 0.016] |
| `deepseek-flash` - `deepseek-v4-pro` | same_but_different | 2 | -0.083 | [-0.167, 0.000] |
| `deepseek-flash` - `deepseek-v4-pro` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-flash` - `deepseek-v4-pro` | camels_back | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-flash` - `deepseek-v4-pro` | subversion | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-flash` - `deepseek-v4-pro` | shaggy_dog | 2 | -0.027 | [-0.032, -0.021] |
| `deepseek-flash` - `glm-4.5-air` | same_but_different | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-4.5-air` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-4.5-air` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-4.5-air` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-4.5-air` | shaggy_dog | 1 | -0.054 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-4.6` | same_but_different | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-4.6` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-4.6` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-4.6` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-4.6` | shaggy_dog | 1 | -0.194 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5-turbo` | same_but_different | 1 | 0.167 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5-turbo` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5-turbo` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5-turbo` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5-turbo` | shaggy_dog | 1 | -0.158 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5.3-flash` | same_but_different | 1 | -0.167 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5.3-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5.3-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5.3-flash` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `glm-5.3-flash` | shaggy_dog | 1 | -0.137 | — (need ≥2 matched seeds) |
| `deepseek-flash` - `google/gemini-2.5-flash` | same_but_different | 2 | 0.417 | [0.333, 0.500] |
| `deepseek-flash` - `google/gemini-2.5-flash` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-flash` - `google/gemini-2.5-flash` | camels_back | 2 | 0.167 | [0.000, 0.333] |
| `deepseek-flash` - `google/gemini-2.5-flash` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `deepseek-flash` - `google/gemini-2.5-flash` | shaggy_dog | 2 | -0.113 | [-0.165, -0.062] |
| `deepseek-flash` - `meta-llama/llama-4-scout` | same_but_different | 2 | 0.500 | [0.500, 0.500] |
| `deepseek-flash` - `meta-llama/llama-4-scout` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-flash` - `meta-llama/llama-4-scout` | camels_back | 2 | 0.500 | [0.000, 1.000] |
| `deepseek-flash` - `meta-llama/llama-4-scout` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `deepseek-flash` - `meta-llama/llama-4-scout` | shaggy_dog | 2 | -0.124 | [-0.160, -0.088] |
| `deepseek-flash` - `mistralai/mistral-small-3.2-24b-instruct` | same_but_different | 2 | 0.417 | [0.333, 0.500] |
| `deepseek-flash` - `mistralai/mistral-small-3.2-24b-instruct` | free_association | 2 | 0.150 | [0.000, 0.300] |
| `deepseek-flash` - `mistralai/mistral-small-3.2-24b-instruct` | camels_back | 2 | 0.167 | [0.000, 0.333] |
| `deepseek-flash` - `mistralai/mistral-small-3.2-24b-instruct` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `deepseek-flash` - `mistralai/mistral-small-3.2-24b-instruct` | shaggy_dog | 2 | -0.035 | [-0.060, -0.010] |
| `deepseek-flash` - `moonshotai/kimi-k2.5` | same_but_different | 2 | 0.000 | [-0.167, 0.167] |
| `deepseek-flash` - `moonshotai/kimi-k2.5` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-flash` - `moonshotai/kimi-k2.5` | camels_back | 2 | 0.500 | [0.000, 1.000] |
| `deepseek-flash` - `moonshotai/kimi-k2.5` | subversion | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-flash` - `moonshotai/kimi-k2.5` | shaggy_dog | 2 | -0.086 | [-0.141, -0.032] |
| `deepseek-flash` - `openai/gpt-4o-mini` | same_but_different | 2 | 0.583 | [0.500, 0.667] |
| `deepseek-flash` - `openai/gpt-4o-mini` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-flash` - `openai/gpt-4o-mini` | camels_back | 2 | 0.167 | [0.000, 0.333] |
| `deepseek-flash` - `openai/gpt-4o-mini` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `deepseek-flash` - `openai/gpt-4o-mini` | shaggy_dog | 2 | -0.029 | [-0.105, 0.047] |
| `deepseek-v4-pro` - `glm-4.5-air` | same_but_different | 1 | 0.167 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-4.5-air` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-4.5-air` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-4.5-air` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-4.5-air` | shaggy_dog | 1 | -0.032 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-4.6` | same_but_different | 1 | 0.167 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-4.6` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-4.6` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-4.6` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-4.6` | shaggy_dog | 1 | -0.173 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5-turbo` | same_but_different | 1 | 0.333 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5-turbo` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5-turbo` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5-turbo` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5-turbo` | shaggy_dog | 1 | -0.137 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5.3-flash` | same_but_different | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5.3-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5.3-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5.3-flash` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `glm-5.3-flash` | shaggy_dog | 1 | -0.115 | — (need ≥2 matched seeds) |
| `deepseek-v4-pro` - `google/gemini-2.5-flash` | same_but_different | 2 | 0.500 | [0.500, 0.500] |
| `deepseek-v4-pro` - `google/gemini-2.5-flash` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-v4-pro` - `google/gemini-2.5-flash` | camels_back | 2 | 0.167 | [0.000, 0.333] |
| `deepseek-v4-pro` - `google/gemini-2.5-flash` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `deepseek-v4-pro` - `google/gemini-2.5-flash` | shaggy_dog | 2 | -0.087 | [-0.133, -0.041] |
| `deepseek-v4-pro` - `meta-llama/llama-4-scout` | same_but_different | 2 | 0.583 | [0.500, 0.667] |
| `deepseek-v4-pro` - `meta-llama/llama-4-scout` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-v4-pro` - `meta-llama/llama-4-scout` | camels_back | 2 | 0.500 | [0.000, 1.000] |
| `deepseek-v4-pro` - `meta-llama/llama-4-scout` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `deepseek-v4-pro` - `meta-llama/llama-4-scout` | shaggy_dog | 2 | -0.097 | [-0.139, -0.056] |
| `deepseek-v4-pro` - `mistralai/mistral-small-3.2-24b-instruct` | same_but_different | 2 | 0.500 | [0.500, 0.500] |
| `deepseek-v4-pro` - `mistralai/mistral-small-3.2-24b-instruct` | free_association | 2 | 0.150 | [0.000, 0.300] |
| `deepseek-v4-pro` - `mistralai/mistral-small-3.2-24b-instruct` | camels_back | 2 | 0.167 | [0.000, 0.333] |
| `deepseek-v4-pro` - `mistralai/mistral-small-3.2-24b-instruct` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `deepseek-v4-pro` - `mistralai/mistral-small-3.2-24b-instruct` | shaggy_dog | 2 | -0.008 | [-0.028, 0.011] |
| `deepseek-v4-pro` - `moonshotai/kimi-k2.5` | same_but_different | 2 | 0.083 | [0.000, 0.167] |
| `deepseek-v4-pro` - `moonshotai/kimi-k2.5` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-v4-pro` - `moonshotai/kimi-k2.5` | camels_back | 2 | 0.500 | [0.000, 1.000] |
| `deepseek-v4-pro` - `moonshotai/kimi-k2.5` | subversion | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-v4-pro` - `moonshotai/kimi-k2.5` | shaggy_dog | 2 | -0.060 | [-0.120, 0.000] |
| `deepseek-v4-pro` - `openai/gpt-4o-mini` | same_but_different | 2 | 0.667 | [0.667, 0.667] |
| `deepseek-v4-pro` - `openai/gpt-4o-mini` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `deepseek-v4-pro` - `openai/gpt-4o-mini` | camels_back | 2 | 0.167 | [0.000, 0.333] |
| `deepseek-v4-pro` - `openai/gpt-4o-mini` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `deepseek-v4-pro` - `openai/gpt-4o-mini` | shaggy_dog | 2 | -0.002 | [-0.084, 0.079] |
| `glm-4.5-air` - `glm-4.6` | same_but_different | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-4.6` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-4.6` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-4.6` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-4.6` | shaggy_dog | 1 | -0.141 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5-turbo` | same_but_different | 1 | 0.167 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5-turbo` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5-turbo` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5-turbo` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5-turbo` | shaggy_dog | 1 | -0.104 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5.3-flash` | same_but_different | 1 | -0.167 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5.3-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5.3-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5.3-flash` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `glm-5.3-flash` | shaggy_dog | 1 | -0.083 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `google/gemini-2.5-flash` | same_but_different | 1 | 0.333 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `google/gemini-2.5-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `google/gemini-2.5-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `google/gemini-2.5-flash` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `google/gemini-2.5-flash` | shaggy_dog | 1 | -0.008 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `meta-llama/llama-4-scout` | same_but_different | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `meta-llama/llama-4-scout` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `meta-llama/llama-4-scout` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `meta-llama/llama-4-scout` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `meta-llama/llama-4-scout` | shaggy_dog | 1 | -0.107 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `mistralai/mistral-small-3.2-24b-instruct` | same_but_different | 1 | 0.333 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `mistralai/mistral-small-3.2-24b-instruct` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `mistralai/mistral-small-3.2-24b-instruct` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `mistralai/mistral-small-3.2-24b-instruct` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `mistralai/mistral-small-3.2-24b-instruct` | shaggy_dog | 1 | 0.044 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `moonshotai/kimi-k2.5` | same_but_different | 1 | -0.167 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `moonshotai/kimi-k2.5` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `moonshotai/kimi-k2.5` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `moonshotai/kimi-k2.5` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `moonshotai/kimi-k2.5` | shaggy_dog | 1 | -0.087 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `openai/gpt-4o-mini` | same_but_different | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `openai/gpt-4o-mini` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `openai/gpt-4o-mini` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `openai/gpt-4o-mini` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.5-air` - `openai/gpt-4o-mini` | shaggy_dog | 1 | -0.051 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5-turbo` | same_but_different | 1 | 0.167 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5-turbo` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5-turbo` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5-turbo` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5-turbo` | shaggy_dog | 1 | 0.036 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5.3-flash` | same_but_different | 1 | -0.167 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5.3-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5.3-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5.3-flash` | subversion | 1 | -0.500 | — (need ≥2 matched seeds) |
| `glm-4.6` - `glm-5.3-flash` | shaggy_dog | 1 | 0.058 | — (need ≥2 matched seeds) |
| `glm-4.6` - `google/gemini-2.5-flash` | same_but_different | 1 | 0.333 | — (need ≥2 matched seeds) |
| `glm-4.6` - `google/gemini-2.5-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `google/gemini-2.5-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `google/gemini-2.5-flash` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `google/gemini-2.5-flash` | shaggy_dog | 1 | 0.132 | — (need ≥2 matched seeds) |
| `glm-4.6` - `meta-llama/llama-4-scout` | same_but_different | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-4.6` - `meta-llama/llama-4-scout` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `meta-llama/llama-4-scout` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `meta-llama/llama-4-scout` | subversion | 1 | -0.500 | — (need ≥2 matched seeds) |
| `glm-4.6` - `meta-llama/llama-4-scout` | shaggy_dog | 1 | 0.034 | — (need ≥2 matched seeds) |
| `glm-4.6` - `mistralai/mistral-small-3.2-24b-instruct` | same_but_different | 1 | 0.333 | — (need ≥2 matched seeds) |
| `glm-4.6` - `mistralai/mistral-small-3.2-24b-instruct` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `mistralai/mistral-small-3.2-24b-instruct` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `mistralai/mistral-small-3.2-24b-instruct` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `mistralai/mistral-small-3.2-24b-instruct` | shaggy_dog | 1 | 0.185 | — (need ≥2 matched seeds) |
| `glm-4.6` - `moonshotai/kimi-k2.5` | same_but_different | 1 | -0.167 | — (need ≥2 matched seeds) |
| `glm-4.6` - `moonshotai/kimi-k2.5` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `moonshotai/kimi-k2.5` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `moonshotai/kimi-k2.5` | subversion | 1 | -0.500 | — (need ≥2 matched seeds) |
| `glm-4.6` - `moonshotai/kimi-k2.5` | shaggy_dog | 1 | 0.053 | — (need ≥2 matched seeds) |
| `glm-4.6` - `openai/gpt-4o-mini` | same_but_different | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-4.6` - `openai/gpt-4o-mini` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `openai/gpt-4o-mini` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-4.6` - `openai/gpt-4o-mini` | subversion | 1 | -0.500 | — (need ≥2 matched seeds) |
| `glm-4.6` - `openai/gpt-4o-mini` | shaggy_dog | 1 | 0.090 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `glm-5.3-flash` | same_but_different | 1 | -0.333 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `glm-5.3-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `glm-5.3-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `glm-5.3-flash` | subversion | 1 | -0.500 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `glm-5.3-flash` | shaggy_dog | 1 | 0.021 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `google/gemini-2.5-flash` | same_but_different | 1 | 0.167 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `google/gemini-2.5-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `google/gemini-2.5-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `google/gemini-2.5-flash` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `google/gemini-2.5-flash` | shaggy_dog | 1 | 0.096 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `meta-llama/llama-4-scout` | same_but_different | 1 | 0.333 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `meta-llama/llama-4-scout` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `meta-llama/llama-4-scout` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `meta-llama/llama-4-scout` | subversion | 1 | -0.500 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `meta-llama/llama-4-scout` | shaggy_dog | 1 | -0.002 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `mistralai/mistral-small-3.2-24b-instruct` | same_but_different | 1 | 0.167 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `mistralai/mistral-small-3.2-24b-instruct` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `mistralai/mistral-small-3.2-24b-instruct` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `mistralai/mistral-small-3.2-24b-instruct` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `mistralai/mistral-small-3.2-24b-instruct` | shaggy_dog | 1 | 0.148 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `moonshotai/kimi-k2.5` | same_but_different | 1 | -0.333 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `moonshotai/kimi-k2.5` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `moonshotai/kimi-k2.5` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `moonshotai/kimi-k2.5` | subversion | 1 | -0.500 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `moonshotai/kimi-k2.5` | shaggy_dog | 1 | 0.017 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `openai/gpt-4o-mini` | same_but_different | 1 | 0.333 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `openai/gpt-4o-mini` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `openai/gpt-4o-mini` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `openai/gpt-4o-mini` | subversion | 1 | -0.500 | — (need ≥2 matched seeds) |
| `glm-5-turbo` - `openai/gpt-4o-mini` | shaggy_dog | 1 | 0.053 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `google/gemini-2.5-flash` | same_but_different | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `google/gemini-2.5-flash` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `google/gemini-2.5-flash` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `google/gemini-2.5-flash` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `google/gemini-2.5-flash` | shaggy_dog | 1 | 0.074 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `meta-llama/llama-4-scout` | same_but_different | 1 | 0.667 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `meta-llama/llama-4-scout` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `meta-llama/llama-4-scout` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `meta-llama/llama-4-scout` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `meta-llama/llama-4-scout` | shaggy_dog | 1 | -0.024 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `mistralai/mistral-small-3.2-24b-instruct` | same_but_different | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `mistralai/mistral-small-3.2-24b-instruct` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `mistralai/mistral-small-3.2-24b-instruct` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `mistralai/mistral-small-3.2-24b-instruct` | subversion | 1 | 0.500 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `mistralai/mistral-small-3.2-24b-instruct` | shaggy_dog | 1 | 0.127 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `moonshotai/kimi-k2.5` | same_but_different | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `moonshotai/kimi-k2.5` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `moonshotai/kimi-k2.5` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `moonshotai/kimi-k2.5` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `moonshotai/kimi-k2.5` | shaggy_dog | 1 | -0.004 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `openai/gpt-4o-mini` | same_but_different | 1 | 0.667 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `openai/gpt-4o-mini` | free_association | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `openai/gpt-4o-mini` | camels_back | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `openai/gpt-4o-mini` | subversion | 1 | 0.000 | — (need ≥2 matched seeds) |
| `glm-5.3-flash` - `openai/gpt-4o-mini` | shaggy_dog | 1 | 0.032 | — (need ≥2 matched seeds) |
| `google/gemini-2.5-flash` - `meta-llama/llama-4-scout` | same_but_different | 2 | 0.083 | [0.000, 0.167] |
| `google/gemini-2.5-flash` - `meta-llama/llama-4-scout` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `google/gemini-2.5-flash` - `meta-llama/llama-4-scout` | camels_back | 2 | 0.333 | [0.000, 0.667] |
| `google/gemini-2.5-flash` - `meta-llama/llama-4-scout` | subversion | 2 | 0.000 | [-0.500, 0.500] |
| `google/gemini-2.5-flash` - `meta-llama/llama-4-scout` | shaggy_dog | 2 | -0.011 | [-0.098, 0.077] |
| `google/gemini-2.5-flash` - `mistralai/mistral-small-3.2-24b-instruct` | same_but_different | 2 | 0.000 | [0.000, 0.000] |
| `google/gemini-2.5-flash` - `mistralai/mistral-small-3.2-24b-instruct` | free_association | 2 | 0.150 | [0.000, 0.300] |
| `google/gemini-2.5-flash` - `mistralai/mistral-small-3.2-24b-instruct` | camels_back | 2 | 0.000 | [0.000, 0.000] |
| `google/gemini-2.5-flash` - `mistralai/mistral-small-3.2-24b-instruct` | subversion | 2 | 0.000 | [0.000, 0.000] |
| `google/gemini-2.5-flash` - `mistralai/mistral-small-3.2-24b-instruct` | shaggy_dog | 2 | 0.079 | [0.052, 0.105] |
| `google/gemini-2.5-flash` - `moonshotai/kimi-k2.5` | same_but_different | 2 | -0.417 | [-0.500, -0.333] |
| `google/gemini-2.5-flash` - `moonshotai/kimi-k2.5` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `google/gemini-2.5-flash` - `moonshotai/kimi-k2.5` | camels_back | 2 | 0.333 | [0.000, 0.667] |
| `google/gemini-2.5-flash` - `moonshotai/kimi-k2.5` | subversion | 2 | -0.250 | [-0.500, 0.000] |
| `google/gemini-2.5-flash` - `moonshotai/kimi-k2.5` | shaggy_dog | 2 | 0.027 | [-0.079, 0.133] |
| `google/gemini-2.5-flash` - `openai/gpt-4o-mini` | same_but_different | 2 | 0.167 | [0.167, 0.167] |
| `google/gemini-2.5-flash` - `openai/gpt-4o-mini` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `google/gemini-2.5-flash` - `openai/gpt-4o-mini` | camels_back | 2 | 0.000 | [0.000, 0.000] |
| `google/gemini-2.5-flash` - `openai/gpt-4o-mini` | subversion | 2 | 0.000 | [-0.500, 0.500] |
| `google/gemini-2.5-flash` - `openai/gpt-4o-mini` | shaggy_dog | 2 | 0.085 | [-0.043, 0.212] |
| `meta-llama/llama-4-scout` - `mistralai/mistral-small-3.2-24b-instruct` | same_but_different | 2 | -0.083 | [-0.167, 0.000] |
| `meta-llama/llama-4-scout` - `mistralai/mistral-small-3.2-24b-instruct` | free_association | 2 | 0.150 | [0.000, 0.300] |
| `meta-llama/llama-4-scout` - `mistralai/mistral-small-3.2-24b-instruct` | camels_back | 2 | -0.333 | [-0.667, 0.000] |
| `meta-llama/llama-4-scout` - `mistralai/mistral-small-3.2-24b-instruct` | subversion | 2 | 0.000 | [-0.500, 0.500] |
| `meta-llama/llama-4-scout` - `mistralai/mistral-small-3.2-24b-instruct` | shaggy_dog | 2 | 0.089 | [0.028, 0.150] |
| `meta-llama/llama-4-scout` - `moonshotai/kimi-k2.5` | same_but_different | 2 | -0.500 | [-0.667, -0.333] |
| `meta-llama/llama-4-scout` - `moonshotai/kimi-k2.5` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `meta-llama/llama-4-scout` - `moonshotai/kimi-k2.5` | camels_back | 2 | 0.000 | [0.000, 0.000] |
| `meta-llama/llama-4-scout` - `moonshotai/kimi-k2.5` | subversion | 2 | -0.250 | [-0.500, 0.000] |
| `meta-llama/llama-4-scout` - `moonshotai/kimi-k2.5` | shaggy_dog | 2 | 0.038 | [0.019, 0.056] |
| `meta-llama/llama-4-scout` - `openai/gpt-4o-mini` | same_but_different | 2 | 0.083 | [0.000, 0.167] |
| `meta-llama/llama-4-scout` - `openai/gpt-4o-mini` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `meta-llama/llama-4-scout` - `openai/gpt-4o-mini` | camels_back | 2 | -0.333 | [-0.667, 0.000] |
| `meta-llama/llama-4-scout` - `openai/gpt-4o-mini` | subversion | 2 | 0.000 | [0.000, 0.000] |
| `meta-llama/llama-4-scout` - `openai/gpt-4o-mini` | shaggy_dog | 2 | 0.095 | [0.055, 0.135] |
| `mistralai/mistral-small-3.2-24b-instruct` - `moonshotai/kimi-k2.5` | same_but_different | 2 | -0.417 | [-0.500, -0.333] |
| `mistralai/mistral-small-3.2-24b-instruct` - `moonshotai/kimi-k2.5` | free_association | 2 | -0.150 | [-0.300, 0.000] |
| `mistralai/mistral-small-3.2-24b-instruct` - `moonshotai/kimi-k2.5` | camels_back | 2 | 0.333 | [0.000, 0.667] |
| `mistralai/mistral-small-3.2-24b-instruct` - `moonshotai/kimi-k2.5` | subversion | 2 | -0.250 | [-0.500, 0.000] |
| `mistralai/mistral-small-3.2-24b-instruct` - `moonshotai/kimi-k2.5` | shaggy_dog | 2 | -0.052 | [-0.131, 0.028] |
| `mistralai/mistral-small-3.2-24b-instruct` - `openai/gpt-4o-mini` | same_but_different | 2 | 0.167 | [0.167, 0.167] |
| `mistralai/mistral-small-3.2-24b-instruct` - `openai/gpt-4o-mini` | free_association | 2 | -0.150 | [-0.300, 0.000] |
| `mistralai/mistral-small-3.2-24b-instruct` - `openai/gpt-4o-mini` | camels_back | 2 | 0.000 | [0.000, 0.000] |
| `mistralai/mistral-small-3.2-24b-instruct` - `openai/gpt-4o-mini` | subversion | 2 | 0.000 | [-0.500, 0.500] |
| `mistralai/mistral-small-3.2-24b-instruct` - `openai/gpt-4o-mini` | shaggy_dog | 2 | 0.006 | [-0.095, 0.107] |
| `moonshotai/kimi-k2.5` - `openai/gpt-4o-mini` | same_but_different | 2 | 0.583 | [0.500, 0.667] |
| `moonshotai/kimi-k2.5` - `openai/gpt-4o-mini` | free_association | 2 | 0.000 | [0.000, 0.000] |
| `moonshotai/kimi-k2.5` - `openai/gpt-4o-mini` | camels_back | 2 | -0.333 | [-0.667, 0.000] |
| `moonshotai/kimi-k2.5` - `openai/gpt-4o-mini` | subversion | 2 | 0.250 | [0.000, 0.500] |
| `moonshotai/kimi-k2.5` - `openai/gpt-4o-mini` | shaggy_dog | 2 | 0.058 | [0.036, 0.079] |

### Task diagnostics

Telephone survival is the fraction of chains not yet collapsed at the full round budget (right-censored chains count as surviving), per condition. Subversion sensitivity/specificity separate within-pair inversion hits from matched-negative false positives. Shaggy gate pass is the share of stories passing the comprehensibility gate. Means across this cohort's runs; — when a run lacks the metric.

| Model | Telephone survival, deterministic | Telephone survival, stochastic | Subversion sensitivity | Subversion specificity | Shaggy gate pass |
|---|---:|---:|---:|---:|---:|
| `anthropic/claude-haiku-4.5` | — | — | 1.000 | 1.000 | 1.000 |
| `deepseek-flash` | — | — | 1.000 | 1.000 | 1.000 |
| `deepseek-v4-pro` | — | — | 1.000 | 1.000 | 1.000 |
| `glm-4.5-air` | — | — | 1.000 | 1.000 | 1.000 |
| `glm-4.6` | — | — | 1.000 | 0.500 | 1.000 |
| `glm-5-turbo` | — | — | 1.000 | 0.500 | 1.000 |
| `glm-5.3-flash` | — | — | 1.000 | 1.000 | 1.000 |
| `google/gemini-2.5-flash` | — | — | 1.000 | 0.750 | 1.000 |
| `meta-llama/llama-4-scout` | — | — | 0.750 | 1.000 | 1.000 |
| `mistralai/mistral-small-3.2-24b-instruct` | — | — | 0.750 | 1.000 | 1.000 |
| `moonshotai/kimi-k2.5` | — | — | 1.000 | 1.000 | 1.000 |
| `openai/gpt-4o-mini` | — | — | 0.750 | 1.000 | 1.000 |

## Notes

- ⚡ denotes fast task budgets. Cohorts split by protocol, tasks, weights, budgets, judge, embedding and generation settings.
- Legacy or incomplete provenance cannot establish compatibility; these runs are shown separately by model and are excluded from paired inference.
- Profile means weight saved runs equally; paired differences weight matched seeds equally after averaging duplicates, so their differences may differ.
- SD describes variation across saved runs, not uncertainty from independent samples. Pairwise story distances are dependent and are never bootstrap units.
- Judge-dependent scores inherit the judge model's biases.
- At least one model graded its own outputs (see Judge); interpret judge-dependent scores with caution.
