# External correlation — this cohort vs EQ-Bench creative-writing boards

Rank correlations over manually verified name matches. Our side: fast-budget,
five-task, n=2 development scores with one pinned judge; their side: LLM-judged
boards with far more per-model effort. Approximate-variant matches (flagged *)
are included; n is small, so treat these as orientation, not validation.

## EQ-Bench Creative Writing v3 (Elo) (n=13)

n.b. * = approximate variant match

| our model | EQ-Bench name | our composite | their score |
|---|---|---|---|
| moonshotai/kimi-k3 | kimi-k3 | 0.88 | 2070.6 |
| glm-5.3-flash* | GLM-5.3 | 0.93 | 2064.1 |
| nvidia/nemotron-3-ultra-550b-a55b | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | 0.81 | 1689.3 |
| x-ai/grok-4.20* | grok-4.20-beta | 0.60 | 1570.7 |
| deepseek-flash | deepseek-ai/DeepSeek-V4-Flash | 0.91 | 1555.7 |
| deepseek-v4-pro | deepseek-ai/DeepSeek-V4-Pro | 0.93 | 1552.1 |
| glm-4.6 | zai-org/GLM-4.6 | 0.87 | 1408.6 |
| openai/gpt-5-mini* | gpt-5-mini-2025-08-07 | 0.74 | 1310.3 |
| mistralai/mistral-small-3.2-24b-instruct | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | 0.72 | 1252.9 |
| cohere/command-a | CohereForAI/c4ai-command-a-03-2025 | 0.74 | 1142.8 |
| google/gemini-2.5-flash* | gemini-2.5-flash-preview | 0.76 | 1134.6 |
| openai/gpt-4o-mini | gpt-4o-mini | 0.71 | 871.0 |
| meta-llama/llama-4-scout | meta-llama/Llama-4-Scout-17B-16E-Instruct | 0.68 | 781.2 |

**Spearman rho = 0.58, Kendall tau = 0.46**

Per-task Spearman vs their score:
- same_but_different: +0.64
- shaggy_dog: -0.04
- subversion: +0.43
- camels_back: +0.21

Chart: `CORRELATION_elo_score.png`

## EQ-Bench Longform Writing (0-100) (n=13)

n.b. * = approximate variant match

| our model | EQ-Bench name | our composite | their score |
|---|---|---|---|
| glm-5.3-flash* | GLM-5.3 | 0.93 | 81.8 |
| moonshotai/kimi-k3 | kimi-k3 | 0.88 | 79.6 |
| deepseek-v4-pro | deepseek-ai/DeepSeek-V4-Pro | 0.93 | 75.6 |
| x-ai/grok-4.20* | grok-4.20-beta | 0.60 | 68.5 |
| deepseek-flash | deepseek-ai/DeepSeek-V4-Flash | 0.91 | 66.0 |
| anthropic/claude-haiku-4.5 | claude-haiku-4.5 | 0.87 | 65.0 |
| glm-4.6 | zai-org/GLM-4.6 | 0.87 | 57.3 |
| openai/gpt-5-mini* | gpt-5-mini-2025-08-07 | 0.74 | 56.2 |
| google/gemini-3-flash-preview | gemini-3-flash-preview | 0.84 | 55.7 |
| google/gemini-2.5-flash* | google/gemini-2.5-flash-preview-05-20 | 0.76 | 48.6 |
| mistralai/mistral-small-3.2-24b-instruct | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | 0.72 | 41.6 |
| meta-llama/llama-4-scout | meta-llama/Llama-4-Scout-17B-16E-Instruct | 0.68 | 31.1 |
| microsoft/phi-4* | microsoft/phi-4-multimodal-instruct | 0.43 | 22.8 |

**Spearman rho = 0.75, Kendall tau = 0.64**

Per-task Spearman vs their score:
- same_but_different: +0.72
- shaggy_dog: -0.34
- subversion: +0.65
- camels_back: +0.49

Chart: `CORRELATION_overall_score_100.png`
