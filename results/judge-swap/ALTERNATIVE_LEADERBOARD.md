# Alternative Same But Different leaderboard (swapped judges)

Acceptance cascades replayed identically from each judge's verdicts over the
same saved stories; unresolved verdicts count as rejections (fail-closed).
Production score shown for reference. n=2 seeds per model; fast budget.

| model | production | replayed original | flash judge |
|---|---|---|---|
| deepseek-v4-pro | 1.00 | 1.00 | 1.00 |
| moonshotai/kimi-k3 | 1.00 | 1.00 | 1.00 |
| deepseek-flash | 0.92 | 0.92 | 0.83 |
| glm-4.5-air | 0.92 | 0.92 | 0.83 |
| glm-5.3-flash | 0.92 | 0.92 | 0.92 |
| moonshotai/kimi-k2.5 | 0.92 | 0.92 | 0.75 |
| glm-4.6 | 0.83 | 0.83 | 0.83 |
| openai/gpt-5-mini | 0.83 | 0.83 | 0.75 |
| deepseek/deepseek-v4-flash | 0.75 | 0.75 | 0.67 |
| glm-5-turbo | 0.75 | 0.75 | 0.83 |
| google/gemini-3-flash-preview | 0.75 | 0.75 | 0.83 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.75 | 0.75 | 0.67 |
| qwen/qwen3.7-flash | 0.75 | 0.75 | 0.42 |
| tencent/hy3 | 0.75 | 0.75 | 0.75 |
| anthropic/claude-haiku-4.5 | 0.67 | 0.67 | 0.67 |
| cohere/command-a | 0.50 | 0.50 | 0.42 |
| google/gemini-2.5-flash | 0.50 | 0.50 | 0.50 |
| mistralai/mistral-small-3.2-24b-instruct | 0.50 | 0.50 | 0.67 |
| google/gemini-2.5-flash-lite | 0.42 | 0.42 | 0.58 |
| meta-llama/llama-4-scout | 0.42 | 0.42 | 0.50 |
| microsoft/phi-4 | 0.33 | 0.33 | 0.33 |
| openai/gpt-4o-mini | 0.33 | 0.33 | 0.33 |
| x-ai/grok-4.20 | 0.33 | 0.33 | 0.33 |

Rank correlations across judges (Spearman):
- replayed original (deepseek-v4-pro) vs deepseek-flash as judge: 0.92

Self-judge check (deepseek-v4-pro): replayed original 1.00 vs under flash judge 1.00.
