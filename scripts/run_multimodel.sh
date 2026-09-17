#!/bin/zsh
# Multi-model cohort on the extended-20260915 fingerprint.
# Five non-embedding tasks, fast sizes, paired seeds 0 and 1.
# Judge fixed to the validated deepseek-v4-pro for every writer.
# Idempotent: skips any (model, seed) already saved in the runs directory.
# The roster lists every model ever run in this cohort; reruns only fill gaps.
set -a
source .env
set +a
set -e
OUT=results/extended-20260915/suite_runs
TASKS=free_association,camels_back,shaggy_dog,subversion,same_but_different

run_one () {
  model=$1
  provider=$2
  seed=$3
  safe=$(printf '%s' "$model" | tr "/" "_")
  if grep -qs "\"seed\": $seed," $OUT/${safe}_*.json 2>/dev/null; then
    echo "skip $model seed $seed (already saved)"
    return
  fi
  echo "=== $model ($provider) seed $seed ==="
  uv run creativity-bench run --model "$model" --provider "$provider" \
    --judge-model deepseek-v4-pro --judge-provider deepseek \
    --tasks $TASKS --seed "$seed" --n 1 --fast --timeout 240 \
    --runs-dir "$OUT"
}

for seed in 0 1; do
  # DeepSeek (direct) and the GLM coding endpoint.
  run_one deepseek-flash deepseek $seed
  run_one deepseek-v4-pro deepseek $seed
  run_one glm-4.6 zai-coding $seed
  run_one glm-5.3-flash zai-coding $seed
  run_one glm-5-turbo zai-coding $seed
  run_one glm-4.5-air zai-coding $seed
done

if [[ -n "$OPENROUTER_API_KEY" ]]; then
  for m in anthropic/claude-haiku-4.5 google/gemini-2.5-flash \
           meta-llama/llama-4-scout mistralai/mistral-small-3.2-24b-instruct \
           moonshotai/kimi-k2.5 openai/gpt-4o-mini \
           deepseek/deepseek-v4-flash google/gemini-2.5-flash-lite \
           microsoft/phi-4 moonshotai/kimi-k3 nvidia/nemotron-3-ultra-550b-a55b \
           qwen/qwen3.7-flash tencent/hy3 \
           cohere/command-a google/gemini-3-flash-preview \
           x-ai/grok-4.20; do
    for seed in 0 1; do
      run_one "$m" openrouter $seed
    done
  done
else
  echo "OPENROUTER_API_KEY not set; openrouter roster skipped"
fi
echo "MULTIMODEL-DONE"
