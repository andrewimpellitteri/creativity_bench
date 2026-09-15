#!/bin/zsh
# Multi-model cohort on the extended-20260915 fingerprint.
# Five non-embedding tasks, fast sizes, paired seeds 0 and 1.
# Judge fixed to the validated deepseek-v4-pro for every writer.
# Idempotent: skips any (model, seed) already saved in the runs directory.
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
  if grep -qs "\"seed\": $seed" $OUT/${model}_*.json 2>/dev/null; then
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
  run_one deepseek-flash deepseek $seed
  run_one deepseek-v4-pro deepseek $seed
  run_one glm-4.6 zai-coding $seed
  run_one glm-5 zai-coding $seed
  run_one glm-5-turbo zai-coding $seed
  run_one glm-4.5-air zai-coding $seed
done

if [[ -n "$OPENROUTER_API_KEY" ]]; then
  for m in google/gemini-2.5-flash google/gemini-2.0-flash-001 \
           mistralai/mistral-small-3.2 meta-llama/llama-4-scout \
           anthropic/claude-3.5-haiku openai/gpt-4o-mini; do
    for seed in 0 1; do
      run_one "$m" openrouter $seed
    done
  done
else
  echo "OPENROUTER_API_KEY not set; openrouter roster skipped"
fi
echo "MULTIMODEL-DONE"
