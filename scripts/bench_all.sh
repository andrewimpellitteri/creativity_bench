#!/usr/bin/env bash
# Benchmark a lineup of models in parallel and plot the leaderboard.
# Requires OPENAI_API_KEY (generation + judge + embeddings) and ZAI_API_KEY (GLM models).
#
# Usage:
#   scripts/bench_all.sh            # full-size tasks, 2 repeats per model
#   FAST=1 scripts/bench_all.sh     # ~3x smaller tasks, 3 repeats, much faster
set -uo pipefail
cd "$(dirname "$0")/.."

OPENAI_MODELS=(gpt-5-mini gpt-5-nano gpt-4.1-mini gpt-4.1-nano gpt-4o-mini)
GLM_MODELS=(glm-4.6 glm-4.5-air)

# Pin the judge and embedder so every model is graded on the same scale.
COMMON=(--judge-model gpt-4.1-mini --judge-provider openai --seed 0)
if [[ "${FAST:-0}" == "1" ]]; then
  COMMON+=(--fast --n 3)
else
  COMMON+=(--n 2)
fi

mkdir -p logs
pids=()
names=()

launch() {
  local provider=$1 model=$2
  echo "starting $model ($provider)"
  uv run creativity-bench run --model "$model" --provider "$provider" "${COMMON[@]}" \
    >"logs/$model.log" 2>&1 &
  pids+=($!)
  names+=("$model")
}

for m in "${OPENAI_MODELS[@]}"; do launch openai "$m"; done
for m in "${GLM_MODELS[@]}"; do launch zai-coding "$m"; done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "done: ${names[$i]}"
  else
    echo "FAILED: ${names[$i]} (see logs/${names[$i]}.log)"
    fail=1
  fi
done

uv run creativity-bench viz
exit $fail
