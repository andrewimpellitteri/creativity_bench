#!/bin/zsh
# Extended cohort live runs; sequential, stop on first failure.
set -a
source .env
set +a
set -e
echo "=== validate ==="
uv run python scripts/live_pilot.py validate --judge deepseek-v4-pro --out results/extended-20260915
for m in deepseek-flash deepseek-v4-pro; do
  echo "=== suite $m seeds 0-1 ==="
  uv run creativity-bench run --model "$m" --provider deepseek \
    --judge-model deepseek-v4-pro --judge-provider deepseek \
    --tasks free_association,camels_back,shaggy_dog,subversion,same_but_different \
    --seed 0 --n 2 --fast --timeout 240 \
    --runs-dir results/extended-20260915/suite_runs
done
echo "=== sbd extension seeds 2-3 ==="
uv run python scripts/live_pilot.py run --judge deepseek-v4-pro \
  --models deepseek-flash deepseek-v4-pro --seeds 2 3 --fast \
  --out results/extended-20260915
echo "EXTENDED-COHORT-DONE"
