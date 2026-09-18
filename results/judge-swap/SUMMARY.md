# Judge-swap sensitivity — 2026-09-16

Alternative judges re-scored every judged Same But Different attempt in the
12-model fast-suite cohort (120 attempts, transcripts unchanged, production
judge prompt and parsing). Original judge: `deepseek-v4-pro`.

| dimension | deepseek-flash (n comparable) | glm-5.3-flash (n comparable) |
|---|---|---|
| premise_adherent | 99.2% (119/120), 1 flip | 100% (80/80), 0 flips |
| comprehensible | 100% (120/120), 0 flips | 100% (80/80), 0 flips |
| plot_distinct | 90.8% (109/120), 11 flips | 86.3% (69/80), 11 flips |

Key observations:

- Validity dimensions are essentially judge-independent; the contested
  dimension is plot distinctness, as designed (~9-14% flip rate).
- `glm-5.3-flash` failed to return a parseable verdict on 40/120 attempts
  (33%). Fail-closed handling gave those attempts no credit and no agreement
  credit. Judge reliability, not just judge accuracy, differs by family.
- Both swaps used the same saved transcripts; no stories were regenerated.
  Raw records with evidence for every flip are in the two JSON reports.

Judge-swap measures judge sensitivity, not story quality or validity; it is
development evidence toward (not a substitute for) human agreement.

## Full-cohort rescore (2026-09-17)

All 46 suite runs re-judged (276 SBD attempts). `deepseek-flash` as judge:
98.9% / 99.6% / 90.9% agreement (premise / comprehensible / plot-distinct),
25 plot-distinct flips. Alternative leaderboard for all 23 models in
`ALTERNATIVE_LEADERBOARD.md`: rank correlation 0.92 with the production judge;
`deepseek-v4-pro` holds 1.00 even under the alternative judge, while
kimi-k2.5 (0.92→0.75) and qwen3.7-flash (0.75→0.42) drop. The `glm-5.3-flash`
rescore of the full cohort is still running (slow endpoint, high retry rate);
this file and the leaderboard gain its column when it lands.
