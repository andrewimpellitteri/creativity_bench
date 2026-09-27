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

All 46 suite runs re-judged (276 SBD attempts), transcripts unchanged.
Agreement per alternative judge, over comparable verdicts:

| dimension | deepseek-flash | glm-5.3-flash |
|---|---|---|
| premise_adherent | 98.9% (273/276), 3 flips | 97.8% (222/227), 5 flips |
| comprehensible | 99.6% (275/276), 1 flip | 99.6% (226/227), 1 flip |
| plot_distinct | 90.9% (251/276), 25 flips | 89.9% (204/227), 23 flips |

The `glm-5.3-flash` rescore landed 2026-09-18T05:19Z. Its parse-failure rate
improved sharply over the 12-model swap (49/276 unresolved, 18%, vs 33%
there) but remains the only source of lost comparability: `deepseek-flash`
resolved all 276. Fail-closed, those 49 attempts earn no acceptance credit.

Alternative leaderboard for all 23 models in `ALTERNATIVE_LEADERBOARD.md`,
now with the glm column: rank correlation 0.87 with the production judge
(flash: 0.92), and 0.83 between the two alternative judges. `deepseek-v4-pro`
holds 1.00 under the flash judge and slips to 0.92 under glm, while kimi-k2.5
(0.92→0.75) and qwen3.7-flash (0.75→0.42) drop under flash. `glm-5.3-flash`
scores itself 0.92→0.67, the largest self-judge penalty in the table.
