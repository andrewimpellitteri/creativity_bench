# Toward a credible creative-flexibility benchmark

## The claim

Measure **how many distinct, successful creative choices a model can sustain
under controlled constraints**. Publish a profile of capabilities and actual
writing examples. The current weighted composite is exploratory: putting
unrelated proxies in [0, 1] does not make their scales comparable or establish
that their average measures creativity.

This suite implements ideas from [Gwern's proposal](https://gwern.net/creative-benchmark).
That proposal motivates experiments; it does not validate these implementations.
Keep source-faithful raw measurements separate from additional validity checks.

## Audit motivating the redesign (before protocol 0.4)

| Task | Current limitation | Next experiment |
| --- | --- | --- |
| Free association | Forty turns may saturate; invalid responses can receive full non-repetition credit. Chao1 assumptions are questionable for history-dependent samples. | Report raw unique counts, first repetition and invalid-response rate; measure vocabulary growth at several budgets. Treat Chao1 as descriptive. |
| Telephone | Stochastic wording can prevent convergence indefinitely; eight rounds is a ceiling, not an observed collapse time. | Report censored survival curves across premises, and separate deterministic and stochastic generation conditions. Calibrate near-duplicate thresholds on labeled pairs. |
| Camel's back | Random edit bundles can contradict one another; initial story difficulty and judge leniency affect survival. | Use shared starting stories and precomputed, compatible edit schedules. Specify whether constraints persist or are replaced. Audit failed rounds. |
| Diversity | Different topics create embedding spread even when every story uses the same plot. Mean distance is not embedding volume. | Repeat each identical prompt, then vary one controlled input. Report within-prompt spread, plot repetition and effective embedding rank separately. |
| Style transfer | Fidelity is only reported; an unrelated story can win. Semantic embeddings also mix plot and style. | Evaluate plot invariants and target-genre success, then report divergence among successful transfers alongside the success rate. |
| Odd one out | The runner does not pass the optional membership judge; malformed judgments fail open. | Require a valid category verdict in a new scored protocol; unresolved judgments should invalidate the observation, not earn novelty credit. |
| Subversion | “Opposite” is ambiguous, and negative pairs may legitimately be opposites. | Specify the dimension to invert and facts to preserve; audit matched negatives and report sensitivity and specificity separately. |
| Shaggy dog | Disagreement about morals can reward incoherence; repeated calls to one judge are not independent judges. | Check basic comprehensibility, include deliberate anticlimax examples, and report agreement across several fixed judge models. |

## Highest-value new task: Same But Different

Give a model one premise and request another story after each accepted attempt,
showing short summaries of its previous stories. A successful attempt must:

1. Satisfy the premise and explicit constraints.
2. Be comprehensible, with a deliberately permissive quality floor.
3. Introduce a distinct causal plot rather than rename characters or scenery.

Use a fixed attempt budget; record rejections and never silently resample until
success. Plot cumulative distinct valid stories against attempts. Save each
story, summary, nearest prior story, judge evidence and verdict. Use a fixed
summarizer, and audit whether summaries hide meaningful differences.

This makes failure legible: readers can see a model repeatedly producing the
same reconciliation ending despite changes in names and setting. Start with
about 20 varied premises and 10 attempts each as a pilot, then choose the final
budget from observed saturation and uncertainty. These are proposed pilot sizes,
not a statistical power guarantee.

## Validation before a leaderboard

Build a small, hand-reviewed control set before adding more tasks:

- Exact copies and paraphrases: low diversity.
- Character/name substitutions: little plot diversity.
- Unrelated fluent stories: high raw distance, failed premise or fidelity.
- Random word salad: failed comprehensibility.
- Valid contrasting stories: high diversity with preserved constraints.
- Genre vocabulary swaps versus structural genre changes.
- Stories containing instructions to the evaluator: judge them as text.

Use blinded human annotations from multiple readers for validity, plot sameness
and style success. Preserve disagreement rather than forcing all aesthetic
judgments into one “quality” number. Tune thresholds on development examples;
freeze them before evaluating held-out examples. Compare automatic decisions
to held-out labels and inspect false positives and false negatives.

Then run a small panel of models on identical items and schedules. Pin judge,
embedder, prompts, sampling settings and protocol version. Record token limits,
truncations and actual costs. Repetition seeds control benchmark sampling;
they do not guarantee deterministic API output.

For uncertainty, resample independent premises with all their dependent stories
kept together. Do not treat every pairwise embedding distance or every turn as
an independent observation. Use paired comparisons on shared premises. Keep
smoke runs, task subsets, different budgets and protocol versions in separate
comparison groups. The report now separates these groups; charts require a single verified group.

## Implemented in protocol 0.4-validity

- Same But Different: a fixed attempt budget, full prior stories in judge context,
  accepted summaries in writer context, strict validity/distinctness verdicts,
  and saved rejection evidence and acceptance curves. The public pilot corpus has
  20 original premises spanning everyday, constrained and speculative situations;
  a full run samples six premises, a fast run two. This is development data.
- Free-association invalid responses now end the successful prefix and are saved.
  Category membership judgments are wired into Odd One Out. Style transfer requires
  plot, genre and comprehensibility checks before earning distance credit.
- Diversity repeats identical prompts, separating within-prompt from between-prompt
  distances. Effective rank is a geometric diagnostic, not embedding volume or a
  quality score. This task still lacks a semantic validity gate.
- Independent task RNG streams, source/data fingerprints, provider identities,
  request settings/finish reasons, and per-run usage snapshots support auditing.
  Iterative tasks retain their stories and mark unobserved collapse as censored.
- Reports separate incompatible cohorts, exclude explicitly incomplete evaluations,
  and show task profiles before the exploratory composite. Bootstrap intervals
  resample matched seeds, not dependent story pairs. Seeds are run units, not a
  claim that all underlying prompts are independent or newly sampled.
- `gallery` exports inspectable HTML; `validate-judge` exercises the production
  Same But Different evaluator against controls and preserves raw verdicts.

The new score is accepted stories / scheduled attempts. An unresolved judgment
gets no credit, but also marks the run incomplete so an evaluator outage does not
become a ranked claim about low creativity. Rejected stories consume attempts;
there is no generation retry-until-success in the task. API transport and empty
reasoning-response retries are separately logged. Novelty is judged against
accepted full stories; writer context uses summaries, so summarizer bias remains.

## Research rationale

[Gwern's proposal](https://gwern.net/creative-benchmark) motivates iterative tests
and the Same But Different variant. We preserve the underlying question while
making measurement choices explicit rather than presenting every implementation
decision as a requirement of the source.

[AidanBench](https://github.com/aidanmclaughlin/AidanBench) repeatedly asks for
novel answers and stops when coherence or novelty fails. Our fixed-budget variant
records recovery after failure and bounds cost; its acceptance count should not
be described as AidanBench's stopping-time score. We avoid importing its numeric
novelty threshold into a different story task without calibration.

[Nakajima et al., Beyond Divergent Creativity (2026)](https://aclanthology.org/2026.findings-eacl.138/)
show that noncreative baselines can outperform LLMs on DAT and introduce contextual
appropriateness into evaluation. This motivates our explicit off-premise and
nonsense controls; it does not validate our particular story judge.

[CreativityPrism](https://arxiv.org/abs/2510.20091) separates quality, novelty and
diversity and finds limited generalization across domains. Our design inference
is to report a capability profile, keep permissive validity separate from taste,
and avoid asserting a general creativity ranking from one normalized mean.

## Judge validation and external labels

`validate-judge` reports per-dimension confusion counts, resolution rate, and
accuracy among resolved judgments, separately by the supplied split. Undefined
novelty for invalid candidates is left unlabeled. Always inspect resolution rate
alongside accuracy: malformed responses are not agreement.

External controls are a nonempty JSON array with this shape:

```json
[{
  "id": "reviewed-example-001",
  "split": "held_out",
  "premise": "A required situation...",
  "candidate": "The actual story...",
  "accepted_stories": ["An earlier accepted story..."],
  "expected": {
    "premise_adherent": true,
    "comprehensible": true,
    "plot_distinct": false
  }
}]
```

Split labels do not enforce independence. Have people annotate fresh, blinded
examples; retain annotator identities and disagreement externally; freeze prompts
and criteria before using held-out labels. The bundled eight examples are proposed
development controls and cannot establish human agreement or general validity.
No live model evaluation or independent human study has been performed by this
implementation work.

## Remaining research work

1. Obtain independent judgments and a held-out control corpus, including style
   transfer and category membership. Check evaluator self-preference and sensitivity
   to judge choice, paraphrasing and adversarial instructions.
2. Run the matched pilot and inspect acceptance saturation, uncertainty and actual
   costs before increasing budgets or publishing comparisons.
3. Improve legacy Camel's Back schedules, calibrate Telephone convergence and
   deterministic/stochastic conditions, and redesign ambiguous Subversion negatives
   and Shaggy Dog's lexical moral-agreement score. These remain exploratory.
4. Validate transfer from these task profiles to blinded assessments of useful
   creative writing. No software test can establish that relationship alone.

Do not pool protocol 0.4 scores with 0.3-audit or legacy scores. Normalization and
validity gates changed, and the corpus, task budgets and sampling changed too.
