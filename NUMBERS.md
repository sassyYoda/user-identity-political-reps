# NUMBERS.md

Source of truth for measured results. Any number that appears in a figure, note,
or draft must have an entry here first; a number without an entry does not exist.

Each entry records:

- **date** — when the run finished
- **stage** — which pipeline stage produced it (reconstruct / cache / sweep / ...)
- **metric and value** — what was measured
- **seed** — the run seed
- **provenance** — the artifact path; its `.meta.json` sidecar carries versions,
  model revision, and git state
- **registered** — `pre` if the analysis was decided before looking at the data,
  `post` if it is post-hoc (post-hoc numbers need confirmation before being cited)

## Environment

| date | stage | metric | value | seed | provenance | registered |
|---|---|---|---|---|---|---|
| 2026-08-26 | setup | pinned model revision (google/gemma-2-9b-it) | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | — | `artifacts/gemma-2-9b-it/model_access_check.json` | pre |
| 2026-08-26 | reconstruct | pinned dataset revision (sarahcen/llm-election-data-2024) | `7bb3c18c2eadfc3f96db0dd394768496f7107a79` | — | `artifacts/prompt_table.csv` | pre |
| 2026-08-26 | reconstruct | matched sets (each: "none" + 21 variations), none excluded for missing baseline | 573 | 0 | `artifacts/prompt_table.csv` | pre |
| 2026-08-26 | reconstruct | prompts in table (573 sets x 22 conditions; under the 1,000-set cap, so no subsampling) | 12,606 | 0 | `artifacts/prompt_table.csv` | pre |

## Probe curve (milestone 1, Gemma-2-9B-IT)

Full run over the real prompt table. Probes: logistic, GroupKFold by base-question
template, 5 splits, fold-local standardization. 124 of the 840 LBFGS fits hit
max_iter=1000 (ConvergenceWarning); with every curve at or near ceiling this can
only understate accuracy, so no number below changes. ADR-0004 (dated after this
run) moves headline numbers to Gemma-3-12B-IT; this table is the 9B replication
anchor. Provenance paths updated 2026-08-27 when the port made cache and
artifact paths model-scoped (`activations/main` -> `activations/gemma-2-9b-it/main`,
`artifacts/<file>` -> `artifacts/gemma-2-9b-it/<file>`); the artifacts themselves
are byte-identical to the original run's.

| date | stage | metric | value | seed | provenance | registered |
|---|---|---|---|---|---|---|
| 2026-08-27 | cache | prompts cached, one forward pass each (42 layers x d_model 3584, last-token resid_post, fp32) | 12,606 | — | `activations/gemma-2-9b-it/main` | pre |
| 2026-08-27 | sweep | multinomial 22-way peak held-out accuracy (chance 0.0455) | 0.9998 at layer 8 | 0 | `artifacts/gemma-2-9b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | multinomial layer-0 accuracy (leakage diagnostic; pre-registered expectation was near-chance) | 0.8539 | 0 | `artifacts/gemma-2-9b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | multinomial minimum accuracy over layers 1-41 (the floor of the post-layer-0 saturation; the all-layer minimum is the layer-0 row above) | 0.9370 (layer 32) | 0 | `artifacts/gemma-2-9b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | multinomial shuffled-label reference, max over layers | 0.0501 | 0 | `artifacts/gemma-2-9b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | binary "I am a Democrat. {}" vs "I am a Republican. {}" peak (chance 0.5) | 1.0000 at layers 5-8 | 0 | `artifacts/gemma-2-9b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | binary layer-0 accuracy | 0.9493 | 0 | `artifacts/gemma-2-9b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | binary shuffled-label reference, max over layers | 0.5183 | 0 | `artifacts/gemma-2-9b-it/probe_curve.json` | pre |
| 2026-08-27 | displacement | difference-in-means vectors saved, raw + unit-norm (conditions x layers x d_model; 573 matched sets per condition) | 21 x 42 x 3584 | — | `artifacts/gemma-2-9b-it/displacements.npz` | pre |
| 2026-08-27 | leakage | prefix-form conditions only, 13-way at layer 0 (chance 0.0769; distinctive tokens far from measured position) | 0.8046 | 0 | `artifacts/gemma-2-9b-it/leakage_check.json` | post |
| 2026-08-27 | leakage | suffix-form conditions only, 8-way at layer 0 (chance 0.1250; distinctive tokens adjacent to measured position) | 0.9954 | 0 | `artifacts/gemma-2-9b-it/leakage_check.json` | post |
| 2026-08-27 | leakage | prefix-form 13-way at layers 1 and 8 | 0.9782, 1.0000 | 0 | `artifacts/gemma-2-9b-it/leakage_check.json` | post |

## Probe curve (Gemma-3-12B-IT port, MATS arc ticket 01)

The milestone-1 pipeline re-run on the subject model of ADR-0004, same prompt
table, same probe settings (logistic, GroupKFold by base-question template,
5 splits, fold-local standardization). Per the ticket-05 findings this curve is
a sanity panel, not evidence: it is saturated from layer 1, so its peaks must
not be used to pick layers or to claim an engaged representation. LBFGS
ConvergenceWarnings appeared as on the 9B run; with the curves at or near
ceiling, non-convergence can only understate accuracy.

| date | stage | metric | value | seed | provenance | registered |
|---|---|---|---|---|---|---|
| 2026-08-27 | setup | pinned model revision (google/gemma-3-12b-it) | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | — | `artifacts/gemma-3-12b-it/model_access_check.json` | pre |
| 2026-08-27 | cache | prompts cached, one forward pass each (48 layers x d_model 3840, last-token resid_post, fp32) | 12,606 | — | `activations/gemma-3-12b-it/main` | pre |
| 2026-08-27 | sweep | multinomial 22-way peak held-out accuracy (chance 0.0455) | 0.9965 at layer 10 | 0 | `artifacts/gemma-3-12b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | multinomial layer-0 accuracy (leakage diagnostic; 9B was 0.8539) | 0.7029 | 0 | `artifacts/gemma-3-12b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | multinomial minimum accuracy over layers 1-47 (the floor of the post-layer-0 saturation) | 0.9412 (layer 1) | 0 | `artifacts/gemma-3-12b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | multinomial shuffled-label reference, max over layers | 0.0490 | 0 | `artifacts/gemma-3-12b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | binary "I am a Democrat. {}" vs "I am a Republican. {}" peak (chance 0.5) | 1.0000 at layers 10, 13, 15, 16 | 0 | `artifacts/gemma-3-12b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | binary layer-0 accuracy | 0.7583 | 0 | `artifacts/gemma-3-12b-it/probe_curve.json` | pre |
| 2026-08-27 | sweep | binary shuffled-label reference, max over layers | 0.5210 | 0 | `artifacts/gemma-3-12b-it/probe_curve.json` | pre |
| 2026-08-27 | displacement | difference-in-means vectors saved, raw + unit-norm (conditions x layers x d_model; 573 matched sets per condition) | 21 x 48 x 3840 | — | `artifacts/gemma-3-12b-it/displacements.npz` | pre |
| 2026-08-27 | replication | two-model overlay on normalized depth: multinomial peaks (Gemma-2 vs Gemma-3) | 0.9998 at layer 8/41 vs 0.9965 at layer 10/47 | — | `artifacts/replication_probe_curve.json` | pre |
| 2026-08-27 | leakage | prefix-form conditions only, 13-way at layer 0 (chance 0.0769; distinctive tokens far from measured position) | 0.5263 | 0 | `artifacts/gemma-3-12b-it/leakage_check.json` | post |
| 2026-08-27 | leakage | suffix-form conditions only, 8-way at layer 0 (chance 0.1250; distinctive tokens adjacent to measured position) | 0.9799 | 0 | `artifacts/gemma-3-12b-it/leakage_check.json` | post |
| 2026-08-27 | leakage | prefix-form 13-way at layers 1 and 10 | 0.9116, 0.9976 | 0 | `artifacts/gemma-3-12b-it/leakage_check.json` | post |

## Ideology direction and transfer test (Gemma-3-12B-IT, MATS arc ticket 02)

Content corpus: 114th-Congress floor speeches (Gentzkow/Shapiro/Taddy
hein-daily, ODC-BY 1.0) labeled with the speaker's DW-NOMINATE dim1 via a
seat+lastname join to Voteview, filtered to 100-400 words and verified free of
every scaffold phrase; provenance and sha256s in `data/raw/*/provenance.json`.
Cached at the same seam as the scaffold prompts (one user turn, last
pre-generation token). Transfer scorers fit almost nothing by construction:
content statements are scored by a per-fold threshold on the projection
(GroupKFold by speaker), scaffold rows by a zero-parameter paired comparison
within matched sets under the fixed conservative-positive sign convention.
Headline: transfer is asymmetric. The content-derived ideology direction reads
the Democrat/Republican scaffold pairs near-perfectly in the late stack — and
still does when re-derived from statements containing no party-family token —
while the scaffold-derived axis does not separate content statements above
its shuffled reference at any layer.

| date | stage | metric | value | seed | provenance | registered |
|---|---|---|---|---|---|---|
| 2026-08-27 | corpus | statements sampled (1,000 per party, ≤10 per speaker, 470 speakers, congress 114) | 2,000 | 0 | `artifacts/content_corpus.csv` | pre |
| 2026-08-27 | corpus | speech->speaker join: labeled speeches; hein-vs-Voteview party disagreements among them | 67,257; 0 | — | `artifacts/content_corpus.csv` | pre |
| 2026-08-27 | corpus | eligible statements containing a scaffold phrase, removed before sampling | 5 of 18,746 | — | `artifacts/content_corpus.csv` | pre |
| 2026-08-27 | cache | content statements cached, one forward pass each (48 layers x d_model 3840, last-token resid_post, fp32) | 2,000 | — | `activations/gemma-3-12b-it/content` | pre |
| 2026-08-27 | direction | diff-in-means vs ridge-weights cosine, range over layers (0.271 at the working layer; both estimators share labels, so this is a consistency check, not independent confirmation) | +0.078..+0.420 | — | `artifacts/gemma-3-12b-it/ideology_direction.json` | pre |
| 2026-08-27 | transfer | content->scaffold, paired accuracy over 573 matched sets (chance 0.5; swapped-pair reference max 0.555): peak / at layer 46 | 0.9983 (layer 46) / 0.9983 | 0 | `artifacts/gemma-3-12b-it/transfer_test.json` | pre |
| 2026-08-27 | transfer | content->scaffold with the ridge direction (chance 0.5): peak / at layer 46 | 1.0000 (layer 11) / 0.9878 | 0 | `artifacts/gemma-3-12b-it/transfer_test.json` | pre |
| 2026-08-27 | transfer | content->scaffold with the direction re-derived from the 1,688 statements with no party-family token (token-detector kill check) | 1.0000 (layer 31) / 0.9983 at layer 46 | 0 | `artifacts/gemma-3-12b-it/transfer_test.json` | post |
| 2026-08-27 | transfer | scaffold->content, threshold scorer, speaker-grouped CV (chance 0.5; shuffled reference max 0.5325): best layer | 0.5500 (layer 47); at chance | 0 | `artifacts/gemma-3-12b-it/transfer_test.json` | pre |
| 2026-08-27 | transfer | scaffold->content on no-party-token statements (chance 0.5450; shuffled max 0.5279): best layer | 0.5616 (layer 47); at chance | 0 | `artifacts/gemma-3-12b-it/transfer_test.json` | pre |
| 2026-08-27 | alignment | H2a cosine, (Republican minus Democrat) scaffold displacement vs ideology direction, layer 46; random-direction null sd; empirical p (100,000 draws) | +0.1349; 0.0162; <= 1e-5 | 0 | `artifacts/gemma-3-12b-it/transfer_test.json` | pre |
| 2026-08-27 | alignment | per-layer cosine extremes: the sign is unstable mid-stack, stable positive from layer 38 on (+0.13..+0.30) | +0.844 (layer 23), -0.827 (layer 11) | 0 | `artifacts/gemma-3-12b-it/transfer_test.json` | pre |
| 2026-08-27 | working layer | chosen for downstream analyses: argmax of min accuracy over the informative transfer variants (those clearing shuffled max + 0.05; the three content->scaffold curves) among layers with positive diff and ridge alignment | 46 | 0 | `artifacts/gemma-3-12b-it/transfer_test.json` | post |

## Projection gradient (Gemma-3-12B-IT, MATS arc ticket 03)

Every condition's displacement, meaning the 21 Cen variations plus five
generated controls (two inert, one syntactic, two partisan paraphrases;
`artifacts/control_table.csv`, cached at the same seam over the same 573 base
questions), projected per matched set onto the unit ideology direction at
working layer 46 (the post-hoc-chosen layer of ticket 02), then averaged over
sets. Nulls per ADR-0003: within-set label permutations (10,000 draws, pooled
across the 26 conditions) and matched-norm random directions (100,000 draws).
The anchor leans were pinned before any projection was computed, from Pew's
validated-voter study of the 2024 election, published June 2025
(`scripts/fetch_partisan_lean.py`; PDF sha256-pinned; margins
hand-transcribed, owner re-check is a ticket-07 task). Per ADR-0003 the
anchor correlation is a consistency check, never a headline statistic: only
six scaffolds have anchors, so its rows carry the n. One correction between
the first and final run: the first look showed the permutation null is not
centered at zero (every condition shares a large positive projection), so the
per-condition permutation p was changed from a zero-centered magnitude test
to a two-sided test against the null distribution itself; the
planted-gradient unit test now plants a common offset to cover this. No other
difference between the runs, and the ranking and both null distributions were
unaffected.

| date | stage | metric | value | seed | provenance | registered |
|---|---|---|---|---|---|---|
| 2026-08-27 | controls | control prompts cached, one forward pass each (573 sets x 5 conditions, 48 layers x d_model 3840, last-token resid_post, fp32) | 2,865 | — | `activations/gemma-3-12b-it/controls` | pre |
| 2026-08-27 | gradient | common projection component shared by all 26 conditions at layer 46: permutation-null mean; 99% band | +331.6; [+246.8, +416.3] | 0 | `artifacts/gemma-3-12b-it/gradient.json` | post |
| 2026-08-27 | gradient | spectrum top at layer 46 (perm-p and rand-p all at their 7.7e-6 / 1e-5 floors): Republican scaffold / its paraphrase / man scaffold | +1763.3 / +1604.9 / +1211.8 | 0 | `artifacts/gemma-3-12b-it/gradient.json` | pre |
| 2026-08-27 | gradient | spectrum bottom at layer 46: "diversity of opinions" suffix / Black scaffold | -918.7 / -461.6 | 0 | `artifacts/gemma-3-12b-it/gradient.json` | pre |
| 2026-08-27 | gradient | Democrat scaffold at layer 46: statistically indistinguishable from the common component (perm-p 0.82, rand-p 0.23); the partisan separation is carried by the Republican side | +339.2 | 0 | `artifacts/gemma-3-12b-it/gradient.json` | pre |
| 2026-08-27 | gradient | controls at layer 46: syntactic "I am a person" / inert left-handed / inert born-in-June. All three deviate from the common component (perm-p at the 7.7e-6 floor; person and its near-zero absolute projection sit below the band). On the random-direction null person is clean (0.78), left-handed borderline (0.050), June clearly loaded (4.2e-4) | +53.8 / +534.1 / +870.4 | 0 | `artifacts/gemma-3-12b-it/gradient.json` | pre |
| 2026-08-27 | gradient | non-identity variations vs the random-direction null: indistinguishable from a random axis (rand-p > 0.05) for 9 of 10; the exception is the "diversity of opinions" suffix (rand-p 2.0e-5) | 9/10 | 0 | `artifacts/gemma-3-12b-it/gradient.json` | pre |
| 2026-08-27 | gradient | partisan-lean anchor (consistency check, small n): Spearman between demographic projections at layer 46 and the net Republican margin in the 2024 vote from Pew's validated-voter study (6 anchored scaffolds; American Indian, Pacific Islander, non-binary unreported by Pew) | +0.928 (exact p 0.0222, n=6) | 0 | `artifacts/gemma-3-12b-it/gradient.json` | pre |
| 2026-08-27 | gradient | paraphrase check, layer 46: displacement cosine vs the original scaffold ("I'm a member of the Democratic/Republican party" vs "I am a Democrat/Republican"); spectrum ranks adjacent (13 vs 14, 2 vs 1) | +0.979 / +0.991 | 0 | `artifacts/gemma-3-12b-it/gradient.json` | pre |
| 2026-08-27 | gradient | rank stability of the spectrum, layer 46 vs 39: all 26 conditions (layer 39 is ticket 02's recorded alternative; no written plan pre-registered this comparison) | +0.605 (p 0.0011) | 0 | `artifacts/gemma-3-12b-it/gradient.json` | post |
| 2026-08-27 | gradient | rank stability restricted to identity scaffolds and controls (n=16; the suffix variations swing hard at 39, up to +5664) | +0.721 (p 0.0023) | 0 | `artifacts/gemma-3-12b-it/gradient.json` | post |
| 2026-08-27 | gradient | partisan-lean anchor at layer 39 | +0.812 (exact p 0.072, n=6) | 0 | `artifacts/gemma-3-12b-it/gradient.json` | post |

## Black-box baseline (Gemma-3-12B-IT, MATS arc ticket 04)

Under all 27 conditions (21 Cen variations, 5 generated controls, and the
"none" baseline), over a seeded subsample of 60 of the 573 matched sets, the
model was asked directly for its best guess of the user's political leaning:
the scaffolded question with a fixed probe appended, so the context matches
the cached activation measurement up to one constant suffix. Decoding greedy,
40 new tokens max, settings and probe text in the generations sidecar.
Answers are scored by a deliberately dumb rule (`polreps/blackbox.py`): the
first paragraph naming exactly one scale option (whole words only) scores it,
"unknown" or a can't-tell phrase abstains, anything else is unscorable. Two
design caveats. The probe offers "unknown" as an explicit option, so the
abstention rates measure uptake of an offered out, not spontaneous refusal.
And the internal side of every comparison is the 573-set gradient artifact
while the verbal side covers the 60 sampled sets; recomputing the internal
means over just those 60 sets moves no condition materially (unregistered
sanity check, recorded in the ticket-04 comments). Internal deviations
are read against ticket 03's common offset as planned; one change after the
first look, labeled post below: the "none" baseline itself reports
"conservative" from question content alone, so verbal leaning is judged by
the delta from that baseline (mirroring the internal offset), not by distance
from zero. Ranks are unchanged by the shift; only the lean/no-lean readings
depend on it. A seeded random draw of raw answers (2 per condition, never
filtered on content) is in `artifacts/gemma-3-12b-it/blackbox_examples.md`.

| date | stage | metric | value | seed | provenance | registered |
|---|---|---|---|---|---|---|
| 2026-08-27 | ask | direct-question generations (27 conditions x 60 sets, greedy, one forward context per prompt) | 1,620 | 0 | `artifacts/gemma-3-12b-it/blackbox_generations.jsonl` | pre |
| 2026-08-27 | score | answers scored on the 5-point scale / abstained / unscorable (the 11 unscorables are all 40-token truncations in explain-style suffix conditions) | 766 / 843 / 11 | 0 | `artifacts/gemma-3-12b-it/blackbox.json` | pre |
| 2026-08-27 | baseline | verbal report under "none": mean of scored answers (scale -2 very liberal .. +2 very conservative); share scored; abstain rate | +0.72; 32/60; 0.47 | 0 | `artifacts/gemma-3-12b-it/blackbox.json` | pre |
| 2026-08-27 | compare | Republican scaffold / its paraphrase: verbal mean (delta vs baseline), all 60 scored, beside internal deviation from the common offset | +1.02 (+0.30) / +1.00 (+0.28); internal +1431.7 / +1273.4 | 0 | `artifacts/gemma-3-12b-it/blackbox.json` | pre |
| 2026-08-27 | compare | Democrat scaffold / its paraphrase: verbal mean (delta), 59-60 of 60 scored, beside internal deviation (perm-p). The model verbalizes a liberal user; the layer-46 projection sits at the offset | -1.00 (-1.72) both; internal +7.7 (p 0.82) / -8.4 (p 0.80) | 0 | `artifacts/gemma-3-12b-it/blackbox.json` | pre |
| 2026-08-27 | compare | abstention asymmetry across demographics: abstain rate for non-binary / Pacific Islander / Black / Asian / Hispanic / American Indian, then White / woman / man | 0.98 / 0.97 / 0.95 / 0.95 / 0.93 / 0.90, then 0.57 / 0.73 / 0.47 | 0 | `artifacts/gemma-3-12b-it/blackbox.json` | pre |
| 2026-08-27 | compare | "knows more than it says" cells (internal deviation at the 7.7e-6 perm-p floor, no verbal lean vs baseline): Black / non-binary / Pacific Islander / American Indian / left-handed / man | -793.2 / -588.0 / -266.7 / +181.9 / +202.6 / +880.3 | 0 | `artifacts/gemma-3-12b-it/blackbox.json` | post |
| 2026-08-27 | compare | born-in-June control: verbal delta (23/60 scored) against internal deviation; ticket 03's June-conservative loading is internal only, and the verbal side leans the other way | -0.46 vs +538.8 | 0 | `artifacts/gemma-3-12b-it/blackbox.json` | post |
| 2026-08-27 | correlate | verbal mean vs internal projection, Spearman over conditions with >= 10 scored answers, layer 46; layer 39 | +0.438 (p 0.063, n=19); +0.483 (p 0.040) | 0 | `artifacts/gemma-3-12b-it/blackbox.json` | pre |
| 2026-08-27 | correlate | the same restricted to identity scaffolds and controls (the suffix variations verbalize the question content's lean, near the baseline), layer 46; layer 39 | +0.828 (p 0.0086, n=9); +0.720 (p 0.035) | 0 | `artifacts/gemma-3-12b-it/blackbox.json` | post |

## Minimal causal steering (Gemma-3-12B-IT, MATS arc ticket 05)

Steering in the CONTEXT.md sense: during generation, alpha times the
unit-norm (Republican minus Democrat) displacement from
`displacements.npz` is added into the layer-46 residual stream (the
working layer) at every position, through an HF forward hook verified
numerically against the cached `resid_post` measurement point (max abs
diff 0). The subject answers each base question with no scaffold; the
matched-norm control is a seeded random unit direction on the same grid.
The judge is the subject model itself, unsteered and blind to the
steering condition (no second model or API judge is configured; the LLM
slant-judge precedent is Kim et al. 2025 — GPT-4o judge, ICC 0.91
against human raters — and self-judging is the documented residual
caveat: that validation covers a different judge, so this instrument is
unvalidated until the owner reads the 36-example dump against its
ratings, the first ticket-07 task). Scoring reuses ticket 04's dumb rule with an explicit
"no discernible stance" option and a separate coherence probe; every one
of the 1,170 judge answers was scorable. The alpha grid was calibrated
before the dose-response run under a rule fixed in code (both signs
coherent at >= 0.8 with zero gibberish): Kim et al.'s Llama-calibrated
|30| window is off by three orders of magnitude here — layer-46
residual norms are ~1.3e5 and the R-D displacement norm is 10,554.9, and
past the cliff the text degenerates *on axis* (at +80k it repeats
"conservative", at +160k it repeats the name of an Indian right-wing
party). The pre-registered sign prediction (module docstring, committed
before any generation): slant increases with alpha, with the ticket-03
Republican-carried asymmetry making the liberal half the weaker half.
The headline is honest and two-sided: the pooled dose-response is
positive and significant but *small* (rho +0.082, about a 0.16-point
swing on a 4-point scale across an 8x-the-natural-displacement steer),
and the asymmetry caveat was wrong — per-alpha, only the liberal end
individually clears its CI. The random control is flat where it is
readable, and unreadable exactly where it would have to be large: it
breaks coherence at |40k| while the displacement direction does not —
the model tolerates ~4x-displacement-norm perturbations along the
partisan axis but not along a random one, which is itself evidence the
axis is meaningful, and it makes the random extreme cells
survivor-biased (reported with their thinned n).

| date | stage | metric | value | seed | provenance | registered |
|---|---|---|---|---|---|---|
| 2026-08-27 | calibrate | coherence cliff on the displacement direction (ladder 2.5k-160k, both signs, 6 questions, 120 new tokens): first magnitude failing the fixed rule; resulting grid | \|80,000\|; ±40k, ±20k, 0 | 1 | `artifacts/gemma-3-12b-it/steering_calibration.json` | pre |
| 2026-08-27 | generate | steered generations (60 political questions, ticket 04's seeded subsample, x 9 direction-alpha conditions + 5 off-target prompts x 9; greedy, 200 new tokens) | 585 | 0 | `artifacts/gemma-3-12b-it/steering_generations.jsonl` | pre |
| 2026-08-27 | judge | slant + coherence probes answered by the unsteered subject model (greedy, so no seed applies); unscorable answers | 1,170; 0 | — | `artifacts/gemma-3-12b-it/steering_judgments.jsonl` | pre |
| 2026-08-27 | dose-response | primary statistic, displacement direction: pooled Spearman alpha-vs-slant over scored political generations, within-question permutation p (10,000 draws) | rho +0.082 (p 0.0031, n=295, 60 questions) | 0 | `artifacts/gemma-3-12b-it/steering.json` | pre |
| 2026-08-27 | dose-response | matched-norm random control, same statistic (its \|40k\| cells thinned to 12 and 37 scored by the coherence collapse below) | rho +0.043 (p 0.20, n=229) | 0 | `artifacts/gemma-3-12b-it/steering.json` | pre |
| 2026-08-27 | dose-response | extremes contrast, paired within question: slant(+40k) − slant(−40k), displacement; random (survivor-biased n) | +0.16 ± 0.10 (57 pairs); +0.25 ± 0.32 (8 pairs) | 0 | `artifacts/gemma-3-12b-it/steering.json` | pre |
| 2026-08-27 | dose-response | cell means, displacement −40k / −20k / 0 / +20k / +40k (58-60 of 60 scored per cell; no-stance rate ≤ 0.03 throughout) | −0.05 / +0.03 / +0.05 / +0.00 / +0.10 | 0 | `artifacts/gemma-3-12b-it/steering.json` | pre |
| 2026-08-27 | dose-response | per-alpha paired deltas vs alpha=0, displacement: −40k / −20k / +20k / +40k. Only −40k individually excludes 0; the pre-stated "liberal half weaker" caveat did not materialize | −0.103 ± 0.093 / −0.017 / −0.052 / +0.051 ± 0.088 | 0 | `artifacts/gemma-3-12b-it/steering.json` | post |
| 2026-08-27 | coherence | judged coherence (0-2) across the grid: displacement direction; random direction at ±40k (with judge no-stance rates 0.80 / 0.38) | 1.98-2.00 everywhere; 0.78 / 1.37 | 0 | `artifacts/gemma-3-12b-it/steering.json` | pre |
| 2026-08-27 | spot-check | off-target prompts (5 x 9 conditions): answers judged partisan (nonzero slant) at any alpha | 0 of 45 (13 moderate, 32 no-stance) | 0 | `artifacts/gemma-3-12b-it/steering.json` | pre |
| 2026-08-27 | sign | pre-registered sign prediction evaluated: pooled trend positive as predicted (rho > 0, extremes contrast > 0); the pre-stated asymmetry caveat (liberal side weaker) contradicted by the per-alpha deltas | direction confirmed, caveat not | 0 | `artifacts/gemma-3-12b-it/steering.json` | pre |

## Internal-to-behavioral link (Gemma-3-12B-IT, MATS arc ticket 06)

The stretch question: does the size of a scaffold's activation displacement
predict how much the scaffold moves the model's actual answers? Under the 16
identity scaffolds, partisan paraphrases, and controls plus "none" (the ten
suffix variations were excluded for the generation budget; the exclusion is a
stated limitation, not a robustness choice), the model answered the base
questions bare: greedy, 100 new tokens, ticket 04's seeded 60-set subsample,
through the ticket-04 harness. Ticket 04's own generations answer the appended
leaning probe, not the question, so this pass was new. Behavioral displacement
per condition is Cen et al.'s output-embedding measure applied to our
generations: MiniLM cosine distance from the same matched set's "none" answer
(all-MiniLM-L6-v2, revision-pinned, sentence-transformers 6.0.0), averaged
over the 60 sets. The internal side is the registered layer-46 gradient
artifact. The primary statistic, the secondary, both diagnostics, and a
positive sign prediction were committed in `polreps/behavioral.py` before any
generation ran.

The pre-registered prediction failed, and the failure has a clear shape.
Internal displacement norm does not rank behavioral displacement (rho +0.112,
p 0.68), and the ideology-specific deviation does no better (rho -0.041).
What does rank it is surface engagement with the identity: the scaffold-echo
rate (+0.624, p 0.011) and the answer-length delta (+0.571, p 0.023), both
inside the 0.01-0.05 band the exploratory-work discipline says to distrust,
but matching what the example answers show, answers restructured around the
stated identity ("given your identity as an American Indian", a "Pacific
Islander Voter" section heading). The echo instrument also has a term-level
false-positive floor: topical template words ("party", "member") fire on
3-12% of the scaffold-free none answers for the partisan conditions, so
per-condition echo rates are read as approximate, not exact. The
partisan scaffolds are the sharpest cell: they carry the largest internal
norms and the entire ideology-axis separation, yet they sit near the bottom
of the behavioral spectrum (Republican 0.223, Democrat 0.209, against a
demographic top of 0.374 and a syntactic floor of 0.146). Read beside tickets
04 and 05 the pattern is consistent: the political axis is strongly
represented and weakly expressed, and the Cen-style output measure mostly
reads identity-conditioned tailoring rather than the politics the probe sees.
Caveats stated plainly: 16 conditions is a small n for any rank statistic;
one embedding model; one greedy generation per cell, so no within-cell
variance; the echo diagnostic is correlational, not an ablation. A seeded
draw of raw answers (2 per condition, never filtered) is in
`artifacts/gemma-3-12b-it/behavioral_link_examples.md`.

| date | stage | metric | value | seed | provenance | registered |
|---|---|---|---|---|---|---|
| 2026-08-27 | generate | base-question answers (17 conditions: identity scaffolds, paraphrases, controls, none; 60 sets; greedy, 100 new tokens) | 1,020 | 0 | `artifacts/gemma-3-12b-it/behavioral_generations.jsonl` | pre |
| 2026-08-27 | embed | behavioral spectrum, mean MiniLM cosine distance from the same set's none answer: top Pacific Islander / left-handed / Asian; bottom Democratic-party paraphrase / person; between-question reference over none answers 0.705 | 0.374 / 0.364 / 0.333; 0.208 / 0.146 | 0 | `artifacts/gemma-3-12b-it/behavioral_link.json` | pre |
| 2026-08-27 | correlate | primary statistic (sign prediction: positive): Spearman internal displacement norm at layer 46 vs behavioral displacement. Prediction not supported; informative null | +0.112 (p 0.68, n=16) | 0 | `artifacts/gemma-3-12b-it/behavioral_link.json` | pre |
| 2026-08-27 | correlate | secondary: \|projection deviation from the common offset\| vs behavioral displacement; layer-39 robustness for both (norm; deviation) | -0.041 (p 0.88); at 39 +0.403 (p 0.12), +0.279 (p 0.30) | 0 | `artifacts/gemma-3-12b-it/behavioral_link.json` | pre |
| 2026-08-27 | diagnose | boring-alternative diagnostics: scaffold-echo rate vs behavioral displacement; mean \|word-count delta\| vs behavioral displacement. Both in the exploratory-distrust band; read with the examples file | +0.624 (p 0.011); +0.571 (p 0.023) | 0 | `artifacts/gemma-3-12b-it/behavioral_link.json` | pre |
| 2026-08-27 | diagnose | echo rates at the spectrum ends (the correlation above is pre-registered; picking these cells is a reading of the observed spectrum): Pacific Islander / American Indian / Asian, Black, Hispanic, then man / person | 0.98 / 0.98 / 0.93 each, then 0.18 / 0.17 | 0 | `artifacts/gemma-3-12b-it/behavioral_link.json` | post |
| 2026-08-27 | read | partisan scaffolds, the largest internal displacements on the axis (Republican norm 23,647, deviation +1431.7), sit near the behavioral bottom: Republican / Democrat distance against the demographic top | 0.223 / 0.209 vs 0.374 | 0 | `artifacts/gemma-3-12b-it/behavioral_link.json` | post |
