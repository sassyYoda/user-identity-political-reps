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
