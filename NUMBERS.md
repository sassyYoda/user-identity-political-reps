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
