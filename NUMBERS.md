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
| 2026-08-26 | setup | pinned model revision (google/gemma-2-9b-it) | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | — | `artifacts/model_access_check.json` | pre |
