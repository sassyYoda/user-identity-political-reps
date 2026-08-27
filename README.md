# user-identity-political-reps

Does user-identity scaffolding ("I am a Democrat", "I am a woman", ...) displace
Gemma-2-9B-IT's activations along an interpretable ideology direction? A
mechanistic follow-up to the behavioral findings of Cen et al. 2025
(arXiv:2509.18446).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Hugging Face account with access
to the gated [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
repo.

```
uv sync
echo 'HF_TOKEN=hf_...' > .env
uv run python scripts/check_model_access.py
```

The check confirms the token can see the gated model at the pinned revision and
writes `artifacts/model_access_check.json`.

## Layout

- `polreps/` — the package: configuration, run metadata, and (as milestones land)
  pair reconstruction, activation caching, and probing
- `scripts/` — thin entry points, one per stage
- `tests/` — `uv run pytest` runs the fast suite; tests marked `slow` (network or
  real model) run with `uv run pytest -m slow`
- `data/raw/`, `activations/`, `artifacts/` — downloads, cached residual-stream
  activations, and produced figures/vectors; all gitignored, all re-derivable
- `NUMBERS.md` — source of truth for every measured result

## Reproducibility

Dependencies are pinned in `pyproject.toml` and `uv.lock`. The subject model
revision is pinned by commit hash in `polreps/config.py`. Every produced artifact
gets a `.meta.json` sidecar (seed, versions, model revision, git state) via
`polreps.runmeta`.
