"""Reference ideology direction from content-labeled activations, per layer.

Two extractors on the content corpus cache, with a conservative-positive sign
convention matching DW-NOMINATE's first dimension: difference-in-means (mean
Republican-speaker activation minus mean Democrat-speaker activation) and the
weight vector of a ridge regression onto the speakers' nominate_dim1 scores.
Their per-layer cosine is the internal robustness check the ticket asks for,
with a caveat: party is nearly a function of dim1's sign, so the two
estimators share their labels and can agree even on label-noise fits —
agreement is necessary, not sufficient, and the transfer test stays the
real arbiter. Because the
probe curve is saturated, no layer is chosen here: every layer's direction is
saved and the working layer is picked downstream by transfer accuracy and
displacement alignment, never by separability.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

from polreps import actcache
from polreps.displacement import unit_normalize
from polreps.runmeta import save_run_metadata
from polreps.sweep import join_prompt_table

# fixed grid, generalized-CV selection: the direction must not depend on a
# hand-tuned penalty. The chosen alphas are recorded per layer; picking the
# top endpoint means "no linear signal here worth keeping", which is
# information, not an error
RIDGE_ALPHAS = tuple(float(a) for a in np.logspace(0, 8, 9))


def diff_in_means_directions(acts, parties):
    """(n_layers, d_model) mean-R minus mean-D per layer, float32."""
    parties = np.asarray(parties)
    for party in ("D", "R"):
        if not (parties == party).any():
            raise ValueError(f"no {party!r} rows — cannot take a party difference")
    d_rows, r_rows = parties == "D", parties == "R"
    diff = acts[:, r_rows].mean(axis=1, dtype=np.float64) - acts[:, d_rows].mean(
        axis=1, dtype=np.float64
    )
    return diff.astype(np.float32)


def ridge_directions(acts, scores, alphas=RIDGE_ALPHAS):
    """(n_layers, d_model) ridge weights onto nominate_dim1 per layer, plus the
    per-layer chosen alpha. Features are used raw (centered by the intercept),
    so the weights live in the same activation space as the mean difference."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != acts.shape[1]:
        raise ValueError(f"{len(scores)} scores for {acts.shape[1]} activation rows")
    directions, chosen = [], []
    for layer_acts in acts:
        fit = RidgeCV(alphas=alphas).fit(layer_acts.astype(np.float64), scores)
        directions.append(fit.coef_.astype(np.float32))
        chosen.append(float(fit.alpha_))
    return np.stack(directions), chosen


def cosine_rows(a, b):
    """Per-row cosine between two (n, d) arrays."""
    num = (a.astype(np.float64) * b.astype(np.float64)).sum(axis=-1)
    return num / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def run_ideology_directions(cache_dir, labels_csv, out_stem):
    """The extraction stage: content cache + corpus table in, per-layer
    direction artifacts out (<out>.npz with raw and unit copies of both
    extractors, <out>.json manifest with the cosine check)."""
    acts, prompt_ids = actcache.load_cache(Path(cache_dir))
    parties, scores = join_prompt_table(
        labels_csv, prompt_ids, columns=("party", "nominate_dim1")
    )

    diff_raw = diff_in_means_directions(acts, parties)
    ridge_raw, alphas = ridge_directions(acts, scores.astype(np.float64))
    cosines = cosine_rows(diff_raw, ridge_raw)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = out_stem.with_suffix(".npz")
    np.savez(
        arrays_path,
        diff_raw=diff_raw, diff_unit=unit_normalize(diff_raw),
        ridge_raw=ridge_raw, ridge_unit=unit_normalize(ridge_raw),
    )

    manifest = {
        "sign_convention": "positive points conservative (R minus D; dim1 target)",
        "n_rows": int(acts.shape[1]),
        "n_layers": int(acts.shape[0]),
        "d_model": int(acts.shape[2]),
        "diff_ridge_cosine": [round(float(c), 4) for c in cosines],
        "ridge_alpha": alphas,
    }
    manifest_path = out_stem.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    config = {"cache_dir": str(cache_dir), "labels_csv": str(labels_csv)}
    for artifact in (arrays_path, manifest_path):
        # deterministic given the cache (GCV ridge, no sampling), hence seed=None
        save_run_metadata(artifact, seed=None, config=config)
    return manifest
