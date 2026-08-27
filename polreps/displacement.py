"""Difference-in-means displacement vectors from a cached activation run.

For each scaffold condition and layer: the mean, over matched sets, of that
condition's last-token residual minus the "none" baseline residual of the same
set (CONTEXT.md "Displacement"). Pairing within sets is the point — the
base-question content cancels inside each pair, so the mean is a scaffold
direction, not a topic direction. These are milestone-2 input artifacts;
no analysis of them happens here (spec decision).
"""

import json
from pathlib import Path

import numpy as np

from polreps import actcache
from polreps.runmeta import save_run_metadata
from polreps.sweep import join_prompt_table

BASELINE = "none"


def displacement_means(acts, labels, set_ids, baseline=BASELINE):
    """(conditions, raw, n_sets) difference-in-means over matched sets.

    acts: (n_layers, n_rows, d_model); labels: condition per row; set_ids:
    matched-set key per row (pre_prompt_q_hash). Returns the sorted
    non-baseline conditions, a float32 (n_conditions, n_layers, d_model)
    array of paired mean differences, and per-condition matched-set counts —
    a condition missing from some sets is averaged over the sets that have
    it, with the support recorded so downstream can judge it.
    """
    row_of = {}
    for i, (cond, sid) in enumerate(zip(labels, set_ids)):
        if (sid, cond) in row_of:
            raise ValueError(f"duplicate condition {cond!r} in matched set {sid!r}")
        row_of[(sid, cond)] = i

    sets = sorted(set(set_ids))
    unanchored = [sid for sid in sets if (sid, baseline) not in row_of]
    if unanchored:
        raise ValueError(
            f"{len(unanchored)} matched set(s) have no {baseline!r} baseline "
            f"row (first: {unanchored[0]!r}) — displacement is undefined there"
        )
    conditions = sorted(set(labels) - {baseline})
    if not conditions:
        raise ValueError(f"only {baseline!r} rows present; nothing to displace")

    raw, n_sets = [], {}
    for cond in conditions:
        anchored = [sid for sid in sets if (sid, cond) in row_of]
        cond_rows = [row_of[(sid, cond)] for sid in anchored]
        base_rows = [row_of[(sid, baseline)] for sid in anchored]
        # paired mean of differences; float64 accumulation, fp32 artifact
        diff = acts[:, cond_rows].mean(axis=1, dtype=np.float64) - acts[
            :, base_rows
        ].mean(axis=1, dtype=np.float64)
        raw.append(diff.astype(np.float32))
        n_sets[cond] = len(anchored)
    return conditions, np.stack(raw), n_sets


def unit_normalize(raw):
    """Unit-norm copy of each (condition, layer) vector."""
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    if (norms == 0).any():
        raise ValueError(
            "zero displacement vector — a unit-norm direction is undefined"
        )
    return raw / norms


def run_displacements(cache_dir, labels_csv, out_stem):
    """The displacement stage: cache + prompt table in, vector artifacts out.

    Writes <out>.npz ("conditions", "raw", "unit") and a human-readable
    <out>.json manifest, each with a .meta.json sidecar.
    """
    acts, prompt_ids = actcache.load_cache(Path(cache_dir))
    labels, set_ids = join_prompt_table(
        labels_csv, prompt_ids, columns=("condition", "pre_prompt_q_hash")
    )
    conditions, raw, n_sets = displacement_means(acts, labels, set_ids)
    unit = unit_normalize(raw)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = out_stem.with_suffix(".npz")
    np.savez(arrays_path, conditions=np.array(conditions), raw=raw, unit=unit)

    manifest = {
        "baseline": BASELINE,
        "conditions": conditions,
        "n_sets": n_sets,
        "n_layers": int(raw.shape[1]),
        "d_model": int(raw.shape[2]),
    }
    manifest_path = out_stem.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    config = {"cache_dir": str(cache_dir), "labels_csv": str(labels_csv)}
    for artifact in (arrays_path, manifest_path):
        # a forward-pass byproduct with no randomness, hence seed=None
        save_run_metadata(artifact, seed=None, config=config)
    return manifest
