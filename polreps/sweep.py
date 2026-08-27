"""Per-layer logistic probes on cached activations.

The leakage-critical choices live here and are deliberately not configurable:
CV folds are grouped by base question (the same question must never appear in
train and test under different scaffold conditions), and standardization sits
inside the CV pipeline so its statistics are fit on training folds only.
"""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from polreps import actcache
from polreps.runmeta import save_run_metadata

# L2 at sklearn's default strength; probes are a separability measure, not a
# model we tune, so these stay fixed across every sweep
PROBE_C = 1.0
PROBE_MAX_ITER = 1000


def make_probe():
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER)
    )


def layer_accuracies(acts, labels, groups, n_splits=5):
    """Held-out accuracy for one probe per layer.

    acts: (n_layers, n_rows, d_model); labels: scaffold condition per row;
    groups: base-question id per row. Returns (n_layers, n_splits) fold
    accuracies; mean over axis 1 for the curve.
    """
    cv = GroupKFold(n_splits=n_splits)
    return np.stack(
        [
            cross_val_score(make_probe(), layer, labels, groups=groups, cv=cv)
            for layer in acts
        ]
    )


def chance_level(labels):
    """Accuracy of the best constant prediction (majority-class share)."""
    _, counts = np.unique(labels, return_counts=True)
    return counts.max() / len(labels)


def shuffle_labels(labels, seed):
    return np.random.default_rng(seed).permutation(labels)


def join_prompt_table(labels_csv, cache_prompt_ids):
    """Labels and groups from the prompt table, joined by id into cache row order."""
    with open(labels_csv, newline="") as f:
        table_rows = list(csv.DictReader(f))
    by_id = {row["prompt_id"]: row for row in table_rows}
    if len(by_id) != len(table_rows):
        raise ValueError(
            "duplicate prompt ids in the prompt table — refusing to pick one"
        )

    missing = [pid for pid in cache_prompt_ids if pid not in by_id]
    extra = set(by_id) - set(cache_prompt_ids)
    if missing or extra:
        raise ValueError(
            f"prompt table does not match the cache: {len(missing)} cached "
            f"prompts missing from the table, {len(extra)} table rows not in "
            "the cache — refusing to guess an alignment"
        )
    labels = np.array([by_id[pid]["condition"] for pid in cache_prompt_ids])
    groups = np.array([by_id[pid]["base_q_template_hash"] for pid in cache_prompt_ids])
    return labels, groups


def sweep_variant(acts, labels, groups, n_splits, seed, conditions=None):
    """One accuracy curve plus its shuffled-label reference, JSON-ready.

    conditions restricts to a subset (the binary Democrat-vs-Republican
    variant); None keeps every row for the multinomial probe.
    """
    if conditions is not None:
        absent = sorted(set(conditions) - set(labels))
        if absent:
            raise ValueError(
                f"conditions {absent} not in the data; observed vocabulary: "
                f"{sorted(set(labels))}"
            )
        keep = np.isin(labels, conditions)
        acts, labels, groups = acts[:, keep], labels[keep], groups[keep]
    fold_accs = layer_accuracies(acts, labels, groups, n_splits=n_splits)
    shuffled = layer_accuracies(
        acts, shuffle_labels(labels, seed), groups, n_splits=n_splits
    )
    return {
        "conditions": list(conditions) if conditions is not None else sorted(set(labels)),
        "n_rows": int(len(labels)),
        "chance": chance_level(labels),
        "mean_accuracy": fold_accs.mean(axis=1).tolist(),
        "fold_accuracy": fold_accs.tolist(),
        "shuffled_mean_accuracy": shuffled.mean(axis=1).tolist(),
    }


def plot_probe_curve(variants, path):
    """variants: {panel title: sweep_variant result}."""
    fig = Figure(figsize=(5.5 * len(variants), 4))
    for ax, (title, v) in zip(fig.subplots(1, len(variants), squeeze=False)[0], variants.items()):
        layers = np.arange(len(v["mean_accuracy"]))
        fold_accs = np.array(v["fold_accuracy"])
        ax.fill_between(
            layers, fold_accs.min(axis=1), fold_accs.max(axis=1), alpha=0.2, lw=0
        )
        ax.plot(layers, v["mean_accuracy"], marker="o", ms=3, label="held-out accuracy")
        ax.plot(
            layers, v["shuffled_mean_accuracy"],
            color="gray", ls="--", label="shuffled labels",
        )
        ax.axhline(v["chance"], color="black", ls=":", lw=1, label="chance")
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out accuracy")
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def run_sweep(cache_dir, labels_csv, out_stem, binary=None, n_splits=5, seed=0):
    """The sweep stage: cache + prompt table in, figure + numbers + metadata out."""
    acts, prompt_ids = actcache.load_cache(Path(cache_dir))
    labels, groups = join_prompt_table(labels_csv, prompt_ids)

    multi = sweep_variant(acts, labels, groups, n_splits, seed)
    variants = {"all conditions (multinomial)": multi}
    binary_variant = None
    if binary is not None:
        binary_variant = sweep_variant(
            acts, labels, groups, n_splits, seed, conditions=binary
        )
        variants[f"{binary[0]} vs {binary[1]}"] = binary_variant

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    curve = {
        "n_layers": int(acts.shape[0]),
        "multinomial": multi,
        "binary": binary_variant,
    }
    curve_json = out_stem.with_suffix(".json")
    curve_json.write_text(json.dumps(curve, indent=2) + "\n")
    curve_png = out_stem.with_suffix(".png")
    plot_probe_curve(variants, curve_png)

    config = {
        "cache_dir": str(cache_dir), "labels_csv": str(labels_csv),
        "n_splits": n_splits, "binary": list(binary) if binary else None,
    }
    for artifact in (curve_json, curve_png):
        save_run_metadata(artifact, seed=seed, config=config)
    return curve
