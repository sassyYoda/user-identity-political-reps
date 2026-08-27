"""Per-layer logistic probes on cached activations.

The leakage-critical choices live here and are deliberately not configurable:
CV folds are grouped by base question (the same question must never appear in
train and test under different scaffold conditions), and standardization sits
inside the CV pipeline so its statistics are fit on training folds only.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# L2 at sklearn's default strength; probes are a separability measure, not a
# model we tune, so these stay fixed across every sweep
PROBE_C = 1.0
PROBE_MAX_ITER = 1000


def probe():
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
            cross_val_score(probe(), layer, labels, groups=groups, cv=cv)
            for layer in acts
        ]
    )


def chance_level(labels):
    """Accuracy of the best constant prediction (majority-class share)."""
    _, counts = np.unique(labels, return_counts=True)
    return counts.max() / len(labels)


def shuffle_labels(labels, seed):
    return np.random.default_rng(seed).permutation(labels)
