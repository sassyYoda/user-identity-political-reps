"""Direction-transfer scoring: a fixed 1-D axis, a learned threshold, no probe.

The transfer test (CONTEXT.md) is the milestone's evidence-bearing
measurement: a direction derived on one side (scaffold displacements or
content-labeled statements) must separate the *other* side, which it never
saw. Both scorers are deliberately weaker than a probe. Content statements
are unpaired, so there the scorer projects onto the fixed direction and fits
only a scalar threshold (and its orientation) per training fold, grouped by
speaker so no member appears in both train and test. The scaffold side has
matched pairs — the same question under both scaffolds — so it gets a
zero-parameter paired comparison instead: within each matched set, does the
Republican-scaffold row project higher than the Democrat-scaffold row under
the a-priori conservative-positive convention? Pairing cancels the
question-content variance that would otherwise drown the axis, and fitting
nothing means an anti-aligned direction scores below chance instead of being
rescued.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.model_selection import GroupKFold

from polreps import actcache
from polreps.ideology import cosine_rows, diff_in_means_directions
from polreps.runmeta import save_run_metadata
from polreps.sweep import chance_level, join_prompt_table, shuffle_labels

# a transfer variant is informative only if its peak clears its own
# shuffled-label reference by this margin; variants at chance must not
# steer the working-layer choice
INFORMATIVE_MARGIN = 0.05


def project(acts, direction):
    """(n_layers, n_rows) projections onto one unit direction per layer.

    acts: (n_layers, n_rows, d_model); direction: (n_layers, d_model).
    """
    direction = np.asarray(direction, dtype=np.float64)
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero direction vector — projection is undefined")
    unit = direction / norms
    return np.einsum("lrd,ld->lr", acts.astype(np.float64), unit)


def threshold_accuracy(projections, labels, groups, n_splits=5):
    """Fold accuracies of a midpoint-threshold rule on 1-D projections.

    Per training fold: the threshold is the midpoint of the two class-mean
    projections and the orientation is whichever side the training means
    fall on — so a sign flip of the direction cannot change the score.
    """
    labels = np.asarray(labels)
    classes = sorted(set(labels))
    if len(classes) != 2:
        raise ValueError(f"binary scorer got classes {classes}")
    accs = []
    for train, test in GroupKFold(n_splits=n_splits).split(projections, labels, groups):
        mean_a = projections[train][labels[train] == classes[0]].mean()
        mean_b = projections[train][labels[train] == classes[1]].mean()
        cut = (mean_a + mean_b) / 2
        predicted_b = (
            projections[test] > cut if mean_b > mean_a else projections[test] < cut
        )
        accs.append(np.mean(predicted_b == (labels[test] == classes[1])))
    return np.array(accs)


def transfer_curve(acts, labels, groups, directions, n_splits=5):
    """Per-layer mean and fold accuracies for one direction per layer."""
    if directions.shape[0] != acts.shape[0]:
        raise ValueError(
            f"{directions.shape[0]} directions for {acts.shape[0]} layers"
        )
    projections = project(acts, directions)
    fold_accs = np.stack(
        [
            threshold_accuracy(layer_proj, labels, groups, n_splits=n_splits)
            for layer_proj in projections
        ]
    )
    return fold_accs


def paired_sign_accuracy(projections, labels, set_ids, pos, neg):
    """Zero-parameter paired scorer for the matched-pair side.

    For every matched set holding both a pos and a neg row, the pair is
    scored correct when the pos row projects higher — nothing is fit, the
    orientation is the a-priori sign convention (conservative-positive), so
    an anti-aligned direction lands *below* chance instead of being rescued.
    projections: (n_layers, n_rows). Returns ((n_layers,) accuracies, n_pairs).
    """
    labels, set_ids = np.asarray(labels), np.asarray(set_ids)
    row_of = {}
    for i, (label, sid) in enumerate(zip(labels, set_ids)):
        if (sid, label) in row_of:
            raise ValueError(f"duplicate label {label!r} in matched set {sid!r}")
        row_of[(sid, label)] = i
    paired = [
        (row_of[(sid, pos)], row_of[(sid, neg)])
        for sid in sorted(set(set_ids))
        if (sid, pos) in row_of and (sid, neg) in row_of
    ]
    if not paired:
        raise ValueError(f"no matched set holds both {pos!r} and {neg!r}")
    pos_rows, neg_rows = map(list, zip(*paired))
    diffs = projections[:, pos_rows] - projections[:, neg_rows]
    return (diffs > 0).mean(axis=1), len(paired)


def paired_transfer_curve(acts, labels, set_ids, directions, pos, neg):
    """Per-layer paired accuracies for one direction per layer."""
    if directions.shape[0] != acts.shape[0]:
        raise ValueError(
            f"{directions.shape[0]} directions for {acts.shape[0]} layers"
        )
    return paired_sign_accuracy(
        project(acts, directions), labels, set_ids, pos, neg
    )


def random_cosine_null(d_model, n_draws, seed):
    """Cosines of random unit-vector pairs in d_model dimensions — the null
    every reported alignment cosine is read against. One vector can be held
    fixed by rotational symmetry."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_draws, d_model))
    return v[:, 0] / np.linalg.norm(v, axis=1)


def empirical_p(cosines, null):
    """Two-sided add-one p per layer: how often a random direction beats the
    observed |cosine|."""
    exceed = (np.abs(null)[None, :] >= np.abs(cosines)[:, None]).sum(axis=1)
    return (1 + exceed) / (1 + len(null))


def shuffled_pair_labels(labels, set_ids, pos, neg, seed):
    """Labels with pos/neg swapped in a seeded half of the matched sets —
    the paired scorer's no-information reference."""
    labels, set_ids = np.asarray(labels), np.asarray(set_ids)
    rng = np.random.default_rng(seed)
    flip_sets = [sid for sid in sorted(set(set_ids)) if rng.random() < 0.5]
    flip = np.isin(set_ids, flip_sets)
    swapped = labels.copy()
    swapped[flip & (labels == pos)] = neg
    swapped[flip & (labels == neg)] = pos
    return swapped


def paired_variant(acts, labels, set_ids, directions, pos, neg, seed):
    """One paired transfer curve plus its swapped-pair reference, JSON-ready."""
    accs, n_pairs = paired_transfer_curve(acts, labels, set_ids, directions, pos, neg)
    shuffled, _ = paired_transfer_curve(
        acts, shuffled_pair_labels(labels, set_ids, pos, neg, seed),
        set_ids, directions, pos, neg,
    )
    return {
        "n_pairs": n_pairs,
        "chance": 0.5,
        "mean_accuracy": accs.tolist(),
        "shuffled_mean_accuracy": shuffled.tolist(),
    }


def scored_variant(acts, labels, groups, directions, n_splits, seed):
    """One transfer curve plus its shuffled-label reference, JSON-ready —
    the same record shape as sweep_variant so readers can compare."""
    fold_accs = transfer_curve(acts, labels, groups, directions, n_splits=n_splits)
    shuffled = transfer_curve(
        acts, shuffle_labels(labels, seed), groups, directions, n_splits=n_splits
    )
    return {
        "n_rows": int(len(labels)),
        "chance": chance_level(labels),
        "mean_accuracy": fold_accs.mean(axis=1).tolist(),
        "fold_accuracy": fold_accs.tolist(),
        "shuffled_mean_accuracy": shuffled.mean(axis=1).tolist(),
    }


def plot_transfer(curve, path):
    fig = Figure(figsize=(11, 4))
    ax_acc, ax_cos = fig.subplots(1, 2)
    layers = np.arange(curve["n_layers"])

    panels = [
        ("scaffold direction on content", curve["scaffold_to_content"], "-"),
        ("  (statements with no party token)",
         curve["scaffold_to_content_no_party_token"], "--"),
        ("content direction on scaffold pairs", curve["content_to_scaffold_diff"], "-"),
        ("  (ridge variant)", curve["content_to_scaffold_ridge"], ":"),
        ("  (direction from no-party-token statements)",
         curve["content_to_scaffold_clean_diff"], "--"),
    ]
    for label, variant, style in panels:
        ax_acc.plot(layers, variant["mean_accuracy"], style, lw=1.4, label=label)
    ax_acc.plot(
        layers, curve["scaffold_to_content"]["shuffled_mean_accuracy"],
        color="gray", ls="--", lw=1, label="shuffled labels",
    )
    ax_acc.axhline(0.5, color="black", ls=":", lw=1, label="chance")
    ax_acc.axvline(curve["working_layer"]["layer"], color="black", lw=0.8, alpha=0.4)
    ax_acc.set_xlabel("layer")
    ax_acc.set_ylabel("transfer accuracy")
    # full range: the paired scorer's below-chance excursions at sign-flipped
    # layers are data, not clutter
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_title("transfer accuracy")
    ax_acc.legend(frameon=False, fontsize=7)

    align = curve["alignment"]
    ax_cos.plot(layers, align["cosine_diff"], marker="o", ms=3,
                label="displacement vs diff-in-means")
    ax_cos.plot(layers, align["cosine_ridge"], ls=":",
                label="displacement vs ridge")
    band = 2 * align["null_sd"]
    ax_cos.axhspan(-band, band, color="gray", alpha=0.25,
                   label="random direction (±2 sd)")
    ax_cos.axhline(0, color="black", lw=0.5)
    ax_cos.axvline(curve["working_layer"]["layer"], color="black", lw=0.8, alpha=0.4)
    ax_cos.set_xlabel("layer")
    ax_cos.set_ylabel("cosine")
    ax_cos.set_title("Democrat-Republican displacement vs ideology direction")
    ax_cos.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=200)


def run_transfer_test(scaffold_cache, scaffold_table, content_cache, content_table,
                      displacements_npz, ideology_npz, out_stem, dem, rep,
                      n_splits=5, seed=0, null_draws=100_000):
    """The transfer stage: both caches and both direction artifacts in,
    transfer curves + alignment + working-layer choice out.

    Sign convention throughout: conservative-positive (scaffold axis is the
    Republican displacement minus the Democrat displacement; the ideology
    direction is already R minus D / dim1-oriented).
    """
    arrays = np.load(displacements_npz, allow_pickle=False)
    conditions = [str(c) for c in arrays["conditions"]]
    for name in (dem, rep):
        if name not in conditions:
            raise ValueError(
                f"condition {name!r} not in displacements ({conditions})"
            )
    scaffold_axis = (
        arrays["raw"][conditions.index(rep)] - arrays["raw"][conditions.index(dem)]
    )
    ideology = np.load(ideology_npz, allow_pickle=False)
    diff_dir, ridge_dir = ideology["diff_raw"], ideology["ridge_raw"]

    content_acts, content_ids = actcache.load_cache(Path(content_cache))
    parties, speakers, party_token = join_prompt_table(
        content_table, content_ids, columns=("party", "icpsr", "mentions_party")
    )
    scaffold_acts, scaffold_ids = actcache.load_cache(Path(scaffold_cache))
    cond_labels, sets = join_prompt_table(
        scaffold_table, scaffold_ids, columns=("condition", "pre_prompt_q_hash")
    )
    keep = np.isin(cond_labels, [dem, rep])
    scaffold_acts, cond_labels, sets = (
        scaffold_acts[:, keep], cond_labels[keep], sets[keep],
    )

    shapes = {
        "displacements": scaffold_axis.shape, "ideology": diff_dir.shape,
        "content cache": content_acts.shape[::2], "scaffold cache": scaffold_acts.shape[::2],
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"(n_layers, d_model) disagree across inputs: {shapes}")

    clean = party_token == "0"
    curve = {
        "dem_condition": dem,
        "rep_condition": rep,
        "sign_convention": "conservative-positive on both axes (R minus D)",
        "n_layers": int(content_acts.shape[0]),
        "scaffold_to_content": scored_variant(
            content_acts, parties, speakers, scaffold_axis, n_splits, seed
        ),
        "scaffold_to_content_no_party_token": scored_variant(
            content_acts[:, clean], parties[clean], speakers[clean],
            scaffold_axis, n_splits, seed,
        ),
        "content_to_scaffold_diff": paired_variant(
            scaffold_acts, cond_labels, sets, diff_dir, pos=rep, neg=dem, seed=seed
        ),
        "content_to_scaffold_ridge": paired_variant(
            scaffold_acts, cond_labels, sets, ridge_dir, pos=rep, neg=dem, seed=seed
        ),
        # reverse token-confound check: the same paired scoring under a
        # direction re-derived from only the statements with no party-family
        # token, so it cannot be reading "Democrat"/"Republican" vocabulary
        "content_to_scaffold_clean_diff": paired_variant(
            scaffold_acts, cond_labels, sets,
            diff_in_means_directions(content_acts[:, clean], parties[clean]),
            pos=rep, neg=dem, seed=seed,
        ),
    }

    null = random_cosine_null(scaffold_axis.shape[-1], null_draws, seed)
    cos_diff = cosine_rows(scaffold_axis, diff_dir)
    cos_ridge = cosine_rows(scaffold_axis, ridge_dir)
    curve["alignment"] = {
        "cosine_diff": [round(float(c), 4) for c in cos_diff],
        "cosine_ridge": [round(float(c), 4) for c in cos_ridge],
        "null_sd": float(null.std()),
        "null_draws": null_draws,
        "p_value_diff": [float(p) for p in empirical_p(cos_diff, null)],
    }

    # the working layer for downstream analyses: best transfer over the
    # variants that actually carry signal, at a layer where both extractors'
    # alignment cosines are positive (a sign-flipped layer would poison every
    # downstream projection). Everything feeding the choice is recorded so it
    # can be audited
    variant_names = [
        "scaffold_to_content", "scaffold_to_content_no_party_token",
        "content_to_scaffold_diff", "content_to_scaffold_ridge",
        "content_to_scaffold_clean_diff",
    ]
    informative = [
        name for name in variant_names
        if max(curve[name]["mean_accuracy"])
        > max(curve[name]["shuffled_mean_accuracy"]) + INFORMATIVE_MARGIN
    ]
    if not informative:
        raise ValueError(
            "no transfer variant clears its shuffled reference — there is no "
            "transfer signal to pick a working layer from; read the curves by hand"
        )
    floor = np.min(
        [curve[name]["mean_accuracy"] for name in informative], axis=0
    )
    eligible = (cos_diff > 0) & (cos_ridge > 0)
    if not eligible.any():
        raise ValueError(
            "no layer has positive alignment under both extractors — refusing "
            "to pick a working layer with an unstable sign"
        )
    layer = int(np.argmax(np.where(eligible, floor, -np.inf)))
    curve["working_layer"] = {
        "rule": (
            "argmax over layers with positive diff and ridge alignment cosines "
            "of the minimum accuracy across informative transfer variants "
            f"(peak > shuffled max + {INFORMATIVE_MARGIN})"
        ),
        "informative_variants": informative,
        "layer": layer,
        "min_informative_accuracy": float(floor[layer]),
        "scaffold_to_content": curve["scaffold_to_content"]["mean_accuracy"][layer],
        "content_to_scaffold": curve["content_to_scaffold_diff"]["mean_accuracy"][layer],
        "alignment_cosine": float(cos_diff[layer]),
        "alignment_p": curve["alignment"]["p_value_diff"][layer],
    }

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    curve_json = out_stem.with_suffix(".json")
    curve_json.write_text(json.dumps(curve, indent=2) + "\n")
    curve_png = out_stem.with_suffix(".png")
    plot_transfer(curve, curve_png)

    config = {
        "scaffold_cache": str(scaffold_cache), "scaffold_table": str(scaffold_table),
        "content_cache": str(content_cache), "content_table": str(content_table),
        "displacements": str(displacements_npz), "ideology": str(ideology_npz),
        "n_splits": n_splits, "null_draws": null_draws,
    }
    for artifact in (curve_json, curve_png):
        save_run_metadata(artifact, seed=seed, config=config)
    return curve
