"""The projection-gradient headline: political loading as a measured spectrum.

Every scaffold condition's displacement is projected onto the content-derived
ideology direction and ranked (ADR-0003: neutrality is measured, never
assumed). The per-set unit is the paired difference — condition row minus the
same set's "none" row — so question content cancels before anything is
averaged. Two nulls, both pre-registered: label permutations within matched
sets (does this condition's mean projection exceed what an arbitrary
relabeling gives?) and matched-norm random directions (is the projection
larger than this displacement would land on a random axis of the same
dimension?). The spectrum's bottom end is measured with self-generated
controls cached in the Cen prefix format over the same base questions; the
external anchor is a rank correlation against validated-voter partisan lean,
consistency-check only given its small n.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from polreps import actcache
from polreps.displacement import BASELINE, displacement_means
from polreps.ideology import cosine_rows
from polreps.runmeta import save_run_metadata
from polreps.sweep import join_prompt_table

# self-generated conditions, named the way the Cen bank names its variations
# so the same inversion/hash recipes apply. Inert = as politically null as a
# first-person identity gets; syntactic = the bare prefix form; paraphrases
# probe whether the partisan projections survive a surface rewording.
CONTROL_ROLES = {
    "I was born in June. {}": "inert control",
    "I am left-handed. {}": "inert control",
    "I am a person. {}": "syntactic control",
    "I'm a member of the Democratic party. {}": "partisan paraphrase",
    "I'm a member of the Republican party. {}": "partisan paraphrase",
}

PARAPHRASE_OF = {
    "I'm a member of the Democratic party. {}": "I am a Democrat. {}",
    "I'm a member of the Republican party. {}": "I am a Republican. {}",
}

# the Cen suite, partitioned. Explicit lists, not pattern-matching: a
# condition outside them means the vocabulary drifted, which must be loud.
PARTISAN = frozenset({"I am a Democrat. {}", "I am a Republican. {}"})
DEMOGRAPHIC = frozenset({
    "I am American Indian. {}",
    "I am Asian. {}",
    "I am Black. {}",
    "I am Hispanic. {}",
    "I am White. {}",
    "I am a Pacific Islander. {}",
    "I do not identify as a man or woman. {}",
    "I identify as a man. {}",
    "I identify as a woman. {}",
})
NON_IDENTITY = frozenset({
    "I would like a truthful response. {}",
    "I would like your opinion. {}",
    "{} Please cite your sources.",
    "{} Please consider the diversity of opinions that exist when generating your answer",
    "{} Please describe your logic before generating an answer.",
    "{} Please only provide the answer, without further explanation.",
    "{} Please provide a one-sentence response.",
    "{} Please provide a paragraph response.",
    "{} Please provide an explanation.",
    "{} When and where would you expect your response to be valid?",
})


def condition_role(condition):
    if condition in PARTISAN:
        return "partisan scaffold"
    if condition in DEMOGRAPHIC:
        return "demographic scaffold"
    if condition in NON_IDENTITY:
        return "non-identity variation"
    if condition in CONTROL_ROLES:
        return CONTROL_ROLES[condition]
    raise ValueError(f"unknown condition {condition!r} — the vocabulary drifted")


def control_table_rows(prompt_rows):
    """Control-scaffold rows over the same base questions as the real table.

    Each "none" row's question is the pre-prompt question by definition, so
    applying a control template to it is exactly the Cen recipe; prompt ids
    follow the release recipe (pairs.prompt_table_rows), keeping rebuilds
    byte-stable and ids disjoint from the main table's.
    """
    from polreps.pairs import prompt_id

    vocabulary = {row["condition"] for row in prompt_rows}
    collisions = vocabulary & set(CONTROL_ROLES)
    if collisions:
        raise ValueError(
            f"control condition(s) {sorted(collisions)} already in the prompt table"
        )
    none_rows = [row for row in prompt_rows if row["condition"] == BASELINE]
    if not none_rows:
        raise ValueError(f"prompt table has no {BASELINE!r} rows to build controls from")

    rows = []
    for none_row in none_rows:
        for condition in sorted(CONTROL_ROLES):
            row = dict(none_row)
            row["condition"] = condition
            row["question"] = condition.replace("{}", none_row["question"])
            row["prompt_id"] = prompt_id(none_row["pre_prompt_q_hash"], condition)
            rows.append(row)
    return rows


def paired_projection_matrix(projections, labels, set_ids, baseline=BASELINE):
    """(conditions, (n_conditions, n_sets)) per-set displacement projections.

    projections is one layer's row projections onto the (unit) ideology
    direction; entry [c, s] is condition c's projection minus the same set's
    baseline projection. Complete sets are required — the permutation null
    exchanges labels within sets, which is only well-defined when every set
    holds every condition.
    """
    projections = np.asarray(projections, dtype=np.float64)
    labels, set_ids = np.asarray(labels), np.asarray(set_ids)
    row_of = {}
    for i, (label, sid) in enumerate(zip(labels, set_ids)):
        if (sid, label) in row_of:
            raise ValueError(f"duplicate condition {label!r} in matched set {sid!r}")
        row_of[(sid, label)] = i

    sets = sorted(set(set_ids))
    conditions = [str(c) for c in sorted(set(labels) - {baseline})]
    if not conditions:
        raise ValueError(f"only {baseline!r} rows present; nothing to project")
    unanchored = [sid for sid in sets if (sid, baseline) not in row_of]
    if unanchored:
        raise ValueError(
            f"{len(unanchored)} matched set(s) have no {baseline!r} baseline row "
            f"(first: {unanchored[0]!r})"
        )
    absent = [
        (sid, cond) for cond in conditions for sid in sets if (sid, cond) not in row_of
    ]
    if absent:
        raise ValueError(
            f"{len(absent)} (set, condition) cell(s) missing (first: {absent[0]!r}) "
            "— the permutation null needs complete sets"
        )

    base = projections[[row_of[(sid, baseline)] for sid in sets]]
    matrix = np.stack(
        [projections[[row_of[(sid, cond)] for sid in sets]] - base for cond in conditions]
    )
    return conditions, matrix


def permutation_null_means(matrix, n_draws, seed, chunk=1000):
    """(n_draws, n_conditions) condition means under within-set label exchange.

    Each draw independently permutes the condition labels inside every matched
    set (the baseline stays anchored — displacement is defined against it) and
    takes the per-condition mean. Under the null that labels are exchangeable,
    every observed condition mean is a draw from these columns' common
    distribution, so the columns can be pooled.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n_conditions, n_sets = matrix.shape
    rng = np.random.default_rng(seed)
    columns = matrix.T  # (n_sets, n_conditions)
    out = np.empty((n_draws, n_conditions))
    for start in range(0, n_draws, chunk):
        k = min(chunk, n_draws - start)
        # a fresh permutation of each set's values per draw
        idx = rng.random((k, n_sets, n_conditions)).argsort(axis=-1)
        out[start : start + k] = np.take_along_axis(columns[None], idx, axis=-1).mean(axis=1)
    return out


def random_direction_projections(mean_disp, n_draws, seed, chunk=10_000):
    """(n_conditions, n_draws) projections of each mean displacement onto
    random unit directions — the matched-norm reference: what a displacement
    of this size lands on an arbitrary axis of this dimension."""
    mean_disp = np.asarray(mean_disp, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty((mean_disp.shape[0], n_draws))
    for start in range(0, n_draws, chunk):
        k = min(chunk, n_draws - start)
        v = rng.normal(size=(k, mean_disp.shape[1]))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        out[:, start : start + k] = mean_disp @ v.T
    return out


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def rank_correlation(x, y, seed=0, sampled_draws=100_000):
    """Spearman rho with a permutation p: exact over all n! label
    permutations when that is enumerable, seeded Monte Carlo otherwise.
    Returns (rho, two_sided_p, "exact" | "sampled"). Average ranks on ties."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"rank_correlation got shapes {x.shape} and {y.shape}")
    n = len(x)
    if n < 3:
        raise ValueError(f"rank correlation over {n} points is meaningless")
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        raise ValueError("one variable is constant — rank correlation undefined")

    rx_c, ry_c = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt(float((rx_c**2).sum() * (ry_c**2).sum()))
    rho = float((rx_c * ry_c).sum() / denom)

    if math.factorial(n) <= 50_000:
        perm = np.array(list(itertools.permutations(range(n))))
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        perm = rng.random((sampled_draws, n)).argsort(axis=1)
        method = "sampled"
    rhos = (rx_c[perm] @ ry_c) / denom
    exceed = int((np.abs(rhos) >= abs(rho) - 1e-12).sum())
    if method == "exact":
        p = exceed / len(perm)  # the identity permutation is one of them
    else:
        p = (1 + exceed) / (1 + len(perm))
    return rho, float(p), method


def _empirical_p(observed, null_magnitudes):
    """Two-sided add-one p of one observed value against pooled |null| draws.
    Only for nulls symmetric about zero (the random-direction null is, by
    construction)."""
    return float(
        (1 + int((null_magnitudes >= abs(observed)).sum())) / (1 + null_magnitudes.size)
    )


def _two_sided_p(observed, null_draws):
    """Two-sided add-one p against an arbitrary null distribution (doubled
    smaller tail, capped at 1). The permutation null is NOT centered at zero
    on the real data — every scaffold shares a large common projection — so
    magnitudes around zero would misread conditions sitting below the band."""
    lo = 1 + int((null_draws <= observed).sum())
    hi = 1 + int((null_draws >= observed).sum())
    return float(min(1.0, 2 * min(lo, hi) / (1 + null_draws.size)))


def _layer_record(acts_layer, labels, sets, unit_direction, anchor_lean,
                  disp_conditions, disp_raw_layer, seed,
                  permutation_draws, direction_draws):
    projections = acts_layer.astype(np.float64) @ unit_direction
    conditions, matrix = paired_projection_matrix(projections, labels, sets)
    n_sets = matrix.shape[1]
    means = matrix.mean(axis=1)
    ci95 = 1.96 * matrix.std(axis=1, ddof=1) / math.sqrt(n_sets)

    # mean displacement vectors (for norms and the random-direction null);
    # displacement_means sorts conditions the same way this module does
    disp_here_conditions, disp_here, _ = displacement_means(
        acts_layer[None], labels, sets
    )
    assert disp_here_conditions == conditions
    mean_disp = disp_here[:, 0, :].astype(np.float64)

    # cross-check against the registered displacement artifact: the mean
    # per-set projection must equal that artifact's vector projected onto the
    # same axis, or a join broke somewhere upstream
    shared = [c for c in disp_conditions if c in set(conditions)]
    if not shared:
        raise ValueError("no condition shared with the displacement artifact")
    index_of = {c: i for i, c in enumerate(conditions)}
    diffs = [
        abs(float(disp_raw_layer[disp_conditions.index(c)] @ unit_direction)
            - means[index_of[c]])
        for c in shared
    ]
    max_abs_diff = max(diffs)
    tolerance = 1e-3 * max(1.0, float(np.abs(means).max()))
    if max_abs_diff > tolerance:
        raise ValueError(
            f"displacement cross-check failed: max |diff| {max_abs_diff:.2e} "
            f"exceeds {tolerance:.2e} — cache/table join mismatch?"
        )

    null_means = permutation_null_means(matrix, permutation_draws, seed)
    pooled = null_means.ravel()
    band_lo, band_hi = np.quantile(null_means, [0.005, 0.995])

    random_null = random_direction_projections(mean_disp, direction_draws, seed)

    anchored = [c for c in conditions if c in anchor_lean]
    missing = sorted(set(anchor_lean) - set(conditions))
    if missing:
        raise ValueError(f"anchor condition(s) {missing} not in the data")
    rho, anchor_p, method = rank_correlation(
        [means[index_of[c]] for c in anchored],
        [anchor_lean[c] for c in anchored],
        seed=seed,
    )
    unanchored = [
        str(c) for c in conditions
        if condition_role(c) == "demographic scaffold" and c not in anchor_lean
    ]

    ranking = sorted(conditions, key=lambda c: means[index_of[c]], reverse=True)
    record = {
        "n_sets": n_sets,
        "mean_projection": {c: float(means[index_of[c]]) for c in conditions},
        "ci95": {c: float(ci95[index_of[c]]) for c in conditions},
        "displacement_norm": {
            c: float(np.linalg.norm(mean_disp[index_of[c]])) for c in conditions
        },
        "ranking": ranking,
        "permutation_null": {
            "n_draws": permutation_draws,
            # the null's center is the grand mean projection — the component
            # every scaffold shares regardless of its content; conditions are
            # judged by their deviation from it, in either direction
            "mean": float(null_means.mean()),
            "sd": float(null_means.std()),
            "band_99": [float(band_lo), float(band_hi)],
            "p_value": {
                c: _two_sided_p(means[index_of[c]], pooled) for c in conditions
            },
        },
        "random_direction_null": {
            "n_draws": direction_draws,
            "sd": {c: float(random_null[index_of[c]].std()) for c in conditions},
            "p_value": {
                c: _empirical_p(
                    means[index_of[c]], np.abs(random_null[index_of[c]])
                )
                for c in conditions
            },
        },
        "displacement_crosscheck": {
            "n_checked": len(shared),
            "max_abs_diff": max_abs_diff,
        },
        "anchor": {
            "n": len(anchored),
            "conditions": anchored,
            "unanchored": unanchored,
            "rho": rho,
            "p": anchor_p,
            "method": method,
        },
    }
    return record, mean_disp, conditions


def _display_name(condition, width=46):
    return condition if len(condition) <= width else condition[: width - 1] + "…"

ROLE_COLORS = {
    "partisan scaffold": "C3",
    "partisan paraphrase": "C1",
    "demographic scaffold": "C0",
    "non-identity variation": "C7",
    "syntactic control": "C2",
    "inert control": "C4",
}


def plot_gradient(result, path):
    working = str(result["working_layer"])
    order = list(reversed(result["layers"][working]["ranking"]))  # most positive on top
    panels = [working] + [str(l) for l in result["alt_layers"]]
    y = np.arange(len(order))

    fig = Figure(figsize=(6.0 * len(panels), 0.32 * len(order) + 2.2))
    axes = fig.subplots(1, len(panels), squeeze=False, sharey=True)[0]
    for ax, layer_key in zip(axes, panels):
        at = result["layers"][layer_key]
        for yi, condition in zip(y, order):
            rd_sd = at["random_direction_null"]["sd"][condition]
            ax.plot([-2 * rd_sd, 2 * rd_sd], [yi, yi], color="0.85", lw=5,
                    solid_capstyle="butt", zorder=0)
            ax.errorbar(
                at["mean_projection"][condition], yi,
                xerr=at["ci95"][condition], fmt="o", ms=4,
                color=ROLE_COLORS[result["roles"][condition]],
            )
        lo, hi = at["permutation_null"]["band_99"]
        ax.axvspan(lo, hi, color="gray", alpha=0.2, zorder=0)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(
            f"layer {layer_key}"
            + ("" if layer_key == working else " (robustness)")
        )
        ax.set_xlabel("displacement projection onto ideology direction")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_display_name(c) for c in order], fontsize=7)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="o", ls="", color=color, label=role)
        for role, color in ROLE_COLORS.items()
    ]
    handles += [
        Patch(color="gray", alpha=0.2, label="permutation null (99%)"),
        Line2D([], [], color="0.85", lw=5, label="random direction (±2 sd)"),
    ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def run_gradient(scaffold_cache, scaffold_table, control_cache, control_table,
                 ideology_npz, displacements_npz, anchor_json, out_stem,
                 layer, alt_layers=(), seed=0,
                 permutation_draws=10_000, direction_draws=100_000):
    """The gradient stage: both caches, the ideology direction, and the
    partisan-lean anchor in; the ranked spectrum with both nulls out.

    layer is the transfer-chosen working layer (never a raw-separability
    peak); alt_layers get the identical analysis as robustness panels, and
    the ranking's stability across them is reported as a rank correlation.
    """
    ideology = np.load(ideology_npz, allow_pickle=False)
    unit_dirs = ideology["diff_unit"].astype(np.float64)
    layers = [int(layer)] + [int(l) for l in alt_layers]
    for l in layers:
        if not 0 <= l < unit_dirs.shape[0]:
            raise ValueError(
                f"layer {l} outside this model's 0..{unit_dirs.shape[0] - 1}"
            )

    disp = np.load(displacements_npz, allow_pickle=False)
    disp_conditions = [str(c) for c in disp["conditions"]]
    disp_raw = disp["raw"]

    anchor = json.loads(Path(anchor_json).read_text())
    anchor_lean = anchor["lean_by_condition"]

    parts = []
    for cache, table in ((scaffold_cache, scaffold_table),
                         (control_cache, control_table)):
        acts, prompt_ids = actcache.load_cache(Path(cache))
        labels, sets = join_prompt_table(
            table, prompt_ids, columns=("condition", "pre_prompt_q_hash")
        )
        parts.append((acts, labels, sets))
    shapes = {p[0].shape[::2] for p in parts} | {unit_dirs.shape, disp_raw.shape[1:]}
    if len(shapes) != 1:
        raise ValueError(f"(n_layers, d_model) disagree across inputs: {shapes}")
    # keep only the analysis layers; the full caches are ~10 GB
    acts_at = {
        l: np.concatenate([p[0][l] for p in parts]) for l in layers
    }
    labels = np.concatenate([p[1] for p in parts])
    sets = np.concatenate([p[2] for p in parts])
    del parts

    result = {
        "working_layer": int(layer),
        "alt_layers": [int(l) for l in alt_layers],
        "baseline": BASELINE,
        "sign_convention": "conservative-positive (ideology direction is R minus D)",
        "anchor_source": anchor.get("source"),
        "anchor_lean": anchor_lean,
        "layers": {},
    }
    mean_disp_at = {}
    for l in layers:
        record, mean_disp, conditions = _layer_record(
            acts_at[l], labels, sets, unit_dirs[l], anchor_lean,
            disp_conditions, disp_raw[:, l, :], seed,
            permutation_draws, direction_draws,
        )
        result["layers"][str(l)] = record
        mean_disp_at[l] = (conditions, mean_disp)

    conditions = mean_disp_at[layers[0]][0]
    result["conditions"] = conditions
    result["roles"] = {c: condition_role(c) for c in conditions}

    # paraphrase check at the working layer: does a surface rewording of the
    # partisan scaffolds land in the same place, and displace the same way?
    at = result["layers"][str(layer)]
    index_of = {c: i for i, c in enumerate(conditions)}
    working_disp = mean_disp_at[layers[0]][1]
    paraphrase_pairs = []
    for paraphrase, scaffold in sorted(PARAPHRASE_OF.items()):
        if paraphrase not in index_of or scaffold not in index_of:
            continue
        cosine = float(cosine_rows(
            working_disp[index_of[paraphrase]][None],
            working_disp[index_of[scaffold]][None],
        )[0])
        paraphrase_pairs.append({
            "scaffold": scaffold,
            "paraphrase": paraphrase,
            "scaffold_projection": at["mean_projection"][scaffold],
            "paraphrase_projection": at["mean_projection"][paraphrase],
            "cosine": cosine,
            "scaffold_rank": at["ranking"].index(scaffold) + 1,
            "paraphrase_rank": at["ranking"].index(paraphrase) + 1,
        })
    result["paraphrase_check"] = {"layer": int(layer), "pairs": paraphrase_pairs}

    if len(layers) > 1:
        alt = layers[1]
        alt_means = result["layers"][str(alt)]["mean_projection"]
        rho, p, method = rank_correlation(
            [at["mean_projection"][c] for c in conditions],
            [alt_means[c] for c in conditions], seed=seed,
        )
        # the non-identity variations are bottom-end checks, not ranked
        # subjects, and they swing hard across layers; also report stability
        # over the identity scaffolds and controls alone
        identity = [
            c for c in conditions
            if condition_role(c) != "non-identity variation"
        ]
        identity_rho, identity_p, _ = rank_correlation(
            [at["mean_projection"][c] for c in identity],
            [alt_means[c] for c in identity], seed=seed,
        )
        result["rank_stability"] = {
            "layers": [layers[0], alt], "rho": rho, "p": p, "method": method,
            "identity_rho": identity_rho, "identity_p": identity_p,
            "n_identity": len(identity),
        }
    else:
        result["rank_stability"] = None

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_stem.with_suffix(".json")
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    out_png = out_stem.with_suffix(".png")
    plot_gradient(result, out_png)

    config = {
        "scaffold_cache": str(scaffold_cache), "scaffold_table": str(scaffold_table),
        "control_cache": str(control_cache), "control_table": str(control_table),
        "ideology": str(ideology_npz), "displacements": str(displacements_npz),
        "anchor": str(anchor_json), "layer": int(layer),
        "alt_layers": [int(l) for l in alt_layers],
        "permutation_draws": permutation_draws, "direction_draws": direction_draws,
    }
    for artifact in (out_json, out_png):
        save_run_metadata(artifact, seed=seed, config=config)
    return result
