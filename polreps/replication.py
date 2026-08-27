"""Two-model replication overlay of probe curves.

Takes run_sweep's probe_curve.json from each model and overlays the mean
accuracies on a normalized layer-depth axis (layer / (n_layers - 1)), since
the models differ in depth. Both curves must come from the same prompt table
— same conditions, same chance — or the overlay is not a replication, and
the guard below refuses it. Per the ticket-05 findings this figure is a
sanity panel: it shows whether the saturation shape replicates, and its
peaks must not be used to pick layers.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from polreps.runmeta import save_run_metadata

PANELS = ("multinomial", "binary")


def overlay_variants(curves, panel):
    """{model label: overlay dict} for one panel, or None if any model lacks it.

    curves: {model label: parsed probe_curve.json}. Chance must agree across
    models — the point of the overlay is the same task on two models.
    """
    variants = {label: curve.get(panel) for label, curve in curves.items()}
    if any(v is None for v in variants.values()):
        return None

    chances = {label: v["chance"] for label, v in variants.items()}
    if len(set(chances.values())) > 1:
        raise ValueError(
            f"{panel} chance differs across models ({chances}); the curves "
            "did not come from the same prompt table — refusing to overlay"
        )

    overlay = {}
    for label, v in variants.items():
        accs = v["mean_accuracy"]
        peak = int(np.argmax(accs))
        overlay[label] = {
            "n_layers": len(accs),
            "depth": (np.arange(len(accs)) / (len(accs) - 1)).tolist(),
            "mean_accuracy": accs,
            "shuffled_mean_accuracy": v["shuffled_mean_accuracy"],
            "chance": v["chance"],
            "peak_layer": peak,
            "peak_accuracy": accs[peak],
        }
    return overlay


def plot_replication(summary, path):
    fig = Figure(figsize=(5.5 * len(summary), 4))
    for ax, (panel, overlay) in zip(
        fig.subplots(1, len(summary), squeeze=False)[0], summary.items()
    ):
        for label, v in overlay.items():
            ax.plot(v["depth"], v["mean_accuracy"], marker="o", ms=3, label=label)
            ax.plot(v["depth"], v["shuffled_mean_accuracy"], ls="--", lw=1, alpha=0.5)
        chance = next(iter(overlay.values()))["chance"]
        ax.axhline(chance, color="black", ls=":", lw=1, label="chance")
        ax.set_xlabel("normalized layer depth")
        ax.set_ylabel("held-out accuracy")
        ax.set_ylim(0, 1.02)
        ax.set_title(panel)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def run_replication(curve_jsons, out_stem):
    """The overlay stage: per-model probe_curve.json paths in, figure + summary out.

    curve_jsons: {model label: path}. A panel is included only when every
    model has it (a binary curve with nothing to compare against is noise).
    """
    curves = {
        label: json.loads(Path(path).read_text())
        for label, path in curve_jsons.items()
    }
    summary = {}
    for panel in PANELS:
        overlay = overlay_variants(curves, panel)
        if overlay is not None:
            summary[panel] = overlay
    if not summary:
        raise ValueError(
            f"no panel shared across {sorted(curve_jsons)}; nothing to overlay"
        )

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    summary_json = out_stem.with_suffix(".json")
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    summary_png = out_stem.with_suffix(".png")
    plot_replication(summary, summary_png)

    config = {"curves": {label: str(path) for label, path in curve_jsons.items()}}
    for artifact in (summary_json, summary_png):
        # a deterministic replot of already-recorded sweeps, hence seed=None
        save_run_metadata(artifact, seed=None, config=config)
    return summary
