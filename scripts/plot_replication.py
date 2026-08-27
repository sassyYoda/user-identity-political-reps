"""Overlay the per-model probe curves on a normalized layer-depth axis.

    uv run python scripts/plot_replication.py

Defaults to the subject and replication models' probe_curve.json under their
model-scoped artifact directories; pass --curve LABEL=PATH pairs to override.
Writes artifacts/replication_probe_curve.png / .json (cross-model, so at the
artifacts root) and .meta.json sidecars. This is a sanity panel: both curves
are expected to be saturated (ticket-05 leakage finding), and its peaks must
not be used to pick layers.
"""

import argparse

from polreps.config import ARTIFACTS, MODEL_NAME, REPLICATION_MODEL_NAME, artifacts_dir
from polreps.replication import run_replication


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve", action="append", metavar="LABEL=PATH",
        help="repeatable; default: the subject and replication models' sweeps",
    )
    parser.add_argument("--out", default=str(ARTIFACTS / "replication_probe_curve"))
    args = parser.parse_args()

    if args.curve:
        curves = {}
        for spec in args.curve:
            label, sep, path = spec.partition("=")
            if not sep or not label or not path:
                parser.error(f"--curve wants LABEL=PATH, got {spec!r}")
            if label in curves:
                parser.error(f"--curve label {label!r} given twice")
            curves[label] = path
    else:
        curves = {
            name.split("/")[-1]: artifacts_dir(name) / "probe_curve.json"
            for name in (REPLICATION_MODEL_NAME, MODEL_NAME)
        }

    summary = run_replication(curves, args.out)

    for panel, overlay in summary.items():
        for label, v in overlay.items():
            print(
                f"{panel} {label}: peak {v['peak_accuracy']:.4f} at layer "
                f"{v['peak_layer']}/{v['n_layers'] - 1} (chance {v['chance']:.3f})"
            )
    print(f"wrote {args.out}.png / .json and metadata sidecars")


if __name__ == "__main__":
    main()
