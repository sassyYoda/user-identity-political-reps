"""On-disk activation cache format, shared between caching (03) and probing (04).

A cache is a directory:

    <cache_dir>/
        index.json     {"prompt_ids": [...], "n_layers": L, "d_model": D, "hook": ...}
        layer_00.npy   float32, shape (n_prompts, d_model)
        ...
        layer_{L-1:02d}.npy

Row i of every layer file is the last-token residual vector for prompt_ids[i].
fp32 on disk regardless of compute dtype (spec decision). The caching stage may
write the layer files incrementally (e.g. via memmap), but a finished cache must
satisfy exactly this layout — load_cache() is the arbiter, and it refuses any
cache whose files disagree with index.json or whose prompt ids don't match the
caller's prompt table. Both stages go through this module; that is what keeps
the formats in lockstep.
"""

import json

import numpy as np


def save_cache(cache_dir, acts, prompt_ids, hook="resid_post"):
    """Write activations of shape (n_layers, n_prompts, d_model) as a cache."""
    acts = np.asarray(acts, dtype=np.float32)
    prompt_ids = list(prompt_ids)
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("duplicate prompt ids; the cache is keyed by prompt id")
    n_layers, n_prompts, d_model = acts.shape
    if n_prompts != len(prompt_ids):
        raise ValueError(
            f"{n_prompts} activation rows but {len(prompt_ids)} prompt ids"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(n_layers):
        np.save(cache_dir / f"layer_{layer:02d}.npy", acts[layer])
    index = {
        "prompt_ids": prompt_ids,
        "n_layers": n_layers,
        "d_model": d_model,
        "hook": hook,
    }
    (cache_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def load_cache(cache_dir, expect_prompt_ids=None):
    """Load a cache as ((n_layers, n_prompts, d_model) fp32, prompt_ids).

    Fails loudly on any inconsistency: layer files missing or misshapen
    relative to index.json, or prompt ids that don't exactly match
    expect_prompt_ids (order included). Downstream label joins assume row i
    is expect_prompt_ids[i], so a silent pass here would corrupt every number.
    """
    index = json.loads((cache_dir / "index.json").read_text())
    prompt_ids = index["prompt_ids"]

    if expect_prompt_ids is not None and list(expect_prompt_ids) != prompt_ids:
        raise ValueError(
            f"cache prompt ids do not match the prompt table: cache has "
            f"{len(prompt_ids)} ids, expected {len(list(expect_prompt_ids))}; "
            "order matters — refusing to load"
        )

    layers = []
    for layer in range(index["n_layers"]):
        path = cache_dir / f"layer_{layer:02d}.npy"
        if not path.exists():
            raise ValueError(f"cache incomplete: {path.name} is missing")
        arr = np.load(path)
        if arr.shape != (len(prompt_ids), index["d_model"]):
            raise ValueError(
                f"{path.name} has shape {arr.shape}, index.json says "
                f"({len(prompt_ids)}, {index['d_model']})"
            )
        layers.append(arr.astype(np.float32))
    return np.stack(layers), prompt_ids
