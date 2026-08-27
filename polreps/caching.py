"""Resumable last-token residual caching over the prompt table.

One forward pass per prompt (no generation), residual stream at every layer,
final pre-generation token only. The finished directory is exactly the
actcache layout — load_cache is the arbiter, and finalize() calls it before
declaring success. While a run is in flight the directory instead holds

    layer_XX.npy     full-size fp32 memmaps, filled row by row
    inflight.json    the index-to-be, written before the first forward pass
    done.txt         one prompt id per line, appended after each row lands

so a killed run resumes by diffing done.txt against the table, and an
unfinished cache (no index.json) can never be loaded by the probing stage.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from polreps import actcache
from polreps.config import MODEL_NAME, MODEL_REVISION, hf_token
from polreps.runmeta import save_run_metadata

HOOK = "resid_post"


def read_prompt_table(table_csv, limit=None):
    """(prompt_ids, questions) from a ticket-02 prompt table, in file order."""
    with open(table_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    prompt_ids = [row["prompt_id"] for row in rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("duplicate prompt ids in the prompt table")
    return prompt_ids, [row["question"] for row in rows]


class CacheWriter:
    """Incremental writer for one cache directory, safe to kill and reopen.

    The constructor decides which of three states the directory is in —
    finished (index.json present), in flight (inflight.json present), or
    fresh — and refuses to touch anything written for a different prompt
    table or shape.
    """

    def __init__(self, cache_dir, prompt_ids, n_layers, d_model):
        self.cache_dir = Path(cache_dir)
        self.prompt_ids = list(prompt_ids)
        if len(set(self.prompt_ids)) != len(self.prompt_ids):
            raise ValueError("duplicate prompt ids; the cache is keyed by prompt id")
        self.row_of = {pid: i for i, pid in enumerate(self.prompt_ids)}
        # same schema as actcache.save_cache writes, so finalize is a rename-free
        # "drop the bookkeeping, add index.json"
        self.index = {
            "prompt_ids": self.prompt_ids,
            "n_layers": n_layers,
            "d_model": d_model,
            "hook": HOOK,
        }

        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            if json.loads(index_path.read_text()) != self.index:
                raise ValueError(
                    f"{self.cache_dir} holds a finished cache for a different "
                    "prompt table or shape — refusing to touch it"
                )
            self.finished = True
            self.done = set(self.prompt_ids)
            self.layers = []
            return

        self.finished = False
        inflight_path = self.cache_dir / "inflight.json"
        if inflight_path.exists():
            if json.loads(inflight_path.read_text()) != self.index:
                raise ValueError(
                    f"the interrupted run in {self.cache_dir} used a different "
                    "prompt table or shape — refusing to resume onto it"
                )
            self.done = self._read_done()
            self.layers = [
                open_memmap(self.cache_dir / f"layer_{layer:02d}.npy", mode="r+")
                for layer in range(n_layers)
            ]
            expected = (len(self.prompt_ids), d_model)
            for layer, arr in enumerate(self.layers):
                if arr.shape != expected or arr.dtype != np.float32:
                    raise ValueError(
                        f"layer_{layer:02d}.npy is {arr.dtype} {arr.shape}, "
                        f"expected float32 {expected} — refusing to resume"
                    )
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "done.txt").unlink(missing_ok=True)
            self.done = set()
            self.layers = [
                open_memmap(
                    self.cache_dir / f"layer_{layer:02d}.npy", mode="w+",
                    dtype=np.float32, shape=(len(self.prompt_ids), d_model),
                )
                for layer in range(n_layers)
            ]
            # written after the memmaps exist: inflight.json present means the
            # layer files are all in place, so resume never sees a half-made dir
            inflight_path.write_text(json.dumps(self.index) + "\n")

    def _read_done(self):
        done_path = self.cache_dir / "done.txt"
        if not done_path.exists():
            return set()
        text = done_path.read_text()
        lines = text.split("\n")
        if text and not text.endswith("\n"):
            # a kill mid-append leaves a partial last line; drop it and let
            # that prompt be recomputed (row writes are idempotent)
            lines = lines[:-1]
        done = {line for line in lines if line}
        unknown = done - set(self.prompt_ids)
        if unknown:
            raise ValueError(
                f"done.txt lists {len(unknown)} prompt ids not in the table "
                "— refusing to resume"
            )
        return done

    @property
    def pending(self):
        return [pid for pid in self.prompt_ids if pid not in self.done]

    def write(self, prompt_id, acts):
        acts = np.asarray(acts, dtype=np.float32)
        n_layers, d_model = self.index["n_layers"], self.index["d_model"]
        if acts.shape != (n_layers, d_model):
            raise ValueError(
                f"activations for {prompt_id} have shape {acts.shape}, "
                f"expected ({n_layers}, {d_model})"
            )
        if not np.isfinite(acts).all():
            raise ValueError(
                f"non-finite activations for {prompt_id} — refusing to cache them"
            )
        row = self.row_of[prompt_id]
        for layer, arr in enumerate(self.layers):
            arr[row] = acts[layer]
            arr.flush()
        # the done record goes last: a kill between flush and append just
        # recomputes this one prompt
        with open(self.cache_dir / "done.txt", "a") as f:
            f.write(prompt_id + "\n")
        self.done.add(prompt_id)

    def finalize(self):
        if self.finished:
            return
        if self.pending:
            raise ValueError(f"{len(self.pending)} prompts still uncached")
        for arr in self.layers:
            arr.flush()
        (self.cache_dir / "index.json").write_text(
            json.dumps(self.index, indent=2) + "\n"
        )
        (self.cache_dir / "inflight.json").unlink()
        (self.cache_dir / "done.txt").unlink()
        # load_cache is the format's arbiter; a cache it rejects must blow up
        # here, not hours later in the probing stage
        actcache.load_cache(self.cache_dir, expect_prompt_ids=self.prompt_ids)
        self.finished = True


def cache_prompts(cache_dir, prompt_ids, questions, compute_fn, n_layers, d_model,
                  log_every=50):
    """Fill (or finish filling) a cache; returns how many prompts were computed.

    compute_fn(question) -> (n_layers, d_model) activations for one prompt.
    Already-cached prompts are skipped, so re-invoking after a kill completes
    the same cache without redoing finished work.
    """
    writer = CacheWriter(cache_dir, prompt_ids, n_layers, d_model)
    todo = set(writer.pending)
    if writer.done and todo:
        print(f"resuming: {len(writer.done)}/{len(prompt_ids)} prompts already cached")

    computed, started = 0, time.monotonic()
    for prompt_id, question in zip(prompt_ids, questions):
        if prompt_id not in todo:
            continue
        writer.write(prompt_id, compute_fn(question))
        computed += 1
        if log_every and computed % log_every == 0:
            rate = computed / (time.monotonic() - started)
            print(
                f"cached {len(writer.done)}/{len(prompt_ids)} "
                f"({rate:.2f} prompts/s)", flush=True,
            )
    writer.finalize()
    return computed


def format_chat_prompt(tokenizer, question):
    """One user turn, generation prompt appended: the last token of the result
    is the final pre-generation position we measure at."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


def pick_device(device=None):
    # imported here so the fast tests (fake compute_fn) never pay for torch
    import torch

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return device


def load_model(device=None):
    import torch
    from transformer_lens.model_bridge import TransformerBridge

    hf_token()  # puts HF_TOKEN from .env into the environment; Gemma is gated
    model = TransformerBridge.boot_transformers(
        MODEL_NAME, revision=MODEL_REVISION,
        device=pick_device(device), dtype=torch.bfloat16,
    )
    model.eval()
    return model


def last_token_resids(model, question):
    """(n_layers, d_model) fp32 residual stream at the last prompt token."""
    import torch

    text = format_chat_prompt(model.tokenizer, question)
    # the rendered template already starts with <bos>; prepending another
    # would shift every position (see the TransformerBridge tokenization notes)
    tokens = model.to_tokens(text, prepend_bos=False)

    names = [f"blocks.{layer}.hook_{HOOK}" for layer in range(model.cfg.n_layers)]
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            tokens, names_filter=names, return_cache_object=False
        )

    rows = []
    for layer in range(model.cfg.n_layers):
        # the bridge may key the cache by the alias or its canonical name
        for name in (f"blocks.{layer}.hook_{HOOK}", f"blocks.{layer}.hook_out"):
            if name in cache:
                rows.append(cache[name][0, -1])
                break
        else:
            raise KeyError(f"no {HOOK} activation cached for layer {layer}")
    return torch.stack(rows).float().cpu().numpy()


def run_caching(table_csv, cache_dir, device=None, limit=None):
    """The caching stage: prompt table in, finished activation cache out."""
    prompt_ids, questions = read_prompt_table(table_csv, limit=limit)
    device = pick_device(device)
    model = load_model(device)

    computed = cache_prompts(
        cache_dir, prompt_ids, questions,
        lambda q: last_token_resids(model, q),
        n_layers=model.cfg.n_layers, d_model=model.cfg.d_model,
    )

    # no randomness in a forward pass, hence seed=None; the revision is the
    # reproducibility anchor. A no-op re-invocation must not re-stamp the
    # sidecar — that would replace the record of the run that did the work
    meta_path = Path(cache_dir).with_name(Path(cache_dir).name + ".meta.json")
    if computed > 0 or not meta_path.exists():
        save_run_metadata(
            Path(cache_dir), seed=None,
            config={
                "table_csv": str(table_csv), "limit": limit,
                "n_prompts": len(prompt_ids), "hook": HOOK, "device": device,
            },
            model_revision=MODEL_REVISION,
        )
    return computed
