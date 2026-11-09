"""The forward-pass budget: the ONLY way the weight programmer may run the network while editing.

A "pass" = one call to `probe`, at most MAX_PAIRS (query, passage) pairs. It returns logits
(and optionally attentions / hidden states / input ids). Pairs must come from the probe split,
or be synthetic texts written by the programmer (like the blog's Exp 7 attention tool).
Every pass is appended to logs/ledger.jsonl; exceeding the budget raises.

    from wsp.harness import ForwardBudget
    fb = ForwardBudget("C25", budget=25)
    out = fb.probe("checkpoints/edit_03", pairs_df, want=("attentions",), note="why this pass")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wsp import model as M
from wsp.data import DATA, DATASETS

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "logs" / "ledger.jsonl"
MAX_PAIRS = 256


class BudgetExhausted(RuntimeError):
    pass


def _probe_qids():
    q = set()
    for d in DATASETS:
        for task in ("rerank", "cls"):
            df = pd.read_parquet(DATA / d / f"{task}.parquet", columns=["qid", "split"])
            q |= set(df.qid[df.split == "probe"])
    return q


def ledger(kind=None, run=None):
    if not LEDGER.exists():
        return []
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    return [r for r in rows if (kind is None or r["kind"] == kind) and (run is None or r.get("run") == run)]


class ForwardBudget:
    def __init__(self, run: str, budget: int):
        self.run, self.budget = run, budget
        self._probe = _probe_qids()

    @property
    def used(self):
        return len(ledger("pass", self.run))

    @property
    def remaining(self):
        return self.budget - self.used

    def probe(self, model_dir, pairs: pd.DataFrame, want=(), note=""):
        if self.remaining <= 0:
            raise BudgetExhausted(f"run {self.run}: all {self.budget} forward passes used")
        if len(pairs) > MAX_PAIRS:
            raise ValueError(f"a pass is at most {MAX_PAIRS} pairs (got {len(pairs)})")
        if "qid" in pairs and not set(pairs.qid.dropna()) <= self._probe:
            raise PermissionError("pairs outside the probe split - the test split is locked")
        model, tok, arch = M.load(model_dir)
        t = time.time()
        want = tuple(want)
        stats = None
        if "stats" in want:
            stats, hooks = _install_stat_hooks(model, arch)
            want = tuple(w for w in want if w != "stats")
        res = M.score(model, tok, pairs["query"].values, pairs["text"].values, want=want)
        logits, extra = (res if want else (res, {}))
        if stats is not None:
            for h in hooks:
                h.remove()
            extra["stats"] = _finish(stats)
        LEDGER.parent.mkdir(exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"kind": "pass", "run": self.run, "idx": self.used + 1,
                                "model": str(model_dir), "n_pairs": len(pairs), "want": list(want),
                                "synthetic": "qid" not in pairs, "note": note,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "secs": round(time.time() - t, 2)}) + "\n")
        out = {"logits": logits, **extra}
        if "label" in pairs:
            out["labels"] = pairs["label"].values
        # persist every pass immediately so an analysis bug can never waste one
        raw = LEDGER.parent / "passes" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(raw / f"{self.run}_{self.used:02d}.pkl", "wb") as f:
            pickle.dump({"pairs": pairs, **out}, f)
        return out


class _Mask:
    """Holds the current batch's padding mask; set from a pre-hook on the embeddings."""
    m = None


def _install_stat_hooks(model, arch):
    """Accumulate, over non-pad tokens (and separately at [CLS]):
      ffn{l}      : per-neuron sum / sumsq of GELU outputs
      head{l}     : per-head sum of contribution vectors ctx_h·W_O,hᵀ and sum of squared norms
      ln_att{l}, ln_out{l} : sum of per-token variance·d of the LayerNorm *input* (residual energy)
    Only collected inside a budgeted pass."""
    st, hooks = {}, []
    emb = model.bert.embeddings

    def emb_pre(mod, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        _Mask.m = (ids != 0)  # [PAD] = 0
    hooks.append(emb.register_forward_pre_hook(emb_pre, with_kwargs=True))

    def acc(key, x, cls_only):
        # x: (B,T,K) -> sums over valid tokens and over CLS
        mask = _Mask.m[:, : x.shape[1]].unsqueeze(-1).to(x.dtype)
        d = st.setdefault(key, {"n": 0, "s": 0, "ss": 0, "n_cls": 0, "s_cls": 0, "ss_cls": 0})
        if not cls_only:
            d["n"] += float(mask.sum())
            d["s"] = d["s"] + (x * mask).sum((0, 1)).float().cpu().double()
            d["ss"] = d["ss"] + (x * x * mask).sum((0, 1)).float().cpu().double()
        c = x[:, 0]
        d["n_cls"] += c.shape[0]
        d["s_cls"] = d["s_cls"] + c.sum(0).float().cpu().double()
        d["ss_cls"] = d["ss_cls"] + (c * c).sum(0).float().cpu().double()

    for pos, orig in enumerate(arch["keep_layers"]):
        L = model.bert.encoder.layer[pos]
        last = pos == len(arch["keep_layers"]) - 1 and arch.get("cls_only_last")

        hooks.append(L.intermediate.register_forward_hook(
            lambda mod, i, o, l=orig, c=last: acc(f"ffn{l}", o, c)))

        def head_hook(mod, args, l=orig, c=last, L=L):
            ctx = args[0]  # input to attention.output.dense : (B,T,nh*32)
            if ctx.shape[-1] == 0:
                return
            B, T, _ = ctx.shape
            W = L.attention.output.dense.weight  # (d, nh*32)
            nh = ctx.shape[-1] // M.HEAD_DIM
            contrib = torch.einsum("bthk,dhk->bthd", ctx.view(B, T, nh, M.HEAD_DIM),
                                   W.view(W.shape[0], nh, M.HEAD_DIM))  # (B,T,nh,d)
            acc(f"headvec{l}", contrib.reshape(B, T, -1), c)
        hooks.append(L.attention.output.dense.register_forward_pre_hook(head_hook))

        for nm, ln in (("ln_att", L.attention.output.LayerNorm), ("ln_out", L.output.LayerNorm)):
            def ln_hook(mod, args, l=orig, nm=nm, c=last):
                x = args[0]
                v = ((x - x.mean(-1, keepdim=True)) ** 2).sum(-1, keepdim=True)  # d·σ² per token
                acc(f"{nm}{l}", v, c)
            hooks.append(ln.register_forward_pre_hook(ln_hook))
    return st, hooks


def _finish(st):
    out = {}
    for k, d in st.items():
        out[k] = {kk: (vv.numpy() if torch.is_tensor(vv) else vv) for kk, vv in d.items()}
    return out


def merge_stats(*stats_list):
    out = {}
    for s in stats_list:
        for k, d in s.items():
            o = out.setdefault(k, {kk: 0 for kk in d})
            for kk, vv in d.items():
                o[kk] = o[kk] + vv
    return out


def probe_pool(task="cls", datasets=DATASETS):
    """The probe-split pairs the programmer may sample passes from (reading them costs nothing)."""
    return pd.concat([pd.read_parquet(DATA / d / f"{task}.parquet").assign(dataset=d)
                      .query("split == 'probe'") for d in datasets]).reset_index(drop=True)
