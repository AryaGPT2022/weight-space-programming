"""Structural and surgical weight edits. Every op mutates (model, arch) in place, is
deterministic, and never looks at data. Use `diff_report` to log what changed vs base.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from wsp.model import HEAD_DIM, LowRankLinear, NullSelfAttention, RemappedEmbedding, _get, _set


def _layer(model, arch, orig_layer):
    return model.bert.encoder.layer[arch["keep_layers"].index(orig_layer)]


def _new_linear(W, b):
    lin = nn.Linear(W.shape[1], W.shape[0], bias=b is not None).to(W.device)
    with torch.no_grad():
        lin.weight.copy_(W)
        if b is not None:
            lin.bias.copy_(b)
    return lin


@torch.no_grad()
def prune_heads(model, arch, remove: dict[int, list[int]]):
    """remove = {orig_layer: [orig_head_ids]}"""
    for l, heads in remove.items():
        spec = arch["layers"][str(l)]
        L = _layer(model, arch, l)
        keep_pos = [i for i, h in enumerate(spec["heads"]) if h not in set(heads)]
        idx = torch.cat([torch.arange(i * HEAD_DIM, (i + 1) * HEAD_DIM) for i in keep_pos]) \
            if keep_pos else torch.zeros(0, dtype=torch.long)
        s, o = L.attention.self, L.attention.output
        if keep_pos:
            for n in ("query", "key", "value"):
                lin = getattr(s, n)
                setattr(s, n, _new_linear(lin.weight[idx], lin.bias[idx]))
            s.num_attention_heads = len(keep_pos)
            s.all_head_size = len(keep_pos) * HEAD_DIM
        else:
            L.attention.self = NullSelfAttention()
        dev = o.dense.weight.device
        o.dense = _new_linear(o.dense.weight[:, idx.to(dev)], o.dense.bias.clone())
        spec["heads"] = [spec["heads"][i] for i in keep_pos]


@torch.no_grad()
def prune_heads_folded(model, arch, remove: dict[int, list[int]]):
    """Same as prune_heads, but folds each removed head's constant value-bias contribution
    (W_O[:,h] @ b_V[h]) into the attention output bias first."""
    for l, heads in remove.items():
        spec = arch["layers"][str(l)]
        L = _layer(model, arch, l)
        s, o = L.attention.self, L.attention.output
        if hasattr(s, "value"):
            for h in heads:
                i = spec["heads"].index(h)
                sl = slice(i * HEAD_DIM, (i + 1) * HEAD_DIM)
                o.dense.bias += o.dense.weight[:, sl] @ s.value.bias[sl]
    prune_heads(model, arch, remove)


@torch.no_grad()
def prune_ffn(model, arch, remove: dict[int, list[int]], fold_const: dict[int, np.ndarray] | None = None):
    """remove = {orig_layer: [orig_neuron_ids]}. fold_const[l][j] (optional) = a constant activation
    value to assume for removed neuron j; its contribution act*w_out is folded into the output bias."""
    for l, neurons in remove.items():
        spec = arch["layers"][str(l)]
        L = _layer(model, arch, l)
        rm = set(neurons)
        keep_pos = [i for i, n in enumerate(spec["ffn"]) if n not in rm]
        drop_pos = [i for i, n in enumerate(spec["ffn"]) if n in rm]
        dev = L.output.dense.weight.device
        kp = torch.tensor(keep_pos, dtype=torch.long, device=dev)
        b_out = L.output.dense.bias.clone()
        if fold_const is not None and l in fold_const:
            act = torch.as_tensor(np.asarray(fold_const[l]), dtype=b_out.dtype, device=dev)
            b_out += L.output.dense.weight[:, torch.tensor(drop_pos, device=dev)] @ act
        L.intermediate.dense = _new_linear(L.intermediate.dense.weight[kp], L.intermediate.dense.bias[kp])
        L.output.dense = _new_linear(L.output.dense.weight[:, kp], b_out)
        spec["ffn"] = [spec["ffn"][i] for i in keep_pos]


@torch.no_grad()
def drop_layers(model, arch, layers: list[int]):
    keep = [l for l in arch["keep_layers"] if l not in set(layers)]
    mods = [model.bert.encoder.layer[arch["keep_layers"].index(l)] for l in keep]
    model.bert.encoder.layer = nn.ModuleList(mods)
    model.config.num_hidden_layers = len(keep)
    arch["keep_layers"] = keep


@torch.no_grad()
def low_rank(model, arch, orig_layer: int, name: str, r: int):
    """Replace layer.<name> (a Linear) by its rank-r truncated SVD (data-free)."""
    L = _layer(model, arch, orig_layer)
    lin = _get(L, name)
    W = lin.weight.float()
    U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
    new = LowRankLinear(lin.in_features, lin.out_features, r, lin.bias is not None).to(W.device)
    new.A.weight.copy_((torch.diag(S[:r].sqrt()) @ Vh[:r]).to(W.device))
    new.B.weight.copy_((U[:, :r] @ torch.diag(S[:r].sqrt())).to(W.device))
    if lin.bias is not None:
        new.B.bias.copy_(lin.bias)
    _set(L, name, new)
    arch["layers"][str(orig_layer)].setdefault("low_rank", {})[name] = r


@torch.no_grad()
def project_out(W: torch.Tensor, direction, cols=None, alpha=1.0, axis=0):
    """Remove `direction` (in the output space, i.e. along rows if axis=0) from W[:, cols].
    Mirrors the blog's Exp-8 edit: subtract a direction from selected output columns."""
    v = torch.as_tensor(np.asarray(direction), dtype=W.dtype, device=W.device)
    v = v / v.norm()
    if cols is None:
        cols = torch.arange(W.shape[1], device=W.device)
    sub = W[:, cols]
    W[:, cols] = sub - alpha * torch.outer(v, v @ sub)


@torch.no_grad()
def shrink_vocab(model, arch, keep_ids: list[int], unk_id: int):
    E = model.bert.embeddings.word_embeddings
    n_orig = E.weight.shape[0] if not isinstance(E, RemappedEmbedding) else E.remap.shape[0]
    keep = sorted(set(keep_ids) | {unk_id})
    new = RemappedEmbedding(n_orig, len(keep), E.weight.shape[1]).to(E.weight.device)
    new.weight.copy_(E.weight[torch.tensor(keep, device=E.weight.device)])
    remap = torch.full((n_orig,), keep.index(unk_id), dtype=torch.long)
    remap[torch.tensor(keep)] = torch.arange(len(keep))
    new.remap.copy_(remap.to(new.remap.device))
    model.bert.embeddings.word_embeddings = new
    arch["vocab_keep"] = len(keep)


def diff_report(base_sd: dict, new_sd: dict, arch: dict | None = None):
    """Which tensors differ from base, and what fraction of base weights were changed/removed.
    Pass `arch` so renumbered layers (after drop_layers) are compared with their original ids."""
    if arch is not None:
        import re
        pat = re.compile(r"encoder\.layer\.(\d+)\.")
        new_sd = {pat.sub(lambda m: f"encoder.layer.{arch['keep_layers'][int(m.group(1))]}.", k): v
                  for k, v in new_sd.items()}
    total = sum(v.numel() for k, v in base_sd.items() if v.dtype.is_floating_point)
    changed, removed, touched = 0, 0, []
    for k, v in base_sd.items():
        if not v.dtype.is_floating_point:
            continue
        if k not in new_sd or new_sd[k].shape != v.shape:
            removed += v.numel() - (new_sd[k].numel() if k in new_sd else 0)
            touched.append(k)
            continue
        n = int((new_sd[k].cpu() != v.cpu()).sum())
        if n:
            changed += n
            touched.append(k)
    new_total = sum(v.numel() for v in new_sd.values() if v.dtype.is_floating_point)
    return {"tensors_touched": touched, "n_tensors_touched": len(touched),
            "weights_changed_in_place": changed, "weights_removed": removed,
            "pct_touched": 100 * (changed + removed) / total,
            "params_before": total, "params_after": new_total}
