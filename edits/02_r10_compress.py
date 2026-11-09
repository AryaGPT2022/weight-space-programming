"""Edit 02 - R10: compression informed by passes 1-3 (empirical activation statistics).

Programmer: Claude (Opus 5.5). Uses passes 1-3 (stats), then evaluated on the probe slate in later passes.
Two pre-declared sizes (no selection by score): r10_compress (7.25M encoder+head, matched to the
baselines) and r10_compress_xs (5.40M, -50% encoder).

The program
1. Final layer: CLS-only compute (exact).
2. Final layer FFN: keep the 64 neurons with the largest readout-projected variance at [CLS]
   (4 neurons carry 90% and 17 carry 99% of it). Mean-ablate the rest: fold E[a_i]·w_out_i at [CLS]
   into the output bias.
3. Final layer heads: drop the 5 heads with < 3% of the readout-projected variance (H11,H5,H3,H2,H8),
   folding their mean [CLS] contribution into W_O's bias.
4. Layers 0-4: mean-ablation importance u = Var(contribution ⊙ γ_next-LN) / E[residual energy],
   ranked per parameter across FFN neurons and heads. Remove the lowest until the target is met.
   PROTECTED (from Phase A + pass 1): token-match heads L0H8 L1H1 L2H9 L1H0 L2H4 and the
   confirmed positional heads L0H2 (prev) and L1H8 (next).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import pickle

import numpy as np
import torch
from safetensors.torch import load_file

from importance import importance
from wsp import model as M, surgery as S

ROOT = Path(__file__).resolve().parents[1]
PROTECT = {(0, 8), (1, 1), (2, 9), (1, 0), (2, 4), (0, 2), (1, 8)}
L5_FFN_KEEP = 64
L5_HEAD_MIN_SHARE = 0.03
HEAD_COST, NEURON_COST = 4 * 384 * 32 + 3 * 32, 384 * 2 + 1
VARIANTS = {"r10_compress": 7_250_689, "r10_compress_xs": 5_400_000}


@torch.no_grad()
def fold_heads(model, arch, l, heads, mu_head):
    """Add each removed head's mean contribution to attention.output.dense.bias, then remove it."""
    pos = arch["keep_layers"].index(l)
    o = model.bert.encoder.layer[pos].attention.output.dense
    for h in heads:
        i = arch["layers"][str(l)]["heads"].index(h)
        o.bias += torch.as_tensor(mu_head[i], dtype=o.bias.dtype)
    S.prune_heads(model, arch, {l: heads})


def build(name, target):
    st = pickle.load(open(ROOT / "logs/passes/base_stats.pkl", "rb"))
    m, t, a = M.load(M.CKPT / "base", "cpu")
    imp = importance(m, a, st, last_cls=True)

    # final layer
    u5 = imp[5]["u_ffn"]
    rm5 = np.argsort(u5)[: 1536 - L5_FFN_KEEP]
    share = imp[5]["u_head"] / imp[5]["u_head"].sum()
    rm_h5 = [int(h) for h in np.where(share < L5_HEAD_MIN_SHARE)[0]]

    # layers 0-4: global per-parameter ranking
    saved = len(rm5) * NEURON_COST + len(rm_h5) * HEAD_COST
    need = M.param_counts(m)["params_encoder_head"] - saved - target
    cand = []
    for l in range(5):
        cand += [(u / NEURON_COST, "f", l, i, NEURON_COST) for i, u in enumerate(imp[l]["u_ffn"])]
        cand += [(u / HEAD_COST, "h", l, h, HEAD_COST) for h, u in enumerate(imp[l]["u_head"])
                 if (l, h) not in PROTECT]
    cand.sort()
    rm_f, rm_h, got = {l: [] for l in range(5)}, {l: [] for l in range(5)}, 0
    for _, k, l, i, c in cand:
        if got >= need:
            break
        (rm_f if k == "f" else rm_h)[l].append(i)
        got += c

    # apply: fold means, then remove
    fold_heads(m, a, 5, rm_h5, imp[5]["mu_head"])
    S.prune_ffn(m, a, {5: rm5.tolist()}, fold_const={5: imp[5]["mu_ffn"][rm5]})
    for l in range(5):
        if rm_h[l]:
            fold_heads(m, a, l, rm_h[l], imp[l]["mu_head"])
        S.prune_ffn(m, a, {l: rm_f[l]}, fold_const={l: imp[l]["mu_ffn"][rm_f[l]]})
    a["cls_only_last"] = True
    M.make_cls_only(m.bert.encoder.layer[-1])

    rep = S.diff_report(load_file(M.CKPT / "base" / "model.safetensors"), m.state_dict(), a)
    rep.pop("tensors_touched")
    meta = {"script": Path(__file__).name, "passes_used": 3, "parent": "base", "stats": "logs/passes/base_stats.pkl",
            "heads_removed": {5: rm_h5, **{l: rm_h[l] for l in range(5)}},
            "ffn_removed": {5: len(rm5), **{l: len(rm_f[l]) for l in range(5)}}, **rep}
    sha = M.save(m, t, a, M.CKPT / name, meta=meta)
    print(name, M.param_counts(m), "heads removed", meta["heads_removed"], "ffn removed", meta["ffn_removed"], sha[:12])


if __name__ == "__main__":
    for n, tgt in VARIANTS.items():
        build(n, tgt)
