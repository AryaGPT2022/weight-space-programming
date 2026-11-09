"""Edit 01 - R0: zero-forward-pass compression, written from the weights alone.

Programmer: Claude (Opus 5.5). Budget used: 0 passes. Target: 7.25M encoder+head params, the same
as the data-free baselines, so the comparison is fair.

The program
1. Final layer: only [CLS] reaches the classifier, so compute it CLS-only (exact, ~16% faster).
2. Final-layer FFN: keep the 384 / 1536 neurons with the most readout-visible output energy,
       e_i = Var[gelu(pre_i)] · ‖R w_out_i‖²,   R = diag(w_cls) · W_pool · diag(γ_final)
   with pre_i ~ N(w_in_i·β + b_i, ‖w_in_i ⊙ γ‖²) (data-free activation model). Fold each removed
   neuron's modeled mean E[gelu(pre_i)]·w_out_i into the output bias.
3. Final-layer attention: drop the 4 heads whose OV circuit writes least onto the readout,
   ‖R diag(γ_att) (W_V W_O)ᵀ‖_F, folding their constant value-bias contribution into W_O's bias.
4. Layers 0-4: remove FFN neurons in order of increasing modeled output energy
       Var[gelu(pre_i)] · ‖w_out_i ⊙ γ_out‖²
   normalised by the layer's total, until the parameter target is met. Fold modeled means.
   Attention heads in layers 0-4 are left intact (positional and token-match circuits).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from safetensors.torch import load_file
from scipy.stats import norm

from wsp import model as M, surgery as S
from wsp.inspect import _np, head_mats

NAME = "r0_compress"
TARGET_ENC = 7_250_689          # = bl_mag_uniform33 (the baselines are 7.246-7.251M)
L5_FFN_KEEP = 384
L5_HEADS_DROP = 4
NEURON_PARAMS = 384 * 2 + 1


def gelu_moments(mu, sd, n=2001):
    z = np.linspace(-6, 6, n)
    w = norm.pdf(z)
    w /= w.sum()
    x = mu[:, None] + sd[:, None] * z[None, :]
    g = x * norm.cdf(x)
    m1 = (g * w).sum(1)
    return m1, (g * g * w).sum(1) - m1 ** 2


def ffn_model(L):
    g_in, b_in = _np(L.attention.output.LayerNorm.weight), _np(L.attention.output.LayerNorm.bias)
    Win, b = _np(L.intermediate.dense.weight), _np(L.intermediate.dense.bias)
    mu, sd = Win @ b_in + b, np.linalg.norm(Win * g_in[None, :], axis=1)
    return gelu_moments(mu, sd)


def main():
    m, t, a = M.load(M.CKPT / "base", "cpu")
    Ls = m.bert.encoder.layer
    w_cls = _np(m.classifier.weight)[0]
    R = (w_cls[:, None] * _np(m.bert.pooler.dense.weight)) * _np(Ls[5].output.LayerNorm.weight)[None, :]

    # --- final layer FFN (readout-visible energy)
    mean5, var5 = ffn_model(Ls[5])
    Wout5 = _np(Ls[5].output.dense.weight).T
    e5 = var5 * np.linalg.norm(Wout5 @ R.T, axis=1) ** 2
    rm5 = np.argsort(e5)[: 1536 - L5_FFN_KEEP]

    # --- final layer heads (readout-visible OV)
    g_att5 = _np(Ls[5].attention.output.LayerNorm.weight)
    ov_r = [np.linalg.norm(R @ np.diag(g_att5) @ (h["Wv"] @ h["Wo"]).T) for h in head_mats(Ls[5])]
    rm_heads5 = np.argsort(ov_r)[:L5_HEADS_DROP].tolist()

    # --- layers 0-4: global ranking by layer-normalised modeled energy
    cand = []
    stats = {}
    for l in range(5):
        mean, var = ffn_model(Ls[l])
        Wout = _np(Ls[l].output.dense.weight).T
        e = var * np.linalg.norm(Wout * _np(Ls[l].output.LayerNorm.weight)[None, :], axis=1) ** 2
        e = e / e.sum()
        stats[l] = mean
        cand += [(e[i], l, i) for i in range(1536)]
    cand.sort()
    enc_now = M.param_counts(m)["params_encoder_head"]
    saved_l5 = (1536 - L5_FFN_KEEP) * NEURON_PARAMS + L5_HEADS_DROP * (4 * 384 * 32 + 3 * 32)
    n_more = int(np.ceil((enc_now - saved_l5 - TARGET_ENC) / NEURON_PARAMS))
    rm = {l: [] for l in range(5)}
    for _, l, i in cand[:n_more]:
        rm[l].append(i)

    # --- apply (folding modeled means)
    S.prune_heads_folded(m, a, {5: rm_heads5})
    S.prune_ffn(m, a, {5: rm5.tolist()}, fold_const={5: mean5[rm5]})
    S.prune_ffn(m, a, rm, fold_const={l: stats[l][rm[l]] for l in rm})
    a["cls_only_last"] = True
    M.make_cls_only(m.bert.encoder.layer[-1])

    rep = S.diff_report(load_file(M.CKPT / "base" / "model.safetensors"), m.state_dict(), a)
    rep.pop("tensors_touched")
    meta = {"script": Path(__file__).name, "passes_used": 0, "parent": "base",
            "l5_heads_removed": rm_heads5, "ffn_removed_per_layer": {l: len(v) for l, v in rm.items()} | {5: len(rm5)},
            **rep}
    sha = M.save(m, t, a, M.CKPT / NAME, meta=meta)
    print(NAME, M.param_counts(m), meta["ffn_removed_per_layer"], "L5 heads removed", rm_heads5, sha[:12])


if __name__ == "__main__":
    main()
