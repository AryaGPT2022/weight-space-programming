"""Read-only, data-free views of the weights (0 forward passes).

All functions take a loaded model and return numpy/pandas. No inputs are ever run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from wsp.model import HEAD_DIM


def _np(t):
    return t.detach().float().cpu().numpy()


def layers(model):
    return list(model.bert.encoder.layer)


def head_mats(layer):
    """Per-head (Wq, Wk, Wv, Wo) with shapes (d,h),(d,h),(d,h),(h,d) acting on row vectors x@W."""
    s, o = layer.attention.self, layer.attention.output.dense
    if not hasattr(s, "query"):
        return []
    Wq, Wk, Wv = _np(s.query.weight).T, _np(s.key.weight).T, _np(s.value.weight).T
    bq, bk = _np(s.query.bias), _np(s.key.bias)
    Wo = _np(o.weight).T
    nh = Wq.shape[1] // HEAD_DIM
    out = []
    for h in range(nh):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        out.append(dict(Wq=Wq[:, sl], Wk=Wk[:, sl], Wv=Wv[:, sl], Wo=Wo[sl], bq=bq[sl], bk=bk[sl]))
    return out


def eff_rank(m):
    s = np.linalg.svd(m, compute_uv=False)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def head_table(model, arch):
    rows = []
    for pos, orig in enumerate(arch["keep_layers"]):
        L = layers(model)[pos]
        ln_g = _np(L.attention.output.LayerNorm.weight)
        for i, (hid, m) in enumerate(zip(arch["layers"][str(orig)]["heads"], head_mats(L))):
            QK = m["Wq"] @ m["Wk"].T
            OV = m["Wv"] @ m["Wo"]
            rows.append(dict(layer=orig, head=hid,
                             qk_fro=np.linalg.norm(QK), qk_spec=np.linalg.norm(QK, 2), qk_erank=eff_rank(QK),
                             ov_fro=np.linalg.norm(OV), ov_spec=np.linalg.norm(OV, 2), ov_erank=eff_rank(OV),
                             ov_fro_post_ln=np.linalg.norm(OV * ln_g[None, :]),
                             qk_sym=float(np.trace(QK) / (np.linalg.norm(QK) + 1e-9)),  # >0: "attend to same token type"
                             bq_dot_k=float(np.linalg.norm(m["bq"] @ m["Wk"].T))))
    return pd.DataFrame(rows)


def ffn_table(model, arch):
    rows = []
    for pos, orig in enumerate(arch["keep_layers"]):
        L = layers(model)[pos]
        Win, bin_ = _np(L.intermediate.dense.weight), _np(L.intermediate.dense.bias)
        Wout = _np(L.output.dense.weight).T  # (ffn, d)
        g = _np(L.output.LayerNorm.weight)
        for i, nid in enumerate(arch["layers"][str(orig)]["ffn"]):
            rows.append(dict(layer=orig, neuron=nid, in_norm=np.linalg.norm(Win[i]), bias=bin_[i],
                             out_norm=np.linalg.norm(Wout[i]), out_norm_post_ln=np.linalg.norm(Wout[i] * g)))
    return pd.DataFrame(rows)


def readout_direction(model):
    """Linearised relevance direction in the final residual stream (pre final LayerNorm scale):
    logit ≈ w_cls · tanh(W_pool · (γ ⊙ x̂ + β) + b). At tanh'≈1 the gradient wrt x̂ is γ ⊙ W_poolᵀ w_cls.
    Computed purely from weights."""
    w_cls = _np(model.classifier.weight)[0]
    W_pool = _np(model.bert.pooler.dense.weight)
    g = _np(layers(model)[-1].output.LayerNorm.weight)
    return g * (W_pool.T @ w_cls)


def embedding_table(model):
    E = model.bert.embeddings
    return dict(word=_np(E.word_embeddings.weight), pos=_np(E.position_embeddings.weight),
                type=_np(E.token_type_embeddings.weight))
