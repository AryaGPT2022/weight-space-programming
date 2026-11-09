"""Data-free, non-agent compression baselines. An agent edit counts only if it beats these at a
matched encoder-parameter count.

  bl_drop_top2     : drop layers 4,5 (top-layer dropping, Sajjad et al. 2023)           -> 4 layers
  bl_drop_alt2     : drop layers 1,3                                                   -> 4 layers
  bl_mag_uniform33 : in every layer remove the 4/12 heads with smallest weight L2 norm and the
                     512/1536 FFN neurons with smallest ||w_in||·||w_out||               -> ~-33% encoder

    uv run python edits/baselines.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from safetensors.torch import load_file

from wsp import model as M, surgery as S
from wsp.model import HEAD_DIM

BASE = M.CKPT / "base"


def mag_uniform(frac=1 / 3):
    m, t, a = M.load(BASE)
    rm_h, rm_f = {}, {}
    for l, L in enumerate(m.bert.encoder.layer):
        s, o = L.attention.self, L.attention.output.dense
        norms = []
        for h in range(12):
            sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            norms.append(sum(float(w.norm() ** 2) for w in (s.query.weight[sl], s.key.weight[sl],
                                                             s.value.weight[sl], o.weight[:, sl])) ** .5)
        rm_h[l] = list(np.argsort(norms)[: round(12 * frac)].tolist())
        sc = (L.intermediate.dense.weight.norm(dim=1) * L.output.dense.weight.norm(dim=0)).detach().cpu().numpy()
        rm_f[l] = np.argsort(sc)[: round(1536 * frac)].tolist()
    S.prune_heads(m, a, rm_h)
    S.prune_ffn(m, a, rm_f)
    return m, t, a, {"heads_removed": rm_h}


def drop(layers):
    m, t, a = M.load(BASE)
    S.drop_layers(m, a, layers)
    return m, t, a, {}


if __name__ == "__main__":
    base_sd = load_file(BASE / "model.safetensors")
    for name, fn in [("bl_drop_top2", lambda: drop([4, 5])), ("bl_drop_alt2", lambda: drop([1, 3])),
                     ("bl_mag_uniform33", mag_uniform),
                     # size-matched to the -50% variant (r10_compress_xs, 5.40M encoder+head)
                     ("bl_drop_alt3", lambda: drop([1, 3, 5])),
                     ("bl_mag_uniform50", lambda: mag_uniform(0.5))]:
        m, t, a, info = fn()
        rep = S.diff_report(base_sd, m.state_dict(), a)
        rep.pop("tensors_touched")
        sha = M.save(m, t, a, M.CKPT / name, meta={"script": "baselines.py", "kind": "baseline", **info, **rep})
        print(name, M.param_counts(m), f"{rep['pct_touched']:.1f}% touched", sha[:12])
