"""Empirical mean-ablation importance of every FFN neuron and attention head (from pass stats).
u = Var(contribution) after the next LayerNorm's γ, as a fraction of that LayerNorm's input energy.
Final layer: CLS-only, projected onto the readout matrix R."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle, numpy as np
from wsp import model as M
from wsp.inspect import _np
HD = 32

def mom(d, cls):
    n = d["n_cls"] if cls else d["n"]; s = d["s_cls"] if cls else d["s"]; ss = d["ss_cls"] if cls else d["ss"]
    mu = s / n; return mu, np.maximum(ss / n - mu ** 2, 0)

def importance(model, arch, st, last_cls=True):
    Ls = model.bert.encoder.layer
    w_cls = _np(model.classifier.weight)[0]
    R = (w_cls[:, None] * _np(model.bert.pooler.dense.weight)) * _np(Ls[-1].output.LayerNorm.weight)[None, :]
    out = {}
    for pos, l in enumerate(arch["keep_layers"]):
        L = Ls[pos]; last = pos == len(Ls) - 1
        cls = last and last_cls
        go, ga = _np(L.output.LayerNorm.weight), _np(L.attention.output.LayerNorm.weight)
        E_out = mom(st[f"ln_out{l}"], cls)[0][0]; E_att = mom(st[f"ln_att{l}"], cls)[0][0]   # mean d·σ²
        mu_a, var_a = mom(st[f"ffn{l}"], cls)
        Wout = _np(L.output.dense.weight).T                                           # (n_ffn, d)
        if last:
            u_f = var_a * np.linalg.norm((Wout * go) @ R.T, axis=1) ** 2 / E_out
        else:
            u_f = var_a * np.linalg.norm(Wout * go, axis=1) ** 2 / E_out
        hk = f"headvec{l}"
        if hk in st:
            mu_h, var_h = mom(st[hk], cls)
            nh = len(mu_h) // 384
            mu_h, var_h = mu_h.reshape(nh, 384), var_h.reshape(nh, 384)
            if last:
                # diagonal approximation of the readout-projected variance
                Rdiag = (R ** 2).sum(0) * ga ** 2 * go ** 2
                u_h = (var_h * Rdiag).sum(1) / E_att
            else:
                u_h = (var_h * ga ** 2).sum(1) / E_att
        else:
            mu_h, u_h = np.zeros((0, 384)), np.zeros(0)
        out[l] = dict(u_ffn=u_f, mu_ffn=mu_a, u_head=u_h, mu_head=mu_h,
                      ffn_ids=arch["layers"][str(l)]["ffn"], head_ids=arch["layers"][str(l)]["heads"])
    return out

if __name__ == "__main__":
    from scipy.stats import spearmanr
    import importlib.util
    spec = importlib.util.spec_from_file_location("r0", "edits/01_r0_compress.py"); r0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r0)
    st = pickle.load(open("logs/passes/base_stats.pkl", "rb"))
    m, t, a = M.load("checkpoints/base", "cpu")
    imp = importance(m, a, st, last_cls=True)
    for l in range(6):
        mean_iso, var_iso = r0.ffn_model(m.bert.encoder.layer[l])
        mu_emp, var_emp = mom(st[f"ffn{l}"], l == 5)
        u = np.sort(imp[l]["u_ffn"])[::-1]; c = np.cumsum(u) / u.sum()
        print(f"L{l}: isotropic-model vs empirical  rho(mean)={spearmanr(mean_iso, mu_emp)[0]:.2f} rho(var)={spearmanr(var_iso, var_emp)[0]:.2f} | "
              f"FFN neurons for 90/99% of u: {int(np.searchsorted(c,.9))+1}/{int(np.searchsorted(c,.99))+1} | "
              f"sum u_ffn={imp[l]['u_ffn'].sum():.3f}  heads u: {np.round(imp[l]['u_head'],3)}")
