"""Estimate, from recorded hidden states (passes 10-11; no new forward computation beyond applying
single weight matrices to recorded activations):
  * the positional subspace of the residual stream entering layers 3-5 (per-position means, centered,
    PCA; dimension = PCs above a position-shuffled null),
  * the [CLS] query subspace of each target head (x_CLS · W_Q + b_Q, PCs for 95% energy)."""
import sys, pickle; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
CRIT = 'paired' if '--paired' in sys.argv else 'max'
QE = 0.99 if '--paired' in sys.argv else 0.95
from wsp import model as M, inspect as I

def load_hidden():
    H = {l: [] for l in range(7)}
    for f in ("logs/passes/raw/main_10.pkl", "logs/passes/raw/main_11.pkl"):
        d = pickle.load(open(f, "rb"))
        for Hb, Ib in zip(d["hidden"], d["input_ids"]):
            for b in range(Ib.shape[0]):
                n = int((Ib[b] != 0).sum())
                for l in range(7):
                    H[l].append(Hb[l][b, :n])
        del d
    return H

def positional_subspace(seqs, max_pos=250, seed=0, min_pos=30):
    D = seqs[0].shape[1]
    S, C = np.zeros((max_pos, D)), np.zeros(max_pos)
    for x in seqs:
        n = min(len(x), max_pos); S[:n] += x[:n]; C[:n] += 1
    ok = C >= 20
    ok[:min_pos] = False   # passage-only positions: avoid confounding position with the query segment
    Mu = S[ok] / C[ok, None]; Mu -= Mu.mean(0)
    s = np.linalg.svd(Mu, compute_uv=False)
    # null: shuffle positions within each sequence
    rng = np.random.default_rng(seed); S0, C0 = np.zeros((max_pos, D)), np.zeros(max_pos)
    for x in seqs:
        n = min(len(x), max_pos); p = rng.permutation(len(x))[:n]; S0[:n] += x[p]; C0[:n] += 1
    Mu0 = S0[ok] / C0[ok, None]; Mu0 -= Mu0.mean(0)
    s0 = np.linalg.svd(Mu0, compute_uv=False)
    k = int(np.argmin(s[:len(s0)] > s0)) if CRIT == 'paired' else int((s > s0[0]).sum())
    U = np.linalg.svd(Mu, full_matrices=False)[2][:k].T
    return U, k, s[:k + 3], s0[:6]

if __name__ == "__main__":
    H = load_hidden()
    m, t, a = M.load(M.CKPT / "base", "cpu")
    out = {}
    for l in (3, 4, 5):
        U, k, s, s0 = positional_subspace(H[l])
        print(f"residual entering L{l}: positional dims above null = {k}  (sv {np.round(s[:6],2)}..., null top {np.round(s0,2)})")
        X = np.stack([x[0] for x in H[l]])                       # CLS rows entering layer l
        for h in range(12):
            mm = I.head_mats(m.bert.encoder.layer[l])[h]
            Q = X @ mm["Wq"] + mm["bq"]
            Vq = np.linalg.svd(Q, full_matrices=False)[2]
            e = np.cumsum(np.linalg.svd(Q, compute_uv=False) ** 2); e /= e[-1]
            r = int(np.searchsorted(e, QE)) + 1
            out[(l, h)] = dict(U_pos=U, Vq=Vq[:r].T, r=r)
        print(f"   CLS-query rank (95% energy) per head: {[out[(l,h)]['r'] for h in range(12)]}")
    pickle.dump(out, open("logs/passes/pos_subspaces%s.pkl" % ("_v2" if CRIT == "paired" else ""), "wb"))
