"""Data-free design of the position-blind key edit: pick k (number of positional PCs removed
from match heads' W_K input) = smallest k with >=70% of mean distance-decay removed and >=90% of
mean same-token match strength kept. Weights + embedding tables only."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from wsp import model as M, inspect as I
MATCH = [(0, 8), (1, 0), (1, 1), (2, 4), (2, 9), (3, 5), (4, 5)]
DIST = [20, 40, 80, 120, 160, 200, 240]

def setup():
    m, t, a = M.load(M.CKPT / "base", "cpu")
    E = I.embedding_table(m); el = m.bert.embeddings.LayerNorm
    g, b = I._np(el.weight), I._np(el.bias)
    ln = lambda x: g * (x - x.mean(-1, keepdims=True)) / np.sqrt(((x - x.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-12) + b
    v = t.get_vocab(); ids = [i for w, i in v.items() if w.isalpha() and w.isascii() and len(w) > 3][:3000]
    W = E["word"][np.random.default_rng(0).choice(ids, 300, replace=False)]
    Pc = (E["pos"][:256] - E["pos"][:256].mean(0)) * g
    U = np.linalg.svd(Pc, full_matrices=False)[2].T          # (384, 256) positional directions in LN-output space
    return m, E, ln, W, U

def profile(mm, E, ln, W, Wk):
    xq = ln(W + E["pos"][5] + E["type"][0]) @ mm["Wq"] + mm["bq"]
    same = []
    for d in DIST:
        xk = ln(W + E["pos"][d] + E["type"][1]) @ Wk + mm["bk"]
        same.append(np.mean(np.sum(xq * xk, 1)) / np.sqrt(32))
    xk_other = ln(np.roll(W, 1, 0) + E["pos"][40] + E["type"][1]) @ Wk + mm["bk"]
    diff = np.mean(np.sum(xq * xk_other, 1)) / np.sqrt(32)
    same = np.array(same)
    return same[1] - diff, np.abs(same - same[1]).mean()   # match strength at d=40, decay magnitude

def choose_k(verbose=True):
    m, E, ln, W, U = setup()
    base = {}
    for (l, h) in MATCH:
        mm = I.head_mats(m.bert.encoder.layer[l])[h]; base[(l, h)] = profile(mm, E, ln, W, mm["Wk"])
    for k in (2, 4, 8, 12, 16, 24, 32, 48):
        Pi = np.eye(384) - U[:, :k] @ U[:, :k].T
        keep, dec = [], []
        for (l, h) in MATCH:
            mm = I.head_mats(m.bert.encoder.layer[l])[h]
            s, d = profile(mm, E, ln, W, Pi @ mm["Wk"])
            keep.append(s / base[(l, h)][0]); dec.append(d / base[(l, h)][1])
        ok = np.mean(dec) <= 0.30 and np.mean(keep) >= 0.90
        if verbose:
            print(f"k={k:2d}: match strength kept {np.mean(keep):.2f}   decay remaining {np.mean(dec):.2f}  {'<- chosen' if ok else ''}")
        if ok:
            return k, U[:, :k]
    return None, None

if __name__ == "__main__":
    choose_k()

def drift_dirs(mm, E, ln, W):
    """Directions (in the head's 32-d key space) along which a token's key moves with position."""
    xs = [ln(W + E["pos"][d] + E["type"][1]) @ mm["Wk"] for d in range(20, 256, 4)]
    D = np.stack([x.mean(0) for x in xs]); D = D - D.mean(0)
    return np.linalg.svd(D, full_matrices=False)[2]          # rows = drift directions, by energy

def choose_k_out(verbose=True):
    m, E, ln, W, U = setup()
    res = {}
    for k in (1, 2, 3, 4, 6, 8):
        keep, dec = [], []
        for (l, h) in MATCH:
            mm = I.head_mats(m.bert.encoder.layer[l])[h]
            s0, d0 = profile(mm, E, ln, W, mm["Wk"])
            V = drift_dirs(mm, E, ln, W)[:k].T                    # (32, k)
            Wk = mm["Wk"] @ (np.eye(32) - V @ V.T)
            mm2 = dict(mm, bk=mm["bk"] @ (np.eye(32) - V @ V.T))
            s, d = profile(mm2, E, ln, W, Wk)
            keep.append(s / s0); dec.append(d / d0)
        ok = np.mean(dec) <= 0.30 and np.mean(keep) >= 0.90
        if verbose:
            print(f"k_out={k}: match strength kept {np.mean(keep):.2f}   decay remaining {np.mean(dec):.2f}  {'<- chosen' if ok else ''}")
        if ok:
            return k
    return None

if __name__ == "__main__":
    print("--- head-space (output side) projection")
    choose_k_out()
