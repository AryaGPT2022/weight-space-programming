"""Phase C diagnosis from already-paid-for logits (slate + pass 1). No new passes."""
import sys, pickle, re; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from slate import slate
tok = AutoTokenizer.from_pretrained("checkpoints/base")  # nosec B615 - local checkpoint
STOP = set("the a an of to in and or for is are what how does do on with by from at as be was which who when why can that this it".split())

def feats(q, p):
    qt = [w for w in tok.tokenize(q.lower()) if w not in STOP and w.isalnum()]
    pt = tok.tokenize(p.lower()); ps = set(pt)
    return dict(overlap=np.mean([w in ps for w in qt]) if qt else 0.0, plen=len(pt), qlen=len(qt))

def frame(model="base"):
    pairs = slate(); s = pickle.load(open("logs/passes/slate.pkl", "rb"))[model]
    df = pairs.assign(logit=s)
    if model == "base":
        p1 = pickle.load(open("logs/passes/p01.pkl", "rb"))
        df = pd.concat([df, p1["pairs"].assign(logit=p1["logits"])]).drop_duplicates(["qid", "docid"])
    df = df.reset_index(drop=True)
    df = pd.concat([df, pd.DataFrame([feats(q, p) for q, p in zip(df["query"], df.text)])], axis=1)
    df["y"] = df.label > 0
    return df

if __name__ == "__main__":
    df = frame(); df.to_pickle("logs/passes/diag_df.pkl")
    print("within-dataset spearman(logit, feature | label):")
    for d, g in df.groupby("dataset"):
        out = [f"{c}|y={int(y)}:{spearmanr(h.logit, h[c])[0]:+.2f}" for c in ("overlap", "plen") for y in (0, 1)
               for h in [g[g.y == y]] if len(h) > 8]
        out.append(f"label~plen:{spearmanr(g.y, g.plen)[0]:+.2f} label~ov:{spearmanr(g.y, g.overlap)[0]:+.2f}")
        print(f"  {d:9s} n={len(g):3d}  " + "  ".join(out))
    tr = (df.plen + df.qlen + 3) > 256
    print("truncated share:", df.assign(tr=tr).groupby("dataset").tr.mean().round(2).to_dict())
    for _, r in df[~df.y].sort_values("logit", ascending=False).head(6).iterrows():
        print(f"FP {r.dataset[:6]} {r.logit:+.2f} ov={r.overlap:.2f} len={r.plen:4d} Q:{r['query'][:55]!r} P:{r.text[:95]!r}")
    for _, r in df[df.y].sort_values("logit").head(6).iterrows():
        print(f"FN {r.dataset[:6]} {r.logit:+.2f} ov={r.overlap:.2f} len={r.plen:4d} Q:{r['query'][:55]!r} P:{r.text[:95]!r}")
