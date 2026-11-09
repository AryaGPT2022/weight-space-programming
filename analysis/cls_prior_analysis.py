import sys, pickle; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
d = pickle.load(open("logs/passes/raw/main_11.pkl", "rb"))
pairs = d["pairs"].reset_index(drop=True)
ex = []
for A, H, I in zip(d["attentions"], d["hidden"], d["input_ids"]):
    for b in range(I.shape[0]):
        n = int((I[b] != 0).sum()); ex.append(dict(ids=I[b, :n], att=[a[b, :, 0, :n] for a in A]))  # CLS row only
# relevant span per example: orig/post -> starts right after first SEP; pre -> ends before final SEP
def rel_span(k):
    r = pairs.loc[k]; I = ex[k]["ids"]; s = np.where(I == 102)[0][0]; n = len(I)
    o = pairs.index[(pairs["item"] == r["item"]) & (pairs.variant == "orig")][0]
    L = len(ex[o]["ids"]) - s - 2
    return np.arange(s + 1, s + 1 + L) if r.variant in ("orig", "post160") else np.arange(n - 1 - L, n - 1)
rows = []
for k in range(len(pairs)):
    rel = rel_span(k); n = len(ex[k]["ids"])
    for l in range(6):
        for h in range(12):
            a = ex[k]["att"][l][h]
            rows.append(dict(item=pairs.loc[k, "item"], v=pairs.loc[k, "variant"], l=l, h=h,
                             mass=a[rel].sum(), first_pos=a[[0]].sum() + a[np.where(ex[k]["ids"] == 102)[0]].sum(),
                             com=(a * np.arange(n)).sum() / max(a.sum(), 1e-9)))
R = pd.DataFrame(rows)
T = R.pivot_table(index=["l", "h"], columns="v", values="mass")
T["shift_loss"] = T["post160"] - T["pre160"]   # same tokens, same dilution; only position differs
print("CLS->relevant-content mass, by head (post160 vs pre160 = position effect at equal dilution):")
print(T.sort_values("shift_loss", ascending=False).head(12).round(3).to_string())
S = R[R.v == "orig"].groupby(["l", "h"]).first_pos.mean()
print("\nmean CLS mass on [CLS]/[SEP] (sink) in L3-L5:", S.loc[3:5].round(2).values.reshape(3, 12))
pickle.dump(R, open("logs/passes/p11_cls_prior.pkl", "wb"))
