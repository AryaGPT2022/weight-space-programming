"""Pass 9 (base): mechanism of the lead bias. 24 items x {orig, pre160} with attentions + hidden.
Per head: (a) query-token -> identical-passage-token attention mass ('match signal'),
          (b) [CLS] -> relevant-content attention mass. Per layer: CLS-state divergence."""
import sys, pickle; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from wsp.harness import ForwardBudget
P = pickle.load(open("logs/passes/p08_leadbias.pkl", "rb"))
items = sorted(P.item.unique())[:24]
pairs = P[P.item.isin(items) & P.variant.isin(["orig", "pre160"])].drop(columns="logit").reset_index(drop=True)
fb = ForwardBudget("main", 25)
out = fb.probe("checkpoints/base", pairs, want=("attentions", "hidden"), note="lead-bias mechanism: attentions orig vs pre160")
# flatten batches -> per-example
att, hid, ids = [], [], []
for A, H, I in zip(out["attentions"], out["hidden"], out["input_ids"]):
    for b in range(I.shape[0]):
        n = int((I[b] != 0).sum()); ids.append(I[b, :n])
        att.append([a[b, :, :n, :n] for a in A]); hid.append([h[b, :n] for h in H])
res = []
for k, r in pairs.iterrows():
    I = ids[k]; seps = np.where(I == 102)[0]; q = np.arange(1, seps[0]); p_start = seps[0] + 1; n = len(I)
    # relevant content: the original passage tokens = last part for pre160 (before final SEP), first part for orig
    orig_len = len(ids[pairs.index[(pairs["item"] == r["item"]) & (pairs.variant == "orig")][0]]) - seps[0] - 2
    rel = np.arange(p_start, p_start + orig_len) if r.variant == "orig" else np.arange(n - 1 - orig_len, n - 1)
    rel = rel[(rel > seps[0]) & (rel < n - 1)]
    same = I[q][:, None] == I[rel][None, :]
    for l in range(6):
        for h in range(12):
            A = att[k][l][h]
            match = (A[np.ix_(q, rel)] * same).sum() / max(1, same.any(1).sum())
            cls_rel = A[0, rel].sum()
            res.append(dict(item=r["item"], variant=r.variant, layer=l, head=h, match=match, cls_rel=cls_rel))
R = pd.DataFrame(res)
W = R.pivot_table(index=["layer", "head"], columns="variant", values=["match", "cls_rel"])
W["d_match"] = W[("match", "pre160")] - W[("match", "orig")]
W["d_cls"] = W[("cls_rel", "pre160")] - W[("cls_rel", "orig")]
print("heads with largest drop in query->identical-token match mass when shifted:")
print(W.sort_values("d_match").head(8).round(3).to_string())
print("heads with largest change in [CLS]->relevant-content mass:")
print(W.sort_values("d_cls").head(8).round(3).to_string())
# where does CLS diverge?
cos = []
for it in items:
    a = pairs.index[(pairs.item == it) & (pairs.variant == "orig")][0]; b = pairs.index[(pairs.item == it) & (pairs.variant == "pre160")][0]
    cos.append([float(np.dot(hid[a][l][0], hid[b][l][0]) / np.linalg.norm(hid[a][l][0]) / np.linalg.norm(hid[b][l][0])) for l in range(7)])
print("CLS cosine(orig, pre160) after embeddings, L0..L5:", np.round(np.mean(cos, 0), 3))
pickle.dump(dict(R=R, pairs=pairs), open("logs/passes/p09_mech.pkl", "wb"))
print("passes used:", fb.used)
