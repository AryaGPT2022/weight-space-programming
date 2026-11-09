"""Pass 8 (base): is there a lead/position bias? 51 short probe positives x 5 variants:
orig | 80 filler tokens PREPENDED | 80 APPENDED | 160 PREPENDED | 160 APPENDED.
Filler = sentences from other queries' negative passages (probe data, free to read).
Appending is the dilution control: same extra text, relevant content stays at the front."""
import sys, pickle; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from transformers import AutoTokenizer
from wsp.harness import ForwardBudget, probe_pool
tok = AutoTokenizer.from_pretrained("checkpoints/base")  # nosec B615 - local checkpoint
pool = probe_pool("cls")
pool["plen"] = [len(tok.tokenize(t)) for t in pool.text]
cand = pool[(pool.label > 0) & (pool.plen.between(30, 85))]
pos = pd.concat([g.sample(min(len(g), 13), random_state=8) for _, g in cand.groupby("dataset")]).head(51).reset_index(drop=True)
negs = pool[(pool.label == 0) & (~pool.qid.isin(pos.qid))].sample(400, random_state=9)
rng = np.random.default_rng(0)
def filler(n, qid):
    out = []
    while len(out) < n:
        out += tok.tokenize(negs.text.iloc[rng.integers(len(negs))])
    return tok.convert_tokens_to_string(out[:n])
rows = []
for i, r in pos.iterrows():
    f80, f160 = filler(80, r.qid), filler(160, r.qid)
    for v, txt in [("orig", r.text), ("pre80", f80 + " " + r.text), ("post80", r.text + " " + f80),
                   ("pre160", f160 + " " + r.text), ("post160", r.text + " " + f160)]:
        rows.append(dict(qid=r.qid, query=r["query"], text=txt, variant=v, item=i, dataset=r.dataset))
pairs = pd.DataFrame(rows)
fb = ForwardBudget("main", 25)
out = fb.probe("checkpoints/base", pairs, note="lead-bias test: positives shifted by prepended vs appended filler")
pairs["logit"] = out["logits"]
pickle.dump(pairs, open("logs/passes/p08_leadbias.pkl", "wb"))
w = pairs.pivot(index="item", columns="variant", values="logit")
for v in ["pre80", "post80", "pre160", "post160"]:
    d = w[v] - w["orig"]
    print(f"{v:8s} Δlogit vs orig: mean {d.mean():+.2f}  median {d.median():+.2f}  share dropped >2: {(d < -2).mean():.2f}")
print("shift effect beyond dilution (pre - post): 80:", round((w.pre80 - w.post80).mean(), 2), " 160:", round((w.pre160 - w.post160).mean(), 2))
print("by dataset (pre160 - post160):", (w.pre160 - w.post160).groupby(pos.dataset).mean().round(2).to_dict())
print("passes used:", fb.used)
