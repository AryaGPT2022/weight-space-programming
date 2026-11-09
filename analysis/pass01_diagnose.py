"""Pass 1 (base model): test pre-registered predictions P1, P3, P4, P5."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle, numpy as np
from wsp.harness import ForwardBudget, probe_pool
from wsp import model as M, inspect as I

pool = probe_pool("cls")
pairs = pool.groupby("dataset", group_keys=False).apply(lambda g: g.sample(16, random_state=1)).reset_index(drop=True)
fb = ForwardBudget("main", 25)
out = fb.probe("checkpoints/base", pairs, want=("attentions", "hidden", "stats"),
               note="P1/P3/P4/P5 checks on base; 16 cls pairs per dataset")
pickle.dump({"pairs": pairs, **out}, open("logs/passes/p01.pkl", "wb"))

m, t, a = M.load("checkpoints/base", "cpu")
L5 = m.bert.encoder.layer[5]
g, b = I._np(L5.output.LayerNorm.weight), I._np(L5.output.LayerNorm.bias)
r = I.readout_direction(m); rh = r / np.linalg.norm(r)
# P1: logit vs r̂·LN(x_CLS) (normalised pre-γ residual)
proj, logits = [], out["logits"]
# hidden[-1] is post-LN output of the final layer; undo γ/β to get x̂, then project
for hb in out["hidden"]:
    xc = hb[-1][:, 0]                    # (B,384) final CLS (after LN)
    xhat = (xc - b) / g
    proj.append(xhat @ (g * rh) / np.linalg.norm(g * rh))
proj = np.concatenate(proj)
order = np.argsort(pairs.text.str.len().values) * 0  # score() keeps original order
print(f"P1  corr(logit, r̂·x̂_CLS) = {np.corrcoef(logits, proj)[0,1]:.3f}   (predicted > 0.9)")

# attentions: list over batches -> list over layers (B,H,T,T); ids per batch
pos_heads = {(0,2):-1,(0,5):-1,(1,9):-1,(1,11):-1,(1,8):+1,(2,8):+1}
match_heads = [(1,1),(0,8),(2,9)]
offs = {k: [] for k in pos_heads}; sink = []; match = {}
for att, ids in zip(out["attentions"], out["input_ids"]):
    B, T = ids.shape
    for bi in range(B):
        n = int((ids[bi] != 0).sum()); seps = np.where(ids[bi][:n] == 102)[0]; q_end = seps[0]
        for (l,h), o in pos_heads.items():
            A = att[l][bi, h, :n, :n]
            idx = np.arange(1, n-1); offs[(l,h)].append(A[idx, idx+o].mean())
        A = att[4][bi, 9, :n, :n]; sink.append(A[:, [0, *seps]].sum(1).mean())
        # P4: from query tokens (1..q_end-1) to passage tokens with the SAME id
        qi = np.arange(1, q_end); pj = np.arange(q_end+1, n-1)
        same = (ids[bi][qi][:, None] == ids[bi][pj][None, :])
        if same.sum() == 0: continue
        for l in range(3):
            for h in range(12):
                A = att[l][bi, h, :n, :n][np.ix_(qi, pj)]
                match.setdefault((l,h), []).append((A * same).sum() / max(1, same.any(1).sum()))
print("P3  mass on predicted offset:", {f"L{l}H{h}({o:+d})": round(float(np.mean(offs[(l,h)])),2) for (l,h),o in pos_heads.items()}, "(predicted > 0.8)")
print(f"P5  L4H9 mass on [CLS]/[SEP]: {np.mean(sink):.2f}  (predicted > 0.5)")
mm = sorted(((np.mean(v), k) for k, v in match.items()), reverse=True)
print("P4  top heads by query->identical-passage-token mass:", [(f"L{k[0]}H{k[1]}", round(float(v),3)) for v,k in mm[:8]])
print("    positional heads' match mass:", [(f"L{l}H{h}", round(float(np.mean(match[(l,h)])),3)) for (l,h) in pos_heads if l<3])
