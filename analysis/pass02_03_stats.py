"""Passes 2-3 (base): empirical activation statistics (FFN neurons, per-head contributions,
residual energy at each LayerNorm), all tokens and [CLS]-only. 2 x 256 rerank-candidate pairs
(64 per dataset per pass), the distribution the reranker actually sees."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle
from wsp.harness import ForwardBudget, probe_pool, merge_stats
pool = probe_pool("rerank")
fb = ForwardBudget("main", 25)
outs = []
for seed in (2, 3):
    pairs = pool.groupby("dataset", group_keys=False).apply(lambda g: g.sample(64, random_state=seed)).reset_index(drop=True)
    outs.append(fb.probe("checkpoints/base", pairs, want=("stats",), note=f"activation stats on base, seed {seed}"))
p1 = pickle.load(open("logs/passes/p01.pkl", "rb"))
st = merge_stats(p1["stats"], *(o["stats"] for o in outs))
pickle.dump(st, open("logs/passes/base_stats.pkl", "wb"))
print("tokens:", st["ffn0"]["n"], " cls:", st["ffn0"]["n_cls"], " passes used:", fb.used)
