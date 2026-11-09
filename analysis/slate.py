"""Score models on a FIXED probe slate (256 cls pairs, 64/dataset). 1 pass per model.
Metrics: AUC per dataset, Spearman fidelity to base logits."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from wsp.harness import ForwardBudget, probe_pool
SLATE = Path("logs/passes/slate.pkl")
def slate():
    pool = probe_pool("cls")
    sl = pool.groupby("dataset", group_keys=False).apply(
        lambda g: g.groupby(g.label > 0, group_keys=False).apply(lambda h: h.sample(min(len(h), 32), random_state=7))
    ).reset_index(drop=True)
    if "dataset" not in sl:
        sl = sl.merge(pool[["qid", "docid", "dataset"]].drop_duplicates(), on=["qid", "docid"], how="left")
    return sl
def run(models):
    fb = ForwardBudget("main", 25)
    pairs = slate()
    cache = pickle.load(open(SLATE, "rb")) if SLATE.exists() else {}
    for m in models:
        if m not in cache:
            cache[m] = fb.probe(f"checkpoints/{m}", pairs, note=f"slate eval of {m}")["logits"]
    pickle.dump(cache, open(SLATE, "wb"))
    y = (pairs.label > 0).values
    for m in cache:
        s = cache[m]
        aucs = {d: roc_auc_score(y[pairs.dataset == d], s[pairs.dataset == d]) for d in pairs.dataset.unique()}
        print(f"{m:22s} fidelity ρ={spearmanr(s, cache['base'])[0]:.3f}  AUC " +
              " ".join(f"{d[:6]}={v:.3f}" for d, v in aucs.items()) + f"  mean={np.mean(list(aucs.values())):.3f}")
    print("passes used:", fb.used, "/ 25")
if __name__ == "__main__":
    run(sys.argv[1:])
