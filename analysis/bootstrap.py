"""Paired bootstrap over queries for nDCG@10 differences (needs results/perquery/<name>.csv,
written by wsp.evaluate).

    uv run python analysis/bootstrap.py base c_revertlate c_unlate r25_compress_revertlate

Each candidate is compared with the FIRST name given. Resampling is by query, within each dataset;
the BEIR number is the unweighted mean over scifact/nfcorpus/fiqa (same definition as evaluate.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B, SEED = 10_000, 0
BEIR = ["scifact", "nfcorpus", "fiqa"]


def load(name):
    return pd.read_csv(ROOT / "results" / "perquery" / f"{name}.csv")


def paired(base, cand):
    m = base.merge(cand, on=["dataset", "qid"], suffixes=("_b", "_c"))
    return {d: (g.ndcg_c - g.ndcg_b).values for d, g in m.groupby("dataset")}


def boot(diffs, datasets, rng):
    """Bootstrap the mean over `datasets` of the per-dataset mean paired difference."""
    out = np.zeros(B)
    for d in datasets:
        x = diffs[d]
        idx = rng.integers(0, len(x), size=(B, len(x)))
        out += x[idx].mean(1)
    return out / len(datasets)


def main(names):
    rng = np.random.default_rng(SEED)
    base = load(names[0])
    print(f"paired bootstrap vs {names[0]} (B={B}); values are nDCG@10 points x100")
    print(f"{'candidate':28s} {'MS MARCO':>26s} {'BEIR avg':>26s}")
    for n in names[1:]:
        diffs = paired(base, load(n))
        cells = []
        for ds in (["msmarco"], BEIR):
            obs = np.mean([diffs[d].mean() for d in ds])
            lo, hi = np.percentile(boot(diffs, ds, rng), [2.5, 97.5])
            flag = "*" if lo > 0 or hi < 0 else " "
            cells.append(f"{obs*100:+6.2f} [{lo*100:+5.2f},{hi*100:+5.2f}]{flag}")
        print(f"{n:28s} {cells[0]:>26s} {cells[1]:>26s}")
    print("* = 95% interval excludes 0")


if __name__ == "__main__":
    main(sys.argv[1:])
