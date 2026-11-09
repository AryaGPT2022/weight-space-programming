"""Evaluation of a cross-encoder checkpoint on the locked TEST split.

Quality
  rerank : nDCG@10 and MRR@10 over each query's candidate list (BM25 top-50 for BEIR, the
           ~10 Bing passages for MS MARCO). nDCG is candidate-normalised (IDCG from judged docs in
           the list); queries with no relevant candidate are skipped (every model scores 0).
  cls    : ROC-AUC per dataset, plus F1 at ONE global threshold chosen on the pooled probe-split
           cls pairs (a deployment-style threshold; also tests cross-domain calibration).
Size/speed
  params (total / embedding / encoder+head), safetensors MB, pairs/s on CPU and MPS
  (batch 32 x 256 tokens).

Every test-split evaluation is appended to logs/ledger.jsonl (kind=test_eval) so the audit can
prove the test split was only touched for final candidates.

    uv run python -m wsp.evaluate --model checkpoints/base --name base
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from wsp import model as M
from wsp.data import DATA, DATASETS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LEDGER = ROOT / "logs" / "ledger.jsonl"


def load_split(name, task, split):
    df = pd.read_parquet(DATA / name / f"{task}.parquet")
    return df[df.split == split].reset_index(drop=True)


def score_df(model, tok, df):
    # length-sort for efficient padding, then restore order
    order = np.argsort(df.text.str.len().values)
    s = M.score(model, tok, df["query"].values[order], df["text"].values[order])
    out = np.empty_like(s)
    out[order] = s
    return out


def ndcg_mrr(df, scores, k=10, per_query=None):
    """per_query: optional list; (qid, ndcg, rr) rows are appended to it."""
    df = df.assign(s=scores)
    nd, rr = [], []
    for qid, g in df.groupby("qid"):
        if (g.label > 0).sum() == 0:
            continue
        g = g.sort_values("s", ascending=False)
        gains = (2.0 ** g.label.values - 1)[:k]
        disc = 1 / np.log2(np.arange(2, len(gains) + 2))
        ideal = np.sort(2.0 ** g.label.values - 1)[::-1][:k]
        nd.append((gains * disc).sum() / (ideal * disc[: len(ideal)]).sum())
        hit = np.where(g.label.values[:k] > 0)[0]
        rr.append(1 / (hit[0] + 1) if len(hit) else 0.0)
        if per_query is not None:
            per_query.append((qid, nd[-1], rr[-1]))
    return float(np.mean(nd)), float(np.mean(rr)), len(nd)


def best_threshold(scores, labels):
    cand = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(labels, scores >= t) for t in cand]
    return float(cand[int(np.argmax(f1s))])


def evaluate(path, name, datasets=DATASETS, speed=True, bm25_ref=False):
    model, tok, arch = M.load(path)
    res = {"name": name, "model": str(path), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # threshold from probe-split cls pairs (pooled across datasets)
    probe = pd.concat([load_split(d, "cls", "probe") for d in datasets])
    thr = best_threshold(score_df(model, tok, probe), (probe.label > 0).values)
    res["threshold"] = thr

    per, pq_all = {}, []
    for d in datasets:
        rr = load_split(d, "rerank", "test")
        cl = load_split(d, "cls", "test")
        s_rr, s_cl = score_df(model, tok, rr), score_df(model, tok, cl)
        pq = []
        nd, mrr, nq = ndcg_mrr(rr, s_rr, per_query=pq)
        pq_all.extend((d, q, n, r) for q, n, r in pq)
        y = (cl.label > 0).values
        per[d] = {"ndcg@10": nd, "mrr@10": mrr, "n_queries": nq,
                  "auc": float(roc_auc_score(y, s_cl)), "f1": float(f1_score(y, s_cl >= thr))}
        if bm25_ref:
            per[d]["bm25_ndcg@10"] = ndcg_mrr(rr, -rr.bm25_rank.values.astype(float))[0]
    res["per_dataset"] = per
    beir = [d for d in datasets if d != "msmarco"]
    res["summary"] = {
        "msmarco_ndcg@10": per.get("msmarco", {}).get("ndcg@10"),
        "msmarco_auc": per.get("msmarco", {}).get("auc"),
        "beir_ndcg@10": float(np.mean([per[d]["ndcg@10"] for d in beir])) if beir else None,
        "beir_auc": float(np.mean([per[d]["auc"] for d in beir])) if beir else None,
        "f1_global_thr": float(np.mean([per[d]["f1"] for d in datasets])),
    }
    res["size"] = M.param_counts(model)
    st = Path(path) / "model.safetensors"
    res["size"]["safetensors_mb"] = st.stat().st_size / 2**20 if st.exists() else None
    if speed:
        torch.set_num_threads(4)
        res["speed"] = {"cpu_pairs_per_s": M.throughput(model, tok, n_pairs=256, dev="cpu"),
                        "mps_pairs_per_s": M.throughput(model, tok, n_pairs=1024, dev="mps")
                        if torch.backends.mps.is_available() else None}
    LEDGER.parent.mkdir(exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"kind": "test_eval", "name": name, "model": str(path),
                            "time": res["time"]}) + "\n")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=2))
    (RESULTS / "perquery").mkdir(exist_ok=True)
    pd.DataFrame(pq_all, columns=["dataset", "qid", "ndcg", "rr"]).to_csv(
        RESULTS / "perquery" / f"{name}.csv", index=False)
    return res


def fmt(res):
    s, z = res["summary"], res["size"]
    sp = res.get("speed") or {}
    return (f"{res['name']:28s} MSM nDCG {s['msmarco_ndcg@10']:.4f} AUC {s['msmarco_auc']:.4f} | "
            f"BEIR nDCG {s['beir_ndcg@10']:.4f} AUC {s['beir_auc']:.4f} | F1 {s['f1_global_thr']:.4f} | "
            f"enc {z['params_encoder_head']/1e6:5.2f}M tot {z['params_total']/1e6:5.2f}M | "
            f"cpu {sp.get('cpu_pairs_per_s', 0):6.1f}/s mps {sp.get('mps_pairs_per_s') or 0:6.1f}/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--bm25", action="store_true")
    a = ap.parse_args()
    r = evaluate(a.model, a.name, speed=not a.no_speed, bm25_ref=a.bm25)
    print(fmt(r))
    print(json.dumps(r["per_dataset"], indent=1))
