"""Build the relevance benchmark used throughout the experiment.

Datasets
--------
* msmarco  (in-domain)  : microsoft/ms_marco v1.1 *validation* (not train -> no overlap with the
                          cross-encoder's training data). Each query comes with ~10 Bing passages
                          labelled is_selected. 1,000 sampled queries.
* scifact, nfcorpus, fiqa (out-of-domain, BEIR): BM25 (bm25s) top-K candidates per test query.

Two tasks share the same data
-----------------------------
* rerank : every (query, candidate) pair; metric nDCG@10 / MRR@10 per query.
* cls    : relevance classification. All judged positives + the top-5 BM25 hard negatives per query;
           metric AUC + F1 (threshold chosen on the probe split, applied to test).

Splits
------
Queries are split per dataset into disjoint `probe` (20%) and `test` (80%) using a fixed seed.
Only `probe` may be touched during editing (via the budgeted harness). `test` is locked.

Output: data/<name>/rerank.parquet, data/<name>/cls.parquet with columns
    qid, docid, query, text, label (graded int), bm25_rank, split
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 0
PROBE_FRAC = 0.2
TOPK = 50
N_HARD_NEG = 5
DATASETS = ["msmarco", "scifact", "nfcorpus", "fiqa"]
# Pinned Hugging Face dataset revisions (commit SHAs): the benchmark cannot change under us.
REVISIONS = {
    "microsoft/ms_marco": "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a",
    "BeIR/scifact": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
    "BeIR/scifact-qrels": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
    "BeIR/nfcorpus": "b5026a0e96e8a7ac4f95f482a596389289d46269",
    "BeIR/nfcorpus-qrels": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
    "BeIR/fiqa": "979c07a7cb5ccc6ca009792241fa1250b98055dd",
    "BeIR/fiqa-qrels": "252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
}


def _split_qids(qids, seed=SEED):
    qids = sorted(set(qids))
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_probe = int(round(len(qids) * PROBE_FRAC))
    return {q: ("probe" if i < n_probe else "test") for i, q in enumerate(qids)}


def build_msmarco(n_queries=1000):
    from datasets import load_dataset

    ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation", revision=REVISIONS["microsoft/ms_marco"])
    keep = [i for i, r in enumerate(ds["passages"])
            if 1 <= sum(r["is_selected"]) and len(r["is_selected"]) >= 5]
    rng = random.Random(SEED)
    keep = sorted(rng.sample(keep, n_queries))
    rows = []
    for i in keep:
        r = ds[i]
        qid = str(r["query_id"])
        for j, (txt, sel) in enumerate(zip(r["passages"]["passage_text"], r["passages"]["is_selected"])):
            rows.append(dict(qid=qid, docid=f"{qid}_{j}", query=r["query"], text=txt,
                             label=int(sel), bm25_rank=j))  # bm25_rank = original Bing order
    rr = pd.DataFrame(rows)
    split = _split_qids(rr.qid)
    rr["split"] = rr.qid.map(split)
    # classification: all positives + up to N_HARD_NEG negatives (first in Bing order)
    neg = rr[rr.label == 0].groupby("qid").head(N_HARD_NEG)
    cls = pd.concat([rr[rr.label > 0], neg]).sort_values(["qid", "bm25_rank"]).reset_index(drop=True)
    return rr, cls


def build_beir(name):
    import bm25s
    import Stemmer
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus", revision=REVISIONS[f"BeIR/{name}"])
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries", revision=REVISIONS[f"BeIR/{name}"])
    qrels = load_dataset(f"BeIR/{name}-qrels", split="test", revision=REVISIONS[f"BeIR/{name}-qrels"]).to_pandas()
    qrels.columns = ["qid", "docid", "score"]
    qrels["qid"] = qrels.qid.astype(str)
    qrels["docid"] = qrels.docid.astype(str)
    qrels = qrels[qrels.score > 0]

    doc_ids = [str(x) for x in corpus["_id"]]
    doc_text = [(t + " " + x).strip() if t else x for t, x in zip(corpus["title"], corpus["text"])]
    id2text = dict(zip(doc_ids, doc_text))
    qtext = {str(q): t for q, t in zip(queries["_id"], queries["text"])}
    test_qids = sorted(set(qrels.qid) & set(qtext))

    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(doc_text, stopwords="en", stemmer=stemmer, show_progress=False),
                    show_progress=False)
    q_tok = bm25s.tokenize([qtext[q] for q in test_qids], stopwords="en", stemmer=stemmer,
                           show_progress=False)
    res, _ = retriever.retrieve(q_tok, k=TOPK, show_progress=False, n_threads=4)

    rel = {(q, d): int(s) for q, d, s in qrels.itertuples(index=False)}
    rows = []
    for qi, q in enumerate(test_qids):
        for rank, di in enumerate(res[qi]):
            d = doc_ids[int(di)]
            rows.append(dict(qid=q, docid=d, query=qtext[q], text=id2text[d],
                             label=rel.get((q, d), 0), bm25_rank=rank))
    rr = pd.DataFrame(rows)
    split = _split_qids(test_qids)
    rr["split"] = rr.qid.map(split)

    # classification = the RAG decision "is this *retrieved* passage relevant?":
    #   judged positives among the retrieved candidates (cap 5/query, highest grade first)
    #   + 5 hardest unjudged candidates (BM25 ranks) + 5 random deeper unjudged candidates.
    # (An earlier version used all qrels positives incl. never-retrieved ones vs only top-5 BM25
    #  negatives; unjudged-but-relevant BM25 hits made that set label-noisy - see logs/benchmark_notes.md)
    pos = (rr[rr.label > 0].sample(frac=1.0, random_state=SEED)
           .sort_values(["qid", "label"], ascending=[True, False], kind="stable")
           .groupby("qid").head(N_HARD_NEG))
    negs = rr[rr.label == 0]
    hard = negs.groupby("qid").head(N_HARD_NEG)
    deep = (negs.drop(hard.index).groupby("qid", group_keys=False)
            .apply(lambda g: g.sample(min(N_HARD_NEG, len(g)), random_state=SEED)))
    cls = pd.concat([pos, hard, deep]).sort_values(["qid", "bm25_rank"]).reset_index(drop=True)
    return rr, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=DATASETS)
    args = ap.parse_args()
    for name in args.only:
        out = DATA / name
        out.mkdir(parents=True, exist_ok=True)
        rr, cls = build_msmarco() if name == "msmarco" else build_beir(name)
        rr.to_parquet(out / "rerank.parquet")
        cls.to_parquet(out / "cls.parquet")
        for tag, df in [("rerank", rr), ("cls", cls)]:
            g = df.groupby("split")
            print(f"{name:9s} {tag:6s} pairs={len(df):6d} queries={df.qid.nunique():4d} "
                  f"pos_rate={np.mean(df.label > 0):.3f} "
                  + " ".join(f"{k}:{v.qid.nunique()}q/{len(v)}p" for k, v in g))


if __name__ == "__main__":
    main()
