# Benchmark construction notes (before Phase A; only the unedited base model was run)

**v1 of the BEIR relevance-classification set**
- Positives: all qrels positives, including documents BM25 never retrieves.
- Negatives: the top-5 BM25 non-judged candidates.
- Base AUC on test: NFCorpus 0.34, FiQA 0.47, SciFact 0.74.
- A probe-split diagnosis found two problems:
  - The "negatives" are mostly unjudged documents that are actually relevant. BEIR's sparse judgments cause this.
  - Grade-1 NFCorpus positives are weak.
- Pooled AUC over the real candidate lists was 0.80–0.94. That points to a dataset artifact, not a model artifact.

**v2 (in use)**
- Built from the *retrieved* candidates only: judged positives (capped at 5 per query), plus the 5 hardest unjudged candidates, plus 5 random deeper unjudged candidates.
- This matches the decision a RAG system actually makes: "is this retrieved passage relevant enough to put in context?"
- MS MARCO is unchanged.
- The ledger was reset after the v1 base evaluation.
