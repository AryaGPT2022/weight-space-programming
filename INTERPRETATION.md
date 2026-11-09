# INTERPRETATION.md — Weight-Space Programming a RAG Reranker

A guided, code-level reading of this repo, written as a starting point for research. Everything below comes from reading every source file in `wsp/`, `edits/`, `analysis/`, the logs in `logs/`, the per-edit `edit_meta.json` files, and the numbers in `results/*.json`. Where I am inferring rather than reading, I say so.

> **Status notes on the repo itself**
> - `README.md` points to `WRITEUP.md`. That file does not exist yet.
> - `results/summary.md` and `results/pareto.png` are produced by `analysis/report.py`. They have not been generated yet (`results/` holds only JSON).
> - The ledger shows **17 of 25** forward passes used. Passes 18–25 were deliberately left unused (see §6, Stage 12).
> - "Claude (Opus 5.5) is the weight programmer" is a claim made by `README.md`, `RULES.md` and the edit docstrings. The repo contains the *artifacts* of that work (scripts, logs, ledger), not a transcript of the reasoning. §7 reconstructs the reasoning from those artifacts.

---

## 0. TL;DR

| | |
|---|---|
| **Idea** | *Weight-space programming*: an LLM agent edits a trained network's weights directly. It uses no gradients, no training, no optimizer, and only a tiny budget of forward passes (25). |
| **Model edited** | `cross-encoder/ms-marco-MiniLM-L-6-v2`: 6 layers, 12 heads, hidden 384, FFN 1536, 22.7M parameters. It scores a (query, passage) pair with one relevance logit. |
| **Task** | Reranking / relevance classification for RAG: "is this retrieved passage relevant enough to put in the context?" |
| **Benchmark** | MS MARCO v1.1 validation (in-domain) plus BEIR SciFact, NFCorpus, FiQA (out-of-domain, BM25 top-50). Queries are split 20% *probe* (editable-against) / 80% *test* (locked). |
| **Best result** | `r25_compress_revertlate`: **−3.54M parameters (−15.6% total, −33% encoder+head)**, 1.44× faster on CPU. MS MARCO nDCG@10 **−1.85 pts**; BEIR nDCG@10 **+0.70**; F1 **+1.73**. At matched size, the data-free baselines lose 7–14 pts on MS MARCO and 9–18 pts on BEIR (relative to base). |
| **Honest caveat** | The BEIR gain is probably within noise, and most of the F1 gain is a calibration shift (BEIR AUC is slightly *down*). The robust claims are (a) compression is far less lossy than data-free baselines, and (b) the positional edit helps on longer-passage corpora. See §9–§10. |

---

## 1. Mental model

A **cross-encoder** concatenates `[CLS] query [SEP] passage [SEP]`, runs it through a BERT stack, and reads one scalar from the final `[CLS]` vector:

```
logit = w_cls · tanh( W_pool · LN_final(x_CLS) + b_pool )          (b_cls is exactly 0)
```

A **weight program** is a deterministic script `edits/NN_*.py` that loads a checkpoint, applies surgery (removing neurons, heads or layers, projecting out directions, rewriting rows of a table) and saves a new checkpoint. The "programmer" may *read* any tensor and do data-free linear algebra. It may *run* the network only through `ForwardBudget.probe`, 256 pairs at a time, 25 times in total, with every call written to `logs/ledger.jsonl`. `RULES.md` lists what is forbidden: gradients, optimizers, fitting to labels, sweeping hyperparameters against the probe metric, touching test.

Two families of edit appear in this repo:

| Family | Goal | Edits |
|---|---|---|
| **Compression** | Remove parameters, keep quality | 01 (zero passes), 02 (uses activation statistics) |
| **Quality** | Fix a diagnosed behaviour ("lead bias": the score falls when the relevant text sits later in the passage) | 03 (key-space projection), 04 (position-table edit) |
| **Stacked** | Both | 05 |

---

## 2. The model that is loaded

Defined in `wsp/model.py:34` (`BASE_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"`).

| Component | Shape | Params |
|---|---|---|
| Word embeddings | 30,522 × 384 | ~11.7M |
| Position embeddings | 512 × 384 | 0.2M (only rows 0–255 are ever used: `MAX_LEN=256`) |
| Embeddings total | | **11.92M (52% of the model)** |
| Per layer ×6: Q, K, V, O (12 heads × 32 dims) | 384→384 each | ~0.59M |
| Per layer ×6: FFN | 384→1536→384 | ~1.18M |
| Encoder + head (pooler + classifier) | | **10.80M** |
| **Total** | | **22.71M** |

Two consequences matter for everything that follows:

1. **Every edit in this repo touches only the 10.8M encoder+head parameters.** The 11.9M embedding table is untouched (the vocab-shrink op exists in `surgery.py` but is never used in a final candidate). After compression the model is **62% embedding**.
2. **The final layer is special.** Only the `[CLS]` row of layer 5 reaches the classifier, so layer 5's work at every other position is wasted. This single fact drives the biggest savings.

### How loading works (`wsp/model.py`)

- `load(path)` returns `(model, tokenizer, arch)`.
  - `path=None` loads the pristine HF model.
  - A directory that has `wsp_arch.json` is rebuilt structurally via `build_from_arch`, then the saved tensors are loaded with `strict=False` and checked for missing and unexpected keys.
- `wsp_arch.json` records which original layers, heads and FFN neurons survive. It exists because stock `transformers` cannot represent per-layer head counts and FFN sizes. This repo shrinks the Q/K/V/O/FFN `Linear`s physically.
- Custom modules:
  - `LowRankLinear`: SVD factorisation (defined, unused in finals).
  - `NullSelfAttention`: a layer with all heads removed.
  - `RemappedEmbedding`: shrunken vocab table (unused in finals).
  - `_cls_only_forward`: the **exact** rewrite of the last layer to compute only the `[CLS]` row. Keys and values still come from all tokens. It is verified to differ by at most 1e-6 in logit and is ~16% faster (prediction **P2**, confirmed).
- `score()` runs `(query, passage)` pairs, truncates the passage only (`truncation="only_second"`, max 256 tokens), and returns logits. Optional outputs are attentions and hidden states.
- `device()` prefers Apple MPS, else CPU.

---

## 3. The task, and how it is benchmarked

### 3.1 Task

RAG context selection: given a query and a retrieved passage, decide how relevant it is. Two views of the same data:

| View | Question | Metric |
|---|---|---|
| `rerank` | Order the candidate list for a query | nDCG@10, MRR@10 |
| `cls` | Is this one retrieved passage relevant? | ROC-AUC, F1 at one global threshold |

### 3.2 Datasets (`wsp/data.py`)

| Dataset | Role | Candidates | Test queries used |
|---|---|---|---|
| **MS MARCO v1.1 validation** | In-domain. The model was fine-tuned on MS MARCO *train*, so validation avoids overlap. | ~10 Bing passages per query, label = `is_selected`, 1,000 queries sampled with seed 0 | 800 |
| **BEIR SciFact** | Out-of-domain (scientific claims) | BM25 top-50 (`bm25s`, English stemmer, stopwords) | 218 |
| **BEIR NFCorpus** | Out-of-domain (medical) | BM25 top-50 | 192 |
| **BEIR FiQA** | Out-of-domain (financial QA) | BM25 top-50 | 352 |

Design decisions worth knowing (all in `data.py` and `logs/benchmark_notes.md`):

- **Probe/test split.** `_split_qids` shuffles query ids with seed 0: 20% *probe*, 80% *test*. The programmer may touch only probe, and only through the budgeted harness. The harness raises `PermissionError` for any pair whose `qid` is outside the probe set.
- **`cls` set is a RAG decision, not "all positives vs. a few negatives".** BEIR `cls` = judged positives among *retrieved* candidates (capped at 5/query) + the 5 hardest unjudged candidates + 5 random deeper unjudged candidates. Version 1 used qrels positives that BM25 never retrieved, and treated unjudged hits as negatives. Base AUC came out at 0.34–0.74 because many "negatives" were relevant but unjudged. This was redesigned **once, before any editing**, and only the unedited base model was run during that redesign.
- **nDCG is candidate-normalised.** The ideal DCG uses the judged docs *in the list*. Queries with no relevant candidate are skipped. This makes scores comparable between models but not to published BEIR numbers (which use the full corpus qrels).
- **F1 uses ONE global threshold per model**, chosen on pooled *probe* `cls` pairs (`evaluate.best_threshold`, 197 quantile candidates), then applied to every test set. So F1 mixes discrimination with cross-domain calibration.

### 3.3 Measurement (`wsp/evaluate.py`)

Per model it records:

- nDCG@10, MRR@10, AUC and F1 per dataset.
- A summary: `msmarco_ndcg@10`, `msmarco_auc`, `beir_ndcg@10` (mean of 3 datasets, **unweighted by query count**), `beir_auc`, and `f1_global_thr` (mean over all 4).
- Parameter counts, safetensors MB, and pairs/s on CPU (4 threads) and MPS (batch 32 × 256 tokens, random token ids).
- A `test_eval` row in the ledger. The audit later checks that only declared finals appear there.

---

## 4. File-by-file walkthrough

For each file: **what it reads**, **which model it touches**, **what it does**.

### 4.1 Core library — `wsp/`

| File | Reads | Model | Role |
|---|---|---|---|
| `wsp/data.py` | HF datasets (`microsoft/ms_marco`, `BeIR/*`) | none | Builds the four `rerank.parquet` and `cls.parquet` files in `data/`, with probe/test split. Run via `python -m wsp.data` (~5 min). |
| `wsp/model.py` | `checkpoints/*/model.safetensors`, `wsp_arch.json` | base and all edited | Load/save with structural architecture, custom modules, the exact CLS-only last layer, `score`, `param_counts`, `throughput`. |
| `wsp/harness.py` | probe parquet (via `_probe_qids`), the checkpoint passed in | any checkpoint | **The budget gate.** `ForwardBudget.probe` enforces ≤256 pairs, probe-only qids, and the pass count. It logs to the ledger and persists every pass to `logs/passes/raw/` *before* analysis (added after pass 9 was wasted by an analysis crash). It can also install **statistics hooks** (`_install_stat_hooks`) that accumulate, per layer, over non-pad tokens and separately at `[CLS]`: per-neuron FFN activation sum/sum-of-squares, per-head contribution vectors `ctx_h · W_O,h`, and LayerNorm-input energy. |
| `wsp/inspect.py` | weights only (**0 forward passes**) | any | Read-only views: per-head QK and OV circuits (`head_mats`, `head_table`), effective rank, FFN in/out norm table, `readout_direction` (γ ⊙ W_poolᵀ w_cls, the linearised relevance direction), embedding tables. |
| `wsp/surgery.py` | weights; never data | base or edited | The edit ops. `prune_heads`, `prune_heads_folded` (folds the removed head's constant value-bias into the output bias), `prune_ffn(fold_const=...)` (folds `act · w_out` of removed neurons into the output bias), `drop_layers`, `low_rank`, `project_out`, `shrink_vocab`, and `diff_report` (percent of weights changed or removed). |
| `wsp/evaluate.py` | `data/*/*.parquet` test split, probe split for the threshold | checkpoint under test | Locked test evaluation, §3.3. Appends to the ledger. |
| `wsp/audit.py` | ledger, `finals.json`, `edits/*.py`, optionally re-runs edits | none | Four checks: passes ≤ budget, test evaluations only for declared finals, regex scan of `edits/` for `backward / optim / autograd / .grad / Trainer`, and (with `--rebuild`) SHA-256 reproducibility. **Limit:** the scan covers `edits/` only, not `analysis/`. |

### 4.2 The weight programs — `edits/`

| File | Passes it depends on | Parent | One-line summary |
|---|---|---|---|
| `baselines.py` | 0 | base | Non-agent comparison points: drop top-2 layers (4,5), drop layers (1,3), magnitude-prune 33% of heads and FFN uniformly, plus size-matched −50% variants (drop 1,3,5; magnitude 50%). |
| `01_r0_compress.py` | **0** | base | Compression written from weights alone, using a *modelled* activation distribution. §7.1. |
| `02_r10_compress.py` | 3 (stats) | base | Compression using *measured* activation statistics (`logs/passes/base_stats.pkl`). Two pre-declared sizes: 7.25M and 5.40M encoder+head. §7.2. |
| `03_posfix.py` | 10–11 (hidden states) | base | Removes the positional component from `[CLS]`-reading heads' keys. Two iterations: `c_posfix`, `c_posfix2`. §7.3. |
| `04_unlate.py` | 0 data; reads a second checkpoint | base | Edits the position-embedding table. `c_unlate` (rank-1) and `c_revertlate` (stronger). §7.4. |
| `05_combined.py` | 0 | `r10_compress`, `r10_compress_xs` | Applies revert-late on top of compression, giving `r25_compress_revertlate` and `r25_compress_xs_revertlate`. The two edits touch disjoint tensors. |

### 4.3 Probe-pass scripts and analysis — `analysis/`

These are the programmer's "lab notebook". They are the only code that uses forward passes, and they all go through `ForwardBudget`.

| File | Pass | What it does |
|---|---|---|
| `pass01_diagnose.py` | 1 | Runs 64 probe `cls` pairs on base (attentions, hidden states, stats). Checks pre-registered predictions P1 (logit vs readout direction), P3 (positional heads' attention offsets), P4 (token-match heads), P5 (L4H9 as a sink). |
| `pass02_03_stats.py` | 2–3 | 2×256 `rerank` pairs with `stats` hooks. Merges with pass 1 → `base_stats.pkl` (~113k tokens, 576 `[CLS]` rows). This is the input to edit 02. |
| `importance.py` | none (offline) | Turns the stats into **mean-ablation importance** `u` for every FFN neuron and head. §7.2. Also prints how poorly the zero-pass isotropic activation model matches reality. |
| `slate.py` | 4–7, 16–17 | A fixed 256-pair probe slate (64 per dataset, 32 pos + 32 neg). One pass per model. Reports AUC and Spearman fidelity to base logits. |
| `diagnose.py` | none | Post-hoc analysis of slate logits: token overlap and passage length vs logit, top false positives and negatives. This is where the "length / position" suspicion came from. |
| `pass08_leadbias.py` | 8 | The key diagnostic experiment. 51 short positives × 5 variants (original, 80/160 filler tokens **prepended**, 80/160 **appended**). Appending is the dilution control: the same extra text with the relevant content still at the front. |
| `pass09_mechanism.py` | 9 (wasted), re-run as 10 | Per head: query→identical-token attention mass ("match signal") and `[CLS]`→relevant-content mass, original vs shifted. Also `[CLS]` cosine drift per layer. |
| `pass11_cls_prior.py` + `cls_prior_analysis.py` | 11 | `[CLS]` attention density on the relevant span for orig / pre80 / pre160 / post160. `shift_loss = post160 − pre160` measures a pure position effect at equal dilution. |
| `pos_subspace.py` | offline on passes 10–11 | Estimates the positional subspace of the residual stream entering layers 3–5, and each head's `[CLS]` query subspace. Feeds edit 03. |
| `posblind_design.py` | 0 | A data-free design for making token-match heads position-blind. The chosen-k criterion was never met, so the idea was **abandoned without spending a pass**. |
| `leadbias_check.py` | 12–15 | Re-runs pass-8's pairs on an edited checkpoint (1 pass) to measure the position effect after an edit. |
| `report.py` | none | Aggregates `results/*.json` → `summary.md` and a Pareto plot. |
| `run_finals.sh` | none | Evaluates the declared finals on the locked test split. |

### 4.4 Records

- `logs/00_reading.md`: Phase-A findings and the pre-registered predictions P1–P7, written **before** pass 1.
- `logs/01_passes.md`: the pass table and prediction scorecard.
- `logs/benchmark_notes.md`: why the benchmark was redesigned once.
- `logs/ledger.jsonl`: every pass and every test evaluation.
- `logs/finals.json`: the 14 declared final candidates.
- `logs/passes/`: persisted tensors and pickles.
- `checkpoints/*/edit_meta.json`: which tensors changed, the percent touched, the heads removed and the neurons removed.

---

## 5. Where the "reading" happens

"Reading" is the zero-cost phase: no network is run. It happens in three places.

1. **Phase A (`logs/00_reading.md`, using `wsp/inspect.py`).** Done before any pass. Findings:
   - The score is **nearly one-dimensional**. The readout matrix `R = diag(w_cls) · W_pool · diag(γ_final)` has singular values 0.070, 0.010, 0.010, …, so one direction holds ~50% of the readout energy and 64 hold 90%.
   - **Most last-layer FFN neurons are invisible to the score.** The median |cos(w_out, r̂)| is 0.002 (random ≈ 0.04); the 99th percentile is 0.12.
   - **Head taxonomy of layers 0–2**, from QK circuits applied to the real embedding tables:
     - *Positional heads* attend to a fixed offset: L0H2 (−1), L0H5 (−1), L1H4 (+2), L1H8 (+1), L1H9 (−1), L1H11 (−1), L2H0 (−2), L2H8 (+1), L2H11 (−1).
     - *Token-match heads* attend to the identical token on the other side. Same-token z-scores: L1H1 4.4, L0H8 3.9, L2H9 3.7, L1H0 2.4, L2H4 1.9. This is the lexical-matching circuit.
     - *Anti-match heads*: L0H7, L1H2, L1H5, L2H7.
   - Odd heads: L0H2 has the largest QK (Frobenius 32). L4H9 has a large low-rank QK and the smallest OV in layer 4. It was predicted to be a sink; that prediction (P5) **failed** (0.17 mass on `[CLS]/[SEP]`), though it is still the least important head by empirical variance.
   - Parameters: embeddings are 11.9M of 22.7M. Low-rank factorisation barely helps at width 384 (matrices need rank 150–270 for 90% energy). A data-free FFN activation model says layers 0–4 need ~1,000 of 1,536 neurons for 80% of modelled output energy, so **FFN importance is diffuse in the middle layers**.
2. **In `edits/01_r0_compress.py`.** It reads LayerNorm γ/β, FFN in/out weights and per-head `W_V W_O` to score components analytically.
3. **In `edits/04_unlate.py`.** It reads the **position-embedding table** of the fine-tuned model and of the *pretrained* MiniLM (`nreimers/MiniLM-L6-H384-uncased`). The comparison reveals what MS-MARCO fine-tuning did to each position row. Note the second checkpoint is extra information beyond the single deployed model, a research-design choice to scrutinise (§10).

---

## 6. The experiment end to end, stage by stage

| Stage | What happens | Code | Passes | Outcome |
|---|---|---|---|---|
| **0. Setup** | `uv sync`; save the base model as `checkpoints/base` | `wsp/model.py` | 0 | Reproducible environment |
| **1. Benchmark** | Build four datasets with probe/test split. Redesign the BEIR `cls` set once, before editing. | `wsp/data.py` | 0 | `data/<name>/{rerank,cls}.parquet` |
| **2. Baselines** | Five data-free, non-agent compressions. An agent edit only counts if it beats these at matched parameter count. | `edits/baselines.py` | 0 | `bl_*` checkpoints. Evaluated on test (these are also declared finals). |
| **3. Phase A reading** | Read the weights; write predictions P1–P7 *before* running anything. | `wsp/inspect.py`, `logs/00_reading.md` | 0 | Head taxonomy, readout direction, "L5 is mostly wasted" |
| **4. R0 compression** | A compression program from weights alone, with an analytic activation model and the CLS-only last layer. | `edits/01_r0_compress.py` | 0 | `r0_compress`: MSM 0.716 / BEIR 0.614, beating all baselines (P6 ✅) |
| **5. Check predictions** | Run base on 64 pairs with attention and hidden-state capture. | `analysis/pass01_diagnose.py` | 1 | P1 ρ=0.988 ✅; P3 1 of 6 ⚠️; P4 ✅; P5 ❌ |
| **6. Measure activations** | Two 256-pair passes with stats hooks → empirical FFN and head statistics. Finding: the isotropic model was nearly useless (ρ ≈ 0.0–0.4). At `[CLS]`, **4 layer-5 FFN neurons carry 90% of the readout variance and 17 carry 99%.** | `pass02_03_stats.py`, `importance.py` | 2–3 | `base_stats.pkl` |
| **7. R10 compression** | Rebuild the compression using empirical importance, and protect the circuits Phase A found. | `edits/02_r10_compress.py` | (uses 1–3) | `r10_compress` (7.25M), `r10_compress_xs` (5.40M) |
| **8. Fidelity check** | Score a fixed slate with base, R0, R10, R10-xs. | `analysis/slate.py` | 4–7 | Spearman fidelity to base: R0 0.927, R10 0.988, R10-xs 0.975 |
| **9. Diagnose a behavioural flaw** | After looking at errors (`diagnose.py`), test for a **lead bias**: does the score fall when the relevant text is pushed later? | `pass08_leadbias.py` | 8 | Moving a positive 160 tokens later costs **−8.5 logits**; appending the same filler costs −2.0. So the **pure position effect is −6.5 logits**. |
| **10. Find the mechanism** | Where does the shift hurt? | `pass09_mechanism.py` (pass 9 wasted by a crash → harness now persists first), pass 10; `pass11_cls_prior.py` | 9–11 | Token-match heads lose query→match attention when content moves. `[CLS]` states diverge at layers 4–5. L5H11, L4H8, L5H0, L4H3, L4H6 lose about half their `[CLS]`→content attention. |
| **11a. Edit 03 (key projection)** | Remove positional components from those heads' keys. | `edits/03_posfix.py` + `pos_subspace.py` | 12–13 | Position effect −6.54 → −5.70 (−13%) → −5.20 (−20%) |
| **11b. Edit 04 (position table)** | Compare the fine-tuned position table with the pretrained one. Rows 128–256 were *rotated* in fine-tuning (cosine 0.63–0.67 vs 0.95 for early rows). Subtract that direction (rank 1), or revert it. | `edits/04_unlate.py` | 14–15 | rank-1: −24%; revert-late: **−37%** (−6.54 → −4.10). Unshifted positives are unchanged. |
| **12. Combine and sanity-check** | Stack revert-late on the compressed models. Check for harm on the slate. | `edits/05_combined.py`, `slate.py` | 16–17 | Slate mean AUC 0.760 (c_revertlate) / 0.755 (r25) vs 0.766 base: neutral within noise (SE ≈ 0.03). **Passes 18–25 deliberately not used**: the slate is too small to separate further variants, and sweeping more would amount to selecting by score. |
| **13. Declare finals, test once** | Evaluate the 14 declared finals on the locked test split. | `analysis/run_finals.sh`, `wsp/evaluate.py` | 0 | `results/*.json` |
| **14. Audit** | Passes ≤ budget, test touched only by finals, no gradients in `edits/`, bit-reproducible edits. | `wsp/audit.py` | 0 | `AUDIT PASSED` (re-checked in this session without `--rebuild`) |

> Not yet evaluated on test (declared finals with no result file): `c_posfix`, `c_unlate`, `c_revertlate`, `r25_compress_xs_revertlate`. `c_posfix2` is not in `finals.json`, so it was never meant to be tested.

---

## 7. Explainability: how the edited weights were chosen

The question is *why these heads, these neurons, these directions*. Each edit follows the same pattern: **(1) find a signal in the weights or in recorded statistics → (2) define a scalar importance or direction → (3) pick by rule (not by probe score) → (4) repair what the removal breaks → (5) validate.**

### 7.1 Edit 01 — zero-pass compression (`r0_compress`)

*Programmer knowledge:* weights, LayerNorm scales, the readout structure.

| Step | What was removed | Selection rule | Repair |
|---|---|---|---|
| Layer 5 FFN | **1152 of 1536 neurons** (keeps 384) | Rank by **readout-visible energy** `e_i = Var[gelu(pre_i)] · ‖R w_out_i‖²`, with `R = diag(w_cls)·W_pool·diag(γ_final)`. Pre-activation is modelled as `N(w_in·β + b, ‖w_in ⊙ γ‖²)`, data-free. | Fold the modelled mean `E[gelu]·w_out_i` into the output bias. |
| Layer 5 heads | **4 of 12** (H3, H5, H8, H11) | Lowest `‖R·diag(γ_att)·(W_V W_O)ᵀ‖_F`, i.e. OV circuits that write least onto the readout. | Fold the removed head's constant value-bias into W_O's bias. |
| Layers 0–4 FFN | 691 / 521 / 668 / 642 / 679 neurons (≈ 3,200 total) | Global ranking by layer-normalised modelled energy `Var[gelu] · ‖w_out ⊙ γ_out‖²` until the encoder+head target (7,250,689, matched to the magnitude baseline) is met. | Fold modelled means. |
| Layers 0–4 heads | **None removed.** | Phase A said these hold the positional and token-match circuits. | — |
| Last layer compute | CLS-only forward | Exact algebra. | — |

**Why this is explainable:** every removed unit has a number attached that says "how much of what this writes can the classifier see". **How it did:** MS MARCO 0.716, BEIR 0.614, versus the best baseline at 0.674 / 0.577. The weak spot is that the isotropic activation model correlates only ρ≈0.0–0.4 with measured statistics (pass 2–3). It worked mostly because the *readout* structure did the heavy lifting.

### 7.2 Edit 02 — measured-statistics compression (`r10_compress`, `_xs`)

*Programmer knowledge:* 576 `[CLS]` rows and ~113k tokens of real activations (passes 1–3).

**The importance score** (`analysis/importance.py`), per unit, is the *variance* of what the unit writes to the residual stream, seen through the next LayerNorm's γ, as a fraction of that LayerNorm's input energy:

```
u_ffn  = Var(a_i) · ‖w_out_i ⊙ γ_out‖² / E[d·σ²]            (layers 0-4)
u_head = Σ_d Var(contribution_h,d) · γ_att,d² / E[d·σ²]
```

Final layer: the same, projected through the readout `R` and evaluated at `[CLS]` only.

This is **mean-ablation** reasoning: a unit whose output barely varies can be replaced by its *mean* (folded into a bias) at almost no cost. Variance, not magnitude, is the correct signal.

| Where | Removed | Why |
|---|---|---|
| Layer 5 FFN | **1472 of 1536** (keeps the 64 with the largest readout-projected `[CLS]` variance) | 4 neurons carry 90% and 17 carry 99% of the variance. |
| Layer 5 heads | **5 of 12** (H2, H3, H5, H8, H11) | Each has <3% of the readout-projected variance (`L5_HEAD_MIN_SHARE=0.03`). |
| Layers 0–4 | Global per-parameter ranking of `u / param_cost` across FFN neurons and heads, until the target is hit. For `r10_compress` this removed **only FFN neurons** (566 / 272 / 470 / 493 / 1016). For the −50% `_xs` variant it also removed heads L3H11, L3H6, L4H9, L4H1, L4H4, L4H6, with more neurons. | The cost-normalised ranking means cheap-per-parameter neurons go first. |
| **Protected** | L0H8, L1H1, L2H9, L1H0, L2H4 (token-match) and L0H2, L1H8 (confirmed positional) | From Phase A and pass 1. They can never be removed, whatever their variance. |

Parameter arithmetic for `r10_compress`: 4,289 neurons × 769 + 5 heads × 49,248 ≈ 3.54M removed. That equals the 22.71M → 19.17M drop.

### 7.3 Edit 03 — positional projection in keys (`c_posfix`, `c_posfix2`)

Evidence chain: pass 8 (position costs −6.5 logits) → pass 10/11 (which heads lose attention when content moves) → hidden states from passes 10–11.

For each target head `(l, h)`:

```
W_K ← W_K − U (Uᵀ W_K) V Vᵀ
  U : positional subspace of the residual entering layer l   (PCs of per-position means above a shuffled-position null)
  V : span of the head's [CLS] query vectors                  (95%, later 99%, of energy)
```

In words: delete the part of the key that is **both** positional **and** visible to `[CLS]` queries. Everything else in QK is unchanged, so rank ≤ 24 per head. Targets: L5H11, L4H8, L5H0, L4H3, L4H6, L5H6 (v1), plus L3H0, L3H3, L4H7, L4H9, L3H6 (v2), all with shift loss ≥ 0.15. Result: position effect −13% (v1), −20% (v2), at a small cost on unshifted positives (−0.3 logits for v2). Too weak, so the agent looked for a more global cause (§7.4). The v2 target list was chosen by the pass-11 shift-loss number, which is the one place this resembles tuning against a probe measurement.

### 7.4 Edit 04 — the "lateness" direction in the position table (`c_unlate`, `c_revertlate`)

The key finding: heads' positional priors explained only ~15–20% of the lead bias, so something upstream was involved. The programmer compared the fine-tuned position table `P_ft` to the pretrained one `P_pre` (`D_i = P_ft[i] − P_pre[i]`):

- Rows 128–256 were *rotated* by MS-MARCO fine-tuning (cosine 0.63–0.67 against 0.95 for early rows) with **norm unchanged**, so it is not weight decay.
- Those updates share one direction `v` that is ~0 for positions < 96 and > 256 and peaks at 160–192.
- Interpretation: MS MARCO passages are short, so fine-tuning learned a "this is late, discount it" signal that is harmful when passages are long (BEIR).

Two programs, both data-free:

```
c_unlate     c = mean_{2..95} D_i ;  v = normalise(mean_{128..255}(D_i − c))
             P_new[i] = P_ft[i] − ((D_i − c)·v) v                      rank-1, one tensor

c_revertlate P_new[i] = (1−w_i) P_ft[i] + w_i (P_pre[i] + c)          w ramps 0→1 over rows 96–128, 1 until 256
             keep only the update shared with early positions; drop late-specific learning
```

Row indices ≥256 are never used (`MAX_LEN=256`), so the discontinuity at 256 is harmless. **Touched weights: 0.27% of the model, one tensor.** Position effect: −6.54 → −4.98 (rank-1) → **−4.10 (revert)**.

### 7.5 What the choices tell you (a compact explanation)

| Decision | Primary signal | Type of evidence |
|---|---|---|
| Keep only `[CLS]` in the last layer | Architecture: only row 0 feeds the classifier | Exact (verified to 1e-6) |
| Remove almost all L5 FFN | Readout `R` is low-rank; 4 neurons carry 90% of `[CLS]` variance | Weights, then measured |
| Remove 5 L5 heads | They carry <3% of readout-projected variance | Measured |
| Protect token-match and positional heads in layers 0–2 | QK-circuit analysis, same-token z-scores | Weights, confirmed by pass 1 (P4) |
| Remove middle-layer FFN neurons by variance | Mean-ablation logic | Measured |
| Edit the position table | Fine-tuned vs pretrained row rotation | Weights (second checkpoint) |

---

## 8. Results

Locked test split. Δ is relative to `base`. Speeds are on this machine.

| Model | Enc+head | Total | MB | MS MARCO nDCG@10 | MS MARCO AUC | BEIR nDCG@10 | BEIR AUC | F1 | CPU pairs/s |
|---|---|---|---|---|---|---|---|---|---|
| **base** | 10.80M | 22.71M | 86.7 | 0.7430 | 0.7393 | 0.6712 | 0.8086 | 0.3787 | 161 |
| drop layers 1,3 (`bl_drop_alt2`) | 7.25M | 19.16M | 73.1 | 0.6703 | 0.6594 | 0.5765 | 0.7522 | 0.3279 | 233 |
| drop top-2 layers (`bl_drop_top2`) | 7.25M | 19.16M | 73.1 | 0.6001 | 0.5949 | 0.4951 | 0.7015 | 0.3169 | 256 |
| magnitude −33% (`bl_mag_uniform33`) | 7.25M | 19.17M | 73.1 | 0.6741 | 0.6690 | 0.5182 | 0.7301 | 0.3073 | 239 |
| **R0** `r0_compress` (0 passes) | 7.25M | 19.17M | 73.1 | 0.7158 | 0.7144 | 0.6144 | 0.7788 | 0.3477 | 235 |
| **R10** `r10_compress` | 7.25M | 19.17M | 73.1 | 0.7212 (−2.2) | 0.7183 | 0.6681 (−0.3) | 0.8028 | 0.3902 (+1.2) | 233 |
| **R25** `r25_compress_revertlate` | 7.25M | 19.17M | 73.1 | **0.7245 (−1.9)** | 0.7184 (−2.1) | **0.6781 (+0.7)** | 0.8061 (−0.3) | **0.3960 (+1.7)** | 232 |
| `r10_compress_xs` (−50%) | 5.40M | 17.32M | 66.1 | 0.7063 | 0.7009 | 0.6536 | 0.7957 | 0.3637 | 285 |
| `bl_drop_alt3` (−50%) | 5.47M | 17.39M | 66.3 | 0.5630 | 0.5620 | 0.2158 | 0.5789 | 0.2563 | 347 |
| `bl_mag_uniform50` (−50%) | 5.48M | 17.40M | 66.4 | 0.6082 | 0.6103 | 0.3673 | 0.6598 | 0.2594 | 296 |

Per-dataset, R25 vs base (nDCG@10 / F1): SciFact 0.760 vs 0.742 / 0.479 vs 0.45, NFCorpus 0.662 vs 0.655 / 0.398 vs 0.31, FiQA 0.612 vs 0.616 / 0.346 vs 0.385. The F1 threshold moved from 2.90 (base) to 2.33 (R25).

**Not evaluated yet** (no result files): `c_posfix`, `c_unlate`, `c_revertlate`, `r25_compress_xs_revertlate`.

**Matched-size comparison.** At 7.25M, R25 beats the best data-free baseline by **+5.4 MS MARCO nDCG** and **+10.2 BEIR nDCG**. At 5.4M, R10-xs beats the best baseline by **+9.8** and **+28.6** (vs. `bl_mag_uniform50` on BEIR). This is the headline result, and it is a large effect.

**Ablation inside the table.** R25 − R10 isolates revert-late on a compressed model: **+0.3 MS MARCO, +1.0 BEIR nDCG, +0.3 BEIR AUC, +0.6 F1**. R10 − base isolates compression: **−2.2 MS MARCO nDCG, −0.3 BEIR nDCG, −2.1 MS MARCO AUC, −0.6 BEIR AUC, +1.2 F1**.

---

## 9. How to gauge "−3.5M parameters for −1.9 MS MARCO nDCG, but +0.7 BEIR and +1.7 F1"

First, restate the trade accurately: it is **3.54M fewer parameters (−15.6% total, −33% of encoder+head), −15.7% disk, ~1.44× CPU / 1.36× MPS throughput**. Not "−3M". And 62% of what remains is the embedding table, which was not touched.

Judge it on five axes, in this order.

### 9.1 Is each delta real? (statistical noise)

Back-of-envelope, assuming per-query nDCG has a standard deviation around 0.3 (typical; **I did not measure it here**):

| Delta | Queries | Unpaired SE ≈ | Reading |
|---|---|---|---|
| MS MARCO nDCG −1.85 pts | 800 | ~1.1 pts | Paired (same queries) is far tighter, so a loss of this size is likely real, but could be small in absolute terms. |
| BEIR nDCG +0.70 pts | 762 pooled (218/192/352) | ~1.1 pts | **Probably within noise.** The gain is driven by SciFact (+1.8, 218 queries); FiQA is −0.4. |
| F1 +1.73 pts | | | See 9.2 |

**Do this:** save per-query nDCG in `evaluate.py` (it currently keeps only the means) and run a **paired bootstrap** over queries for each delta. Report a confidence interval, not a point change. Until then, call the BEIR nDCG change "no measurable loss" rather than "improvement".

### 9.2 Discrimination vs calibration

- **AUC** is threshold-free and measures ranking quality: MS MARCO **−2.1**, BEIR **−0.25**.
- **F1** at a global threshold also measures where the score distribution sits. The NFCorpus F1 jumped 0.31 → 0.398 while its AUC moved +0.2 (0.775 → 0.777). That is a **calibration effect**: the edited model's scores line up better with a single threshold across domains.

So "F1 +1.7" is real for a deployment that uses one fixed threshold, but it is **not** evidence that the model ranks better. If you re-tuned the threshold per dataset, the F1 gain would shrink or vanish. Report both.

### 9.3 The right comparison is the matched-size baseline, not the base model

Against `base`, any compression looks like a loss. Against what else you could have done at 7.25M parameters (drop layers, magnitude pruning), this is a **−1.9 vs −6.9 to −14.3** MS MARCO result. The relevant question for research is "how much better than a naive method", and the answer here is large.

### 9.4 Cost side: what you get

| Gain | Size |
|---|---|
| Parameters | −3.54M |
| Disk | −13.6 MB (86.7 → 73.1) |
| CPU throughput | +44% (161 → 232 pairs/s) |
| MPS throughput | +36% |
| FLOPs | The last layer's non-CLS compute is removed entirely; ~16% from CLS-only alone |

If your reranker is latency-bound over many candidates per query, +44% throughput for −1.9 nDCG on in-domain data may be a good trade. If it is quality-bound on in-domain data, it may not.

### 9.5 Where the quality is lost matters more than how much

MS MARCO has short passages. The *pure compression* drops 2.2 points there, and revert-late buys back 0.3. BEIR passages are longer, which is where revert-late helps (+1.0). So the decision depends on your **deployment distribution**:

| Your corpus looks like | Use |
|---|---|
| Short web passages (MS MARCO-like) | `r10_compress`: revert-late adds little, and you pay the 2.2 points |
| Longer, domain-specific passages (BEIR-like) | `r25_compress_revertlate` |
| You need tiny | `r10_compress_xs` (−50% encoder, MS MARCO 0.706, BEIR 0.654) still beats every baseline by a wide margin |

A practical stopping rule: accept the model if (a) the paired-bootstrap CI for MS MARCO nDCG lies within your tolerance (say ≥ −2 pts), (b) BEIR nDCG's CI includes 0 or is positive, (c) AUC loss on your own domain is within tolerance, and (d) the throughput gain meets your latency target.

---

## 10. Critical review: threats to validity and loose ends

1. **No error bars.** No confidence intervals anywhere. §9.1 is the first thing to fix.
2. **Edits 03 v2 and 04 revert were designed after seeing earlier lead-bias numbers.** `01_passes.md` states this openly. The measured quantity is a synthetic position effect on 51 positives, *not* test nDCG, which reduces the leakage risk. But it is closer to "iterating against the probe metric" than `RULES.md` ideally allows. All variants were declared finals, so nothing was cherry-picked.
3. **Revert-late uses a second checkpoint** (`nreimers/MiniLM-L6-H384-uncased`). That is outside information beyond the single deployed model. It is legitimate under "read any tensor", but a stricter version of the game would not allow it.
4. **The audit covers `edits/` only.** `analysis/` contains no gradients today, but the scanner would not catch it. The scan is also a regex, not a proof.
5. **Weak predictions are in the log.** P3 (1 of 6), P5 (failed), P7 (untested). The scorecard is honest, which is a plus. The zero-pass isotropic activation model correlated only ρ≈0–0.4 with measured statistics.
6. **BEIR "average" is unweighted over datasets** and nDCG is candidate-normalised, so it is not comparable with published BEIR numbers.
7. **F1 depends on a threshold chosen on the probe split**, a small set. Part of the F1 gain is calibration (§9.2).
8. **Pass 9 was wasted by an analysis crash.** The harness now persists raw outputs first. 8 of 25 passes remained unused.
9. **Speed numbers use random token ids on one machine** (4 CPU threads / MPS). Directionally right, not a benchmark.
10. **The quality edits have not been evaluated alone on test.** `c_unlate`, `c_revertlate` and `c_posfix` have no result files, so "revert-late helps BEIR" rests on the R25 − R10 comparison only.
11. **Vocab shrink is untapped.** The reading log estimates ~10% of the embeddings (about 1.2M) could go via script-based pruning. That is the cheapest next gain.

---

## 11. Kick-starting your own research

### 11.1 Reproduce (from the README; the data and checkpoints already exist here)

```bash
uv sync
uv run python -m wsp.data                                   # ~5 min, only if data/ is missing
uv run python -c "from wsp import model as M; m,t,a=M.load(None); M.save(m,t,a,M.CKPT/'base')"
uv run python edits/baselines.py
uv run python edits/01_r0_compress.py                        # 0 passes
uv run python edits/02_r10_compress.py                       # needs logs/passes/base_stats.pkl
uv run python edits/03_posfix.py                             # needs logs/passes/pos_subspaces*.pkl
uv run python edits/04_unlate.py
uv run python edits/05_combined.py
uv run python -m wsp.evaluate --model checkpoints/r25_compress_revertlate --name r25_compress_revertlate
uv run python -m wsp.audit --rebuild
uv run python analysis/report.py                             # writes results/summary.md and pareto.png
```

### 11.2 Questions that are cheap to ask next

1. **Error bars:** per-query nDCG saved + paired bootstrap (§9.1).
2. **Evaluate the missing finals:** `c_revertlate`, `c_unlate`, `c_posfix`, `r25_compress_xs_revertlate`, to separate the quality edit from compression.
3. **Add vocab shrink** on top of R25 (~1.2M more parameters, no encoder risk). It is a good data-free edit for the zero-pass track.
4. **Quantisation is orthogonal.** int8 on the 19.2M model should stack with everything here, but needs a loader that understands `wsp_arch.json` shapes.
5. **Ablate the protected set.** P7 was never tested: remove the three token-match heads vs. three positional heads at equal cost.
6. **Replace the position-table reference.** Can the "lateness" direction be found without the pretrained checkpoint (e.g. from the table's own spectrum)?
7. **Per-dataset thresholds.** Re-score F1 with a per-dataset threshold to measure how much of the F1 gain is calibration.
8. **Pareto sweep with a fixed rule.** Run the same programme at 20/33/40/50/60% and plot, with sizes pre-declared (as the repo already does for 7.25M and 5.4M) to avoid probe-metric selection.

### 11.3 Conceptual takeaways

- **Structure beats magnitude.** Variance-based importance plus *protecting known circuits* is why this beats magnitude pruning by 5–10+ points at equal size.
- **Read the architecture first.** The largest single win (almost all of layer 5's FFN) came from noticing only `[CLS]` is read, not from any statistic.
- **Bias folding keeps a pruned unit's mean effect.** Removing a unit's *mean* contribution would shift every downstream LayerNorm; folding it into the bias preserves that shift.
- **Diagnose a behaviour, then edit the minimum.** The best quality edit touches 0.27% of weights, in one tensor.
- **The harness is part of the science.** A hard pass budget, a probe/test lock, a ledger, an audit and reproducible SHA-256 hashes are what make "an LLM edited the weights" a checkable claim.
