# Phase A: reading the weights (0 forward passes)

The programmer is Claude (Opus 5.5). Everything below came from weights and vocab strings alone, using `wsp/inspect.py` and the ad-hoc numpy recorded in this log. The only network run so far is the unedited base model, during benchmark construction (see `benchmark_notes.md`) and baseline evaluation. **These predictions were written before pass #1 of run `main`.**

## What the weights say

### 1. The relevance score is close to one-dimensional
`logit = w_cls · tanh(W_pool · LN_final(x_CLS) + b_pool)`, and `b_cls` is exactly 0.
- Scaling W_pool's rows by w_cls and by the final LN's γ gives a "readout matrix" R. Its singular values are 0.070, 0.010, 0.010, ..., so **one direction holds 50% of the readout energy**. 64 directions hold 90%.
- |w_cls| is spread fairly evenly across the 384 pooler units (0.019–0.061), so the pooler itself is not sparse.

### 2. Most last-layer FFN neurons are invisible to the score
- Median |cos(w_out, r̂)| in layer 5 is **0.002**. A random direction would give about 0.04 at this width.
- The 99th percentile is 0.12, so a minority of "writer" neurons carry the direct effect.
- Only the [CLS] row of layer 5 reaches the classifier. Layer 5's queries, residual and FFN at every other position are wasted compute.

### 3. Head taxonomy for layers 0–2 (QK circuits applied to the real embedding and position tables)
- **Positional heads.** These attend to a fixed relative offset with p ≈ 1.0: L0H2 (−1), L0H5 (−1), L1H4 (+2), L1H8 (+1), L1H9 (−1), L1H11 (−1), L2H0 (−2), L2H8 (+1), L2H11 (−1). Near-positional: L1H2 (−2), L1H6 (−2), L2H10 (−4).
- **Token-match heads.** These attend from a query-side token to the *same* token on the passage side. Same-token z-scores: **L1H1 4.4, L0H8 3.9, L2H9 3.7**, L1H0 2.4, L2H4 1.9. The QK trace (`qk_sym`) independently flags the same heads. **This is the cross-encoder's lexical-matching circuitry.**
- **Anti-match heads** attend away from identical tokens: L0H7 (−1.9), L1H2 (−2.4), L1H5 (−2.0), L2H7 (−2.0).

### 4. Odd heads
- L0H2 has by far the largest QK (Frobenius 32, spectral 15) and the largest query-bias·key term (12.2). It's a hard previous-token head driven by position.
- L4H9 has a large, low-rank QK (spectral 9.2, effective rank 25) and the **smallest OV** in layer 4 (2.19). It looks like an attention sink / no-op head.

### 5. Where the parameters are
- Embeddings are 11.9M of 22.7M parameters. Only about 2.9k tokens are non-English scripts or `[unused]`, so script-based vocabulary pruning saves about 10% of the embeddings.
- The embedding spectrum is flat (rank 278 for 90% energy), so factorizing the embedding table isn't attractive without data.
- Encoder matrices need rank 150–270 for 90% energy. Low-rank factorization barely saves anything at 384 width.
- A data-free FFN activation model, with pre-activation ~ N(w·β + b, ‖w⊙γ‖²): layers 0–4 need about 1,000 of 1,536 neurons for 80% of the modeled output energy. **FFN importance is diffuse in the middle layers**, which is why naive pruning hurts.

## Pre-registered predictions (checked with pass #1 and onward)
- **P1.** On real probe pairs, the logit correlates with r̂·LN_final(x_CLS) at **ρ > 0.9**. r̂ is the linearized readout direction.
- **P2.** CLS-only computation in the last layer is exact. *Already verified: max |Δlogit| = 1e-6, 16% faster.*
- **P3.** In real attention maps, L0H2, L0H5, L1H9 and L1H11 put **> 80%** of their mass on token i−1, and L1H8 and L2H8 on i+1.
- **P4.** On pairs where a query term appears in the passage, the token-match heads (L1H1, L0H8, L2H9) put more attention mass from query tokens onto identical passage tokens than any positional head does.
- **P5.** L4H9 sends **> 50%** of its attention to [SEP] or [CLS] (sink behavior).
- **P6.** The zero-pass compression edit (`edits/01_r0_compress.py`) is matched to the baselines' 7.25M encoder+head parameters. It **beats all three data-free baselines** on MS MARCO nDCG@10 and BEIR nDCG@10. Point predictions: MS MARCO ≥ 0.70 and BEIR ≥ 0.60 (base: 0.743 / 0.671; best baseline: 0.674 / 0.577).
- **P7.** Removing the three strongest token-match heads hurts reranking far more than removing three positional heads.
