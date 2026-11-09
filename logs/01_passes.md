# Forward-pass log, run `main` (budget 25; 17 used)

Every pass is in `logs/ledger.jsonl`. From pass 10 on, the raw outputs are also in `logs/passes/raw/`.

| # | model | pairs | purpose | outcome |
|---|---|---|---|---|
| 1 | base | 64 (attn, hidden, stats) | Check Phase-A predictions P1, P3, P4, P5 | See the scorecard below |
| 2–3 | base | 2×256 (stats) | Empirical FFN and head activation statistics (113k tokens, 576 [CLS]) | The zero-pass isotropic activation model was nearly useless (ρ≈0.0–0.4). At [CLS], 4 final-layer FFN neurons carry 90% of the readout variance. |
| 4–7 | base, r0, r10, r10-xs | 4×256 | Score each model on a fixed probe slate | Fidelity ρ = 0.927 (R0), 0.988 (R10), 0.975 (R10-xs) |
| 8 | base | 255 synthetic | Lead-bias test: shift a positive with prepended vs appended filler | Shifting by 160 tokens costs −8.5 logits; appending costs −2.0. Position effect is **−6.5**. |
| 9 | base | 48 (attn, hidden) | Mechanism | **Wasted.** The analysis crashed (`r.item` is a pandas method) before the outputs were saved. The harness now persists every pass. |
| 10 | base | 48 (attn, hidden) | Pass 9 rerun | Token-match heads lose query→match attention when the content moves. [CLS] diverges at layers 4–5. |
| 11 | base | 128 (attn, hidden) | [CLS] positional prior, at equal dilution | L5H11, L4H8, L5H0, L4H3 and L4H6 lose about half of their [CLS]→content attention |
| 12 | c_posfix | 255 | Lead-bias test of edit 03 | Position effect −6.54 → −5.70 (−13%) |
| 13 | c_posfix2 | 255 | Edit 03 iteration 2 (larger subspaces, more heads) | −6.54 → −5.20 (−20%). Unshifted positives lose 0.3 logits. |
| 14 | c_unlate | 255 | Edit 04: subtract the rank-1 "lateness" direction from the position table | −6.54 → −4.98 (−24%). Unshifted positives unchanged. |
| 15 | c_revertlate | 255 | Edit 04, stronger variant: revert late-specific learning in rows 96–256 | −6.54 → **−4.10 (−37%)**. Unshifted positives unchanged. |
| 16–17 | c_revertlate, r25_compress_revertlate | 2×256 | Slate check for harm | Mean AUC 0.760 / 0.755, against 0.766 for base. Neutral within noise (SE ≈ 0.03). |

Passes 18–25 were not used. The slate is too small to tell further quality variants apart, and further sweeps would have amounted to selecting by score.

## Phase-A prediction scorecard
| | Prediction | Result |
|---|---|---|
| P1 | The logit tracks r̂·LN(x_CLS) with ρ > 0.9 | ✅ ρ = 0.988 |
| P2 | Computing the last layer for [CLS] only is exact | ✅ max \|Δ\| = 1e-6, 16% faster |
| P3 | Six positional heads put > 80% of their mass on the predicted offset | ⚠️ 1 of 6 (L0H2 0.90; L1H8 0.73; the others 0.18–0.42). The offsets were right but the heads are softer than predicted. |
| P4 | L1H1, L0H8 and L2H9 are the top query→identical-token heads | ✅ They are exactly the top 3 (0.54, 0.34, 0.11). Positional heads score ≈ 0. |
| P5 | L4H9 is an attention sink (> 50% on [CLS]/[SEP]) | ❌ 0.17. It is still the least important head by empirical variance. |
| P6 | The zero-pass compression beats all baselines, with MS MARCO ≥ 0.70 and BEIR ≥ 0.60 | ✅ 0.716 / 0.614 (best baseline: 0.674 / 0.577) |
| P7 | Removing match heads hurts more than removing positional heads | Not tested. No pass was spent on it. |

## Process notes
- **Benchmark.** The relevance-classification set was redesigned once before Phase A. See `benchmark_notes.md`.
- **Activation statistics.** Passes 10–11 recorded hidden states. Single weight matrices were applied to them offline, for example x_CLS·W_Q to get the [CLS] query subspace. No multi-layer recomputation was done outside a pass.
- **Iteration after seeing results.** Edit 03 v2 and edit 04's revert variant were each designed after seeing the previous pass's lead-bias number. That is iterative use of passes, as in the blog's Exp 8. All variants were declared final and evaluated on test, so none was cherry-picked.
- **First criterion choice.** The k chosen for the position-blind keys in the match heads (`analysis/posblind_design.py`) never met the pre-declared criterion. That approach was dropped without spending any pass on it.
