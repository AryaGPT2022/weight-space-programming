# Rules of the game

These rules are adapted from the constraints in *Sparks of Weight-Space Programming* (Takehi et al.). The **weight programmer** in this repo is Claude (Opus 5.5) running in Claude Code. It plays the part the blog's "Astra" agent plays.

## Forbidden while editing
- Gradients of any kind: `backward`, `torch.autograd`, `requires_grad`-based tricks.
- Optimizers, training loops, and fine-tuning.
- Searching over edit hyperparameters with the probe metric as the objective. For example, you may not sweep ranks or pruning ratios and keep whichever one scores best.
- Fitting any parameter to labels. This includes regression, least squares against labels, and calibration fits.
- Touching the **test** split. It is evaluated only for final candidates, and every evaluation is logged.

## Allowed
- Reading any tensor and running data-free linear algebra on the weights: norms, SVD, eigen-decompositions, cosine similarities, compositions such as W_O·W_V and W_Q·W_Kᵀ, and reading the LayerNorm scales.
- Hand-written edits using `wsp/surgery.py`, or new ops written in the same spirit.
- Reading the probe-split *data* (texts and labels) costs nothing. Only *running the network* costs.
- **Forward passes** through `wsp.harness.ForwardBudget.probe`:
  - A pass is at most **256 pairs**. Those pairs come from the probe split, or are synthetic texts the programmer writes.
  - A pass returns logits. It can also return attentions, hidden states, and input ids.
  - Each run has a fixed budget: **0, 10, or 25** passes. Every pass is written to `logs/ledger.jsonl`.
  - Statistics computed from a pass's outputs may feed an edit. One example is the mean activation of a removed neuron, folded into a bias. This is the same thing the blog's Astra did with its rollouts. Each such use is declared in that edit's log.

## Reproducibility
- Each edit is a deterministic script `edits/NN_name.py`. It takes the base checkpoint, or a declared parent edit, and writes `checkpoints/<name>/` together with `edit_meta.json`. That file holds the diff report: which tensors changed and what percentage of weights were touched.
- `uv run python -m wsp.audit` checks four things:
  - pass counts are within each run's budget;
  - only final candidates have test evaluations;
  - `edits/` contains no gradient or optimizer calls;
  - re-running every edit reproduces its checkpoint's SHA-256.

## Pre-experiment benchmark fixes (before Phase A)
The relevance-classification set was redesigned once, before any editing began. `logs/benchmark_notes.md` describes the change. The only model run during that redesign was the unedited base model.
