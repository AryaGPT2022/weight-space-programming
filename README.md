# Weight-space programming a RAG reranker

This repo takes a first stab at [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) on a small BERT cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2`. The weight programmer is Claude (Opus 5.5) working in Claude Code. It may not use gradients or training. It reads the weights, may run the network only through a fixed budget of forward passes (25 here), and writes new weights directly.

For the motivation, method and results, start with **[WRITEUP.md](WRITEUP.md)**.

## Layout
| Path | Contents |
|---|---|
| `RULES.md` | What the programmer may and may not do |
| `wsp/data.py` | Benchmark builder: MS MARCO v1.1 validation plus BEIR SciFact, NFCorpus and FiQA (BM25 top-50), each split into probe and test |
| `wsp/harness.py` | The forward-pass budget, the ledger, and activation-statistics hooks |
| `wsp/inspect.py` | Read-only weight views: QK/OV circuits, FFN tables, readout direction |
| `wsp/surgery.py` | Edit operations: head/FFN pruning with mean folding, layer drop, low-rank, direction projection, vocab shrink, diff report |
| `wsp/model.py` | Saves and loads structurally edited checkpoints (`wsp_arch.json`), plus the CLS-only last layer |
| `wsp/evaluate.py` | Locked test-split evaluation: nDCG@10, MRR@10, AUC, F1, size, speed |
| `wsp/audit.py` | Checks the budget, test-split access, absence of gradients, and edit reproducibility |
| `edits/` | The weight programs, `01`–`05`, plus the data-free baselines |
| `analysis/` | Every probe-pass script and offline analysis |
| `logs/` | Phase-A reading and predictions, the pass log, ledger, raw pass outputs, benchmark notes |
| `results/` | Per-model JSON, `summary.md`, `pareto.png` |

## Reproduce
```bash
uv sync
uv run python -m wsp.data                 # build the benchmark (~5 min)
uv run python -c "from wsp import model as M; m,t,a=M.load(None); M.save(m,t,a,M.CKPT/'base')"
uv run python edits/baselines.py
uv run python edits/01_r0_compress.py     # needs no passes
uv run python edits/02_r10_compress.py    # needs logs/passes/base_stats.pkl (passes 1-3)
uv run python edits/03_posfix.py          # needs logs/passes/pos_subspaces*.pkl (passes 10-11)
uv run python edits/04_unlate.py          # data-free given the hypothesis
uv run python edits/05_combined.py
uv run python -m wsp.evaluate --model checkpoints/r25_compress_revertlate --name r25_compress_revertlate
uv run python -m wsp.audit --rebuild
```

The edited checkpoints load through `wsp.model.load(path)`. Structural pruning gives each layer its own head and FFN sizes, which stock `transformers` cannot represent. The quality-only edits (`c_*`) keep the original architecture, so plain `CrossEncoder(path)` loads them as drop-in replacements.
