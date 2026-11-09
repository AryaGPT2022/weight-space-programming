"""Re-run the pass-8 lead-bias pairs on a model (1 pass) and compare with base."""
import sys, pickle; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from wsp.harness import ForwardBudget
model = sys.argv[1]
P = pickle.load(open("logs/passes/p08_leadbias.pkl", "rb"))
fb = ForwardBudget("main", 25)
s = fb.probe(f"checkpoints/{model}", P.drop(columns="logit"), note=f"lead-bias test on {model}")["logits"]
P = P.assign(edit=s)
for col in ("logit", "edit"):
    w = P.pivot(index="item", columns="variant", values=col)
    print(f"{'base' if col=='logit' else model:12s} orig {w.orig.mean():+.2f} | Δpre80 {(w.pre80-w.orig).mean():+.2f} Δpost80 {(w.post80-w.orig).mean():+.2f} "
          f"Δpre160 {(w.pre160-w.orig).mean():+.2f} Δpost160 {(w.post160-w.orig).mean():+.2f} | position effect @160 {(w.pre160-w.post160).mean():+.2f}")
print("passes used:", fb.used)
