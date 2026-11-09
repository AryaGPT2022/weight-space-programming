"""Pass 11 (base): do the [CLS]-reading heads have a positional prior? 32 items x
{orig, pre80, pre160, post160} with attentions + hidden. For each head we measure the attention
from [CLS] per relevant-content token (density) - a content-driven head keeps similar density when
the content moves; a positional head loses it."""
import sys, pickle; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from wsp.harness import ForwardBudget
P = pickle.load(open("logs/passes/p08_leadbias.pkl", "rb"))
items = sorted(P["item"].unique())[24:56]   # items not used in pass 10
pairs = P[P["item"].isin(items) & P.variant.isin(["orig", "pre80", "pre160", "post160"])].drop(columns="logit").reset_index(drop=True)
fb = ForwardBudget("main", 25)
out = fb.probe("checkpoints/base", pairs, want=("attentions", "hidden"), note="CLS positional prior: attentions orig/pre80/pre160/post160")
print("passes used:", fb.used)
