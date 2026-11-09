"""Edit 03 - quality: remove the positional prior of [CLS]-reading heads ("lead bias").

Programmer: Claude (Opus 5.5). Evidence: pass 8 (shifting a relevant passage 160 tokens later costs
-8.5 logits vs -2.0 for appending the same filler), pass 10 (token-match heads lose match mass), pass 11
(at equal dilution, [CLS]-reading heads L5H11 L4H8 L5H0 L4H3 L4H6 L5H6 lose ~half their attention on the
relevant content when it moves later).

The program, for each target head (l,h):
    W_K <- W_K - Π_pos(l) · W_K · Π_qCLS(l,h)
  Π_pos(l)    : positional subspace of the residual entering layer l (per-position means of passage
                tokens at positions >= 30, PCs above a shuffled-position null; 8 dims at L4, 3 at L5)
  Π_qCLS(l,h) : span of the head's [CLS] query vectors (95% energy; rank 1-3)
Key components that are positional AND visible to [CLS] queries are removed. Every other
query/key interaction is unchanged. Rank <= 24 per head.
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from safetensors.torch import load_file
from wsp import model as M, surgery as S
from wsp.model import HEAD_DIM

TARGETS = [(5, 11), (4, 8), (5, 0), (4, 3), (4, 6), (5, 6)]
ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def apply_posfix(m, a, subspaces, targets=TARGETS):
    touched = []
    for (l, h) in targets:
        if l not in a["keep_layers"] or h not in a["layers"][str(l)]["heads"]:
            continue
        sp = subspaces[(l, h)]
        U = torch.as_tensor(sp["U_pos"], dtype=torch.float32)        # (384, k)
        V = torch.as_tensor(sp["Vq"], dtype=torch.float32)           # (32, r)
        L = m.bert.encoder.layer[a["keep_layers"].index(l)]
        i = a["layers"][str(l)]["heads"].index(h)
        sl = slice(i * HEAD_DIM, (i + 1) * HEAD_DIM)
        Wk = L.attention.self.key.weight[sl].T.clone().cpu()           # (384, 32): x @ Wk
        Wk_new = Wk - U @ (U.T @ Wk) @ V @ V.T
        L.attention.self.key.weight[sl] = Wk_new.T.to(L.attention.self.key.weight.device)
        touched.append((l, h, U.shape[1], V.shape[1]))
    return touched


def main(parent="base", name="c_posfix"):
    sub = pickle.load(open(ROOT / "logs/passes/pos_subspaces.pkl", "rb"))
    m, t, a = M.load(M.CKPT / parent, "cpu")
    touched = apply_posfix(m, a, sub)
    if a.get("cls_only_last"):
        M.make_cls_only(m.bert.encoder.layer[-1])
    rep = S.diff_report(load_file(M.CKPT / "base" / "model.safetensors"), m.state_dict(), a)
    tens = rep.pop("tensors_touched")
    meta = {"script": Path(__file__).name, "parent": parent, "heads_edited(l,h,k_pos,r_q)": touched,
            "tensors": tens, **rep}
    sha = M.save(m, t, a, M.CKPT / name, meta=meta)
    print(name, "edited", touched, f"{rep['weights_changed_in_place']} weights changed in place", sha[:12])


TARGETS_V2 = TARGETS + [(3, 0), (3, 3), (4, 7), (4, 9), (3, 6)]   # all L3-L5 heads with pass-11 shift loss >= 0.15


def main_v2(parent="base", name="c_posfix2"):
    """Iteration 2 (after pass 12 showed only -13%): paired-null positional dims, 99% CLS-query energy,
    and every L3-L5 head with shift loss >= 0.15."""
    global TARGETS
    TARGETS = TARGETS_V2
    sub = pickle.load(open(ROOT / "logs/passes/pos_subspaces_v2.pkl", "rb"))
    m, t, a = M.load(M.CKPT / parent, "cpu")
    touched = apply_posfix(m, a, sub, TARGETS_V2)
    if a.get("cls_only_last"):
        M.make_cls_only(m.bert.encoder.layer[-1])
    rep = S.diff_report(load_file(M.CKPT / "base" / "model.safetensors"), m.state_dict(), a)
    tens = rep.pop("tensors_touched")
    meta = {"script": Path(__file__).name, "parent": parent, "iteration": 2,
            "heads_edited(l,h,k_pos,r_q)": touched, "tensors": tens, **rep}
    sha = M.save(m, t, a, M.CKPT / name, meta=meta)
    print(name, "edited", touched, f"{rep['weights_changed_in_place']} weights changed in place", sha[:12])


if __name__ == "__main__":
    main()
    main_v2()
