"""Edit 04 - quality: subtract the learned "lateness" direction from the position-embedding table.

Programmer: Claude (Opus 5.5). This is the analog of the blog's Exp-8 edit, which subtracted a
looping direction.

Evidence
  pass 8   : moving a relevant passage 160 tokens later costs -8.5 logits, against -2.0 for
             appending the same filler (a -6.5 position effect)
  pass 11-13 : the [CLS]-reading heads' positional prior explains only ~15-20% of it (edit 03)
  weights  : comparing with the pretrained MiniLM (nreimers/MiniLM-L6-H384-uncased), position rows
             128-256 were *rotated* during MS-MARCO fine-tuning (cos 0.63-0.67 vs 0.95 for early
             rows; norm unchanged, so this is not weight decay). Their updates share a common
             direction v that is ~0 for positions <96 and >256 and peaks at 160-192.

The program (data-free, one tensor, rank 1)
    D_i  = P_ft[i] - P_pre[i]                     fine-tuning update of position row i
    c    = mean_{i in 2..95} D_i                  update shared with early positions
    v    = normalise( mean_{i in 128..255} (D_i - c) )
    P_new[i] = P_ft[i] - ((D_i - c) · v) v        for every position i
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModel

from wsp import model as M, surgery as S

from wsp.config import PRETRAINED_ID as PRETRAINED, PRETRAINED_REVISION


def lateness(P_ft):
    P_pre = AutoModel.from_pretrained(PRETRAINED, revision=PRETRAINED_REVISION).embeddings.position_embeddings.weight.detach().numpy()
    D = P_ft - P_pre
    c = D[2:96].mean(0)
    v = (D[128:256] - c).mean(0)
    v /= np.linalg.norm(v)
    return (D - c) @ v, v


@torch.no_grad()
def apply_unlate(m):
    pe = m.bert.embeddings.position_embeddings.weight
    P = pe.detach().cpu().numpy()
    coef, v = lateness(P)
    pe.copy_(torch.as_tensor(P - np.outer(coef, v), dtype=pe.dtype, device=pe.device))
    return coef


def main(parent="base", name="c_unlate"):
    m, t, a = M.load(M.CKPT / parent, "cpu")
    coef = apply_unlate(m)
    if a.get("cls_only_last"):
        M.make_cls_only(m.bert.encoder.layer[-1])
    rep = S.diff_report(load_file(M.CKPT / "base" / "model.safetensors"), m.state_dict(), a)
    tens = rep.pop("tensors_touched")
    meta = {"script": Path(__file__).name, "parent": parent, "tensors": tens,
            "lateness_coef_by_block": {f"{i}-{i+32}": float(np.mean(coef[i:i + 32])) for i in range(0, 512, 32)}, **rep}
    sha = M.save(m, t, a, M.CKPT / name, meta=meta)
    print(name, "parent", parent, "tensors", tens, f"{rep['pct_touched']:.3f}% of weights", sha[:12])


@torch.no_grad()
def apply_revert_late(m, lo=96, hi=256, ramp=32):
    """Stronger variant (hypothesis E1): drop ALL late-specific fine-tuning of rows lo..hi,
    keeping only the update shared with early positions: P_new[i] = P_pre[i] + c, ramped in
    linearly over lo..lo+ramp so adjacent rows stay continuous."""
    pe = m.bert.embeddings.position_embeddings.weight
    P = pe.detach().cpu().numpy()
    P_pre = AutoModel.from_pretrained(PRETRAINED, revision=PRETRAINED_REVISION).embeddings.position_embeddings.weight.detach().numpy()
    c = (P - P_pre)[2:96].mean(0)
    w = np.clip((np.arange(len(P)) - lo) / ramp, 0, 1) * (np.arange(len(P)) < hi)
    P_new = (1 - w)[:, None] * P + w[:, None] * (P_pre + c)
    pe.copy_(torch.as_tensor(P_new, dtype=pe.dtype, device=pe.device))
    return w


def main_revert(parent="base", name="c_revertlate"):
    m, t, a = M.load(M.CKPT / parent, "cpu")
    apply_revert_late(m)
    if a.get("cls_only_last"):
        M.make_cls_only(m.bert.encoder.layer[-1])
    rep = S.diff_report(load_file(M.CKPT / "base" / "model.safetensors"), m.state_dict(), a)
    tens = rep.pop("tensors_touched")
    sha = M.save(m, t, a, M.CKPT / name, meta={"script": Path(__file__).name, "parent": parent, "variant": "revert_late", "tensors": tens, **rep})
    print(name, "parent", parent, f"{rep['pct_touched']:.3f}% of weights", sha[:12])


if __name__ == "__main__":
    main()
    main_revert()
