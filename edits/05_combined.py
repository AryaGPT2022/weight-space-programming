"""Edit 05: stack the quality edit (04 revert-late) on the compressed models (02).

Compression never touches the position table, and revert-late touches nothing else, so the two edits
are disjoint in weight space.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib

from safetensors.torch import load_file

from wsp import model as M, surgery as S

unlate = importlib.import_module("04_unlate")

for parent, name in [("r10_compress", "r25_compress_revertlate"), ("r10_compress_xs", "r25_compress_xs_revertlate")]:
    m, t, a = M.load(M.CKPT / parent, "cpu")
    unlate.apply_revert_late(m)
    rep = S.diff_report(load_file(M.CKPT / "base" / "model.safetensors"), m.state_dict(), a)
    rep.pop("tensors_touched")
    sha = M.save(m, t, a, M.CKPT / name, meta={"script": Path(__file__).name, "parent": parent,
                                               "stack": ["02_r10_compress", "04_unlate.apply_revert_late"], **rep})
    print(name, M.param_counts(m), f"{rep['pct_touched']:.1f}% touched", sha[:12])
