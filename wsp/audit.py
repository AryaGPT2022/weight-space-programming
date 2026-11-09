"""Audit: budgets respected, test split touched only by declared finals, no gradients in edits,
edits reproduce their checkpoints bit-for-bit.

    uv run python -m wsp.audit [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from wsp.harness import ledger
from wsp.model import CKPT, sha256

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = json.loads((ROOT / "logs" / "budgets.json").read_text()) if (ROOT / "logs" / "budgets.json").exists() else {}
FORBIDDEN = re.compile(r"\.backward\(|torch\.optim|autograd|requires_grad_\(True\)|\.grad\b|Trainer\(")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-run every edit and compare SHA-256")
    a = ap.parse_args()
    ok = True

    passes = Counter(r["run"] for r in ledger("pass"))
    print("forward passes per run:")
    for run, n in sorted(passes.items()):
        b = BUDGETS.get(run)
        flag = "OK" if b is not None and n <= b else "OVER/UNDECLARED"
        ok &= flag == "OK"
        print(f"  {run:12s} used {n:3d} / budget {b}  {flag}")

    finals = set(json.loads((ROOT / "logs" / "finals.json").read_text())) if (ROOT / "logs" / "finals.json").exists() else set()
    tests = Counter(r["name"] for r in ledger("test_eval"))
    print("test-split evaluations:")
    for name, n in sorted(tests.items()):
        flag = "OK" if name in finals else "NOT A DECLARED FINAL"
        ok &= flag == "OK"
        print(f"  {name:28s} x{n}  {flag}")

    print("gradient/optimizer scan of edits/:")
    for f in sorted((ROOT / "edits").glob("*.py")):
        hits = [l for l in f.read_text().splitlines() if FORBIDDEN.search(l)]
        ok &= not hits
        print(f"  {f.name:32s} {'CLEAN' if not hits else 'VIOLATION: ' + str(hits)}")

    if a.rebuild:
        print("reproducibility (re-run edits, compare SHA-256):")
        for f in sorted((ROOT / "edits").glob("[0-9]*.py")):
            meta_files = list(CKPT.glob("*/edit_meta.json"))
            outs = [m.parent for m in meta_files if json.loads(m.read_text()).get("script") == f.name]
            before = {o.name: sha256(o / "model.safetensors") for o in outs}
            subprocess.run([sys.executable, str(f)], check=True, cwd=ROOT, capture_output=True)
            for o in outs:
                same = sha256(o / "model.safetensors") == before[o.name]
                ok &= same
                print(f"  {f.name:32s} -> {o.name:24s} {'IDENTICAL' if same else 'DIFFERENT'}")
    print("AUDIT", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
