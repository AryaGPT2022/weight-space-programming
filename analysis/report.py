"""Collect results/*.json into results/summary.md and results/pareto.png."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = {p.stem: json.loads(p.read_text()) for p in (ROOT / "results").glob("*.json")}
GROUPS = {  # name -> (group, passes, label)
    "base": ("base", "-", "base"),
    "bl_drop_top2": ("baseline", "-", "drop top-2 layers"),
    "bl_drop_alt2": ("baseline", "-", "drop layers 1,3"),
    "bl_mag_uniform33": ("baseline", "-", "magnitude −33%"),
    "bl_drop_alt3": ("baseline", "-", "drop layers 1,3,5"),
    "bl_mag_uniform50": ("baseline", "-", "magnitude −50%"),
    "r0_compress": ("claude", "0", "R0 compress"),
    "r10_compress": ("claude", "7", "R10 compress"),
    "r10_compress_xs": ("claude", "7", "R10 compress-xs"),
    "c_posfix": ("claude-q", "12", "CLS-head pos-fix"),
    "c_unlate": ("claude-q", "14", "unlate (rank-1)"),
    "c_revertlate": ("claude-q", "17", "revert-late"),
    "r25_compress_revertlate": ("claude", "17", "R25 compress+revert-late"),
    "r25_compress_xs_revertlate": ("claude", "17", "R25 compress-xs+revert-late"),
}
order = [k for k in GROUPS if k in R]

rows = ["| model | passes | enc+head params | total params | MS MARCO nDCG@10 | MS MARCO AUC | BEIR nDCG@10 | BEIR AUC | F1 (global thr) | CPU pairs/s | MPS pairs/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
b = R["base"]["summary"]
for k in order:
    r, s, z, sp = R[k], R[k]["summary"], R[k]["size"], R[k].get("speed", {})
    d = lambda key: f"{s[key]:.4f} ({(s[key]-b[key])*100:+.1f})" if k != "base" else f"{s[key]:.4f}"
    rows.append(f"| {GROUPS[k][2]} (`{k}`) | {GROUPS[k][1]} | {z['params_encoder_head']/1e6:.2f}M | {z['params_total']/1e6:.2f}M | "
                f"{d('msmarco_ndcg@10')} | {d('msmarco_auc')} | {d('beir_ndcg@10')} | {d('beir_auc')} | {d('f1_global_thr')} | "
                f"{sp.get('cpu_pairs_per_s', 0):.0f} | {sp.get('mps_pairs_per_s', 0):.0f} |")
per = ["", "Per-dataset nDCG@10 / AUC:", "", "| model | " + " | ".join(R['base']['per_dataset']) + " |",
       "|---|" + "---|" * len(R['base']['per_dataset'])]
for k in order:
    per.append(f"| `{k}` | " + " | ".join(f"{v['ndcg@10']:.3f} / {v['auc']:.3f}" for v in R[k]["per_dataset"].values()) + " |")
(ROOT / "results" / "summary.md").write_text("\n".join(rows + per) + "\n")
print("\n".join(rows + per))

# Pareto: encoder params vs nDCG, two panels (one y-scale each)
COL = {"baseline": "#eb6834", "claude": "#2a78d6", "claude-q": "#1baf7a", "base": "#0b0b0b"}
NAME = {"baseline": "data-free baselines", "claude": "Claude weight edits (compression)",
        "claude-q": "Claude weight edits (quality only)", "base": "base model"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#fcfcfb")
for ax, key, title in [(axes[0], "msmarco_ndcg@10", "In-domain: MS MARCO nDCG@10"),
                       (axes[1], "beir_ndcg@10", "Out-of-domain: BEIR avg nDCG@10")]:
    ax.set_facecolor("#fcfcfb")
    seen = set()
    for k in order:
        g = GROUPS[k][0]
        x, y = R[k]["size"]["params_encoder_head"] / 1e6, R[k]["summary"][key]
        ax.scatter(x, y, s=70, color=COL[g], edgecolor="#fcfcfb", linewidth=2, zorder=3,
                   label=NAME[g] if g not in seen else None)
        seen.add(g)
        if g != "claude-q":
            ax.annotate(GROUPS[k][2], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color="#52514e")
    ax.set_title(title, fontsize=11, color="#0b0b0b", loc="left")
    ax.set_xlabel("encoder + head parameters (M)", color="#52514e")
    ax.grid(color="#e8e7e3", linewidth=0.8)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.tick_params(colors="#52514e")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "results" / "pareto.png", dpi=150)
