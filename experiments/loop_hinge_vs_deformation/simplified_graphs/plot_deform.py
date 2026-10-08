#!/usr/bin/env python
"""Cross-TCR plot of reference-free CDR deformation: eps (fractional) and D_deform (absolute)."""
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from graph_build import HERE

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
files = sorted(glob.glob(f"{HERE}/results_deform/*.npz"))
tcrs = [os.path.basename(f)[:4] for f in files]
EPS = np.array([[float(np.load(f)[f"{c}_eps"]) for c in CDRS] for f in files])
DAB = np.array([[float(np.load(f)[f"{c}_Ddef"]) for c in CDRS] for f in files])
LEN = np.array([[int(np.load(f)[f"{c}_N"]) for c in CDRS] for f in files])
print(f"{len(files)} TCRs")

COL = ["#4C72B0", "#4C72B0", "#2E5A88", "#C44E52", "#C44E52", "#8C2D2F"]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw={"width_ratios": [1.5, 1.5, 1.0]})


def stripbox(ax, M, ylab, title):
    data = [M[:, j] for j in range(6)] + [M.ravel()]
    bp = ax.boxplot(data, positions=range(7), widths=.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.5))
    for k, p in enumerate(bp["boxes"]):
        p.set_facecolor((COL + ["#777"])[k]); p.set_alpha(.22)
    for j in range(6):
        ax.scatter(np.full(len(files), j) + rng.uniform(-.16, .16, len(files)), M[:, j],
                   c=COL[j], s=26, edgecolor="k", linewidths=.3, zorder=3)
    ax.scatter(np.full(M.size, 6) + rng.uniform(-.16, .16, M.size), M.ravel(), c="#555", s=10, alpha=.4, zorder=3)
    ax.set_xticks(range(7)); ax.set_xticklabels(CDRS + ["ALL"], fontsize=8, rotation=15)
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.set_ylim(0, None)


stripbox(axes[0], EPS, "ε_deform  (fractional, dimensionless)",
         "Reference-free CDR deformation ε = √⟨Var[d]/⟨d⟩²⟩  (dot = TCR)")
stripbox(axes[1], DAB, "D_deform (Å, absolute)", "Absolute D_deform = √⟨Var[d]⟩")
# eps vs CDR3 length
ax = axes[2]
for j, c in [(2, "A_CDR3"), (5, "B_CDR3")]:
    ax.scatter(LEN[:, j], EPS[:, j], s=30, c=COL[j], edgecolor="k", linewidths=.3, label=c)
    r = np.corrcoef(LEN[:, j], EPS[:, j])[0, 1]
    ax.text(.03, .12 - .07 * (j > 3), f"{c}: r={r:.2f}", transform=ax.transAxes, fontsize=9, color=COL[j])
ax.set_xlabel("CDR3 loop length (residues)"); ax.set_ylabel("ε_deform"); ax.set_title("ε vs CDR3 length"); ax.legend(fontsize=8)

fig.suptitle("Reference-free CDR internal deformation (variance of intra-loop distances) across 22 TCRs", y=1.02, fontsize=13)
out = f"{HERE}/figures/deform_spread.png"
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("fig ->", out)
print("median eps per CDR:", {c: round(float(np.median(EPS[:, j])), 3) for j, c in enumerate(CDRS)})
