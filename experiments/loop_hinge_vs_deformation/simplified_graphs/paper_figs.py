#!/usr/bin/env python
"""Generate real loop-flexibility figures for the Nature-Comms mock-up from our MD decomposition results,
and copy the rigid/deform/DOF/entropy decomposition figures into the paper. Fills placeholder PNGs in place."""
from __future__ import annotations
import os, sys, json, shutil
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = "/mnt/ssd2tb/lilian/projects/Graphormer/mock_up_nture_comms"
IMG = f"{PAPER}/images_png"; DYN = f"{PAPER}/figures/7-dynamics"
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]

res = json.load(open(f"{HERE}/results_rigid_deform.json"))
tcrs = sorted(res.keys())
# RMSD-like flexibility per (TCR, region) = sqrt(total_variance / N)  (Å)
flex = np.array([[np.sqrt(res[t][c]["total"] / res[t][c]["N"]) for c in CDRS] for t in tcrs])

# --- fig:loop_flexibility_summary  (heatmap TCR x region) ---
fig, ax = plt.subplots(figsize=(8, 9))
im = ax.imshow(flex, aspect="auto", cmap="viridis")
ax.set_xticks(range(6)); ax.set_xticklabels([c.replace("_", " ") for c in CDRS], rotation=30, fontsize=9)
ax.set_yticks(range(len(tcrs))); ax.set_yticklabels(tcrs, fontsize=8)
for y in range(len(tcrs)):
    for x in range(6):
        ax.text(x, y, f"{flex[y,x]:.1f}", ha="center", va="center", fontsize=6.5,
                color="white" if flex[y, x] < flex.max() * 0.6 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("loop flexibility  (mean C$\\alpha$ RMSD, Å)", fontsize=9)
ax.set_title("Ground-truth CDR loop flexibility under MD", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{IMG}/loop_flexibility_analysis.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# --- fig:loop_flex_A / _B  (per-chain grouped bars) ---
for chain, fname in [("A", "loop_flex_analysis_combined_A.png"), ("B", "loop_flex_analysis_combined_B.png")]:
    cols = [c for c in CDRS if c[0] == chain]; ci = [CDRS.index(c) for c in cols]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(tcrs)); w = 0.26
    colmap = {"CDR1": "#4C72B0", "CDR2": "#55A868", "CDR3": "#C44E52"}
    for k, (c, idx) in enumerate(zip(cols, ci)):
        ax.bar(x + (k - 1) * w, flex[:, idx], w, color=colmap[c[2:]], label=c.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels(tcrs, rotation=90, fontsize=8)
    ax.set_ylabel("loop flexibility (mean C$\\alpha$ RMSD, Å)")
    ax.set_title(f"CDR loop flexibility across the dataset — {chain}-chain (framework fit)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{DYN}/{fname}", dpi=150, bbox_inches="tight"); plt.close(fig)

# --- copy decomposition figures into the paper ---
copies = {"rigid_vs_deform_all.png": "loop_rigid_deform.png", "atlas_dof.png": "loop_dof_atlas.png",
          "entropy_atlas.png": "loop_entropy.png", "fraction_vs_length.png": "loop_fraction_length.png"}
for src, dst in copies.items():
    shutil.copy(f"{HERE}/figures/{src}", f"{DYN}/{dst}")
    print("copied", dst)

# --- stats for the manuscript text ---
print("\n--- numbers for the text ---")
for c in CDRS:
    i = CDRS.index(c)
    fr = np.array([res[t][c]["deform"] / res[t][c]["total"] * 100 for t in tcrs])
    print(f"{c}: flex(RMSD) {flex[:,i].min():.1f}-{flex[:,i].max():.1f} Å ; deform% {fr.min():.0f}-{fr.max():.0f} (median {np.median(fr):.0f})")
perR_frac = np.array([sum(res[t][c]["deform"] for c in CDRS) / sum(res[t][c]["total"] for c in CDRS) * 100 for t in tcrs])
print(f"per-TCR deform fraction: {perR_frac.min():.0f}-{perR_frac.max():.0f}%")
b3 = CDRS.index("B_CDR3")
print(f"CDR3beta flex range: {flex[:,b3].min():.1f}-{flex[:,b3].max():.1f} Å")
print("done")
