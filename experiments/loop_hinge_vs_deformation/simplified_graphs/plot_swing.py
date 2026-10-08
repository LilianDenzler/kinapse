#!/usr/bin/env python
"""Plot clamp-anchored rigid-motion (theta swing) across TCRs, and the ε_deform vs θ two-axis map."""
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from graph_build import HERE

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
COL = ["#4C72B0", "#4C72B0", "#2E5A88", "#C44E52", "#C44E52", "#8C2D2F"]
sw = sorted(glob.glob(f"{HERE}/results_swing/*.npz"))
tcrs = [os.path.basename(f)[:4] for f in sw]
TH = np.array([[float(np.load(f)[f"{c}_theta"]) for c in CDRS] for f in sw])
FR = np.array([[float(np.load(f)[f"{c}_frac"]) for c in CDRS] for f in sw])
EPS = np.array([[float(np.load(f"{HERE}/results_deform/{t}.npz")[f"{c}_eps"]) for c in CDRS] for t in tcrs])
print(f"{len(sw)} TCRs")

rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.3, 1.2, 1.1]})

# (1) theta swing spread per CDR + combined
data = [TH[:, j] for j in range(6)] + [TH.ravel()]
bp = ax[0].boxplot(data, positions=range(7), widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.5))
for k, p in enumerate(bp["boxes"]):
    p.set_facecolor((COL + ["#777"])[k]); p.set_alpha(.22)
for j in range(6):
    ax[0].scatter(np.full(len(sw), j) + rng.uniform(-.16, .16, len(sw)), TH[:, j], c=COL[j], s=26, edgecolor="k", linewidths=.3, zorder=3)
ax[0].scatter(np.full(TH.size, 6) + rng.uniform(-.16, .16, TH.size), TH.ravel(), c="#555", s=10, alpha=.4, zorder=3)
ax[0].set_xticks(range(7)); ax[0].set_xticklabels(CDRS + ["ALL"], fontsize=8, rotation=15)
ax[0].set_ylabel("θ rigid swing about clamp (deg, median)"); ax[0].set_ylim(0, None)
ax[0].set_title("Rigid hinge amplitude per CDR across 22 TCRs")

# (2) two-axis map: eps_deform (shape) vs theta (rigid), one point per TCR
for j, c in enumerate(CDRS):
    ax[1].scatter(EPS[:, j], TH[:, j], c=COL[j], s=28, edgecolor="k", linewidths=.3, label=c, alpha=.85)
ax[1].set_xlabel("ε_deform  (internal deformation)"); ax[1].set_ylabel("θ  (rigid hinge swing, deg)")
ax[1].set_title("Two-axis map: deformation vs rigid motion\n(each dot = one TCR×CDR)"); ax[1].legend(fontsize=7, ncol=2)
ax[1].axvline(np.median(EPS), color="0.7", lw=.7, ls=":"); ax[1].axhline(np.median(TH), color="0.7", lw=.7, ls=":")

# (3) pivot fraction per CDR (is the rigid axis on the clamp?)
data2 = [FR[:, j] for j in range(6)]
bp2 = ax[2].boxplot(data2, positions=range(6), widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.5))
for k, p in enumerate(bp2["boxes"]):
    p.set_facecolor(COL[k]); p.set_alpha(.22)
for j in range(6):
    ax[2].scatter(np.full(len(sw), j) + rng.uniform(-.14, .14, len(sw)), FR[:, j], c=COL[j], s=22, edgecolor="k", linewidths=.3, zorder=3)
ax[2].axhline(0.5, color="k", lw=.8, ls="--")
ax[2].text(5.6, 0.52, "C-stem", fontsize=7, va="bottom"); ax[2].text(5.6, 0.48, "N-stem", fontsize=7, va="top")
ax[2].set_xticks(range(6)); ax[2].set_xticklabels(CDRS, fontsize=8, rotation=20)
ax[2].set_ylabel("pivot fraction (0=N stem, 1=C stem)"); ax[2].set_ylim(-0.1, 1.1)
ax[2].set_title("Where the rigid axis pivots\n(clamp: CDR1→C, CDR2/3→N)")

fig.suptitle("Clamp-anchored RIGID motion (θ) — companion to ε_deform — across 22 TCRs", y=1.02, fontsize=13)
out = f"{HERE}/figures/swing_spread.png"
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("fig ->", out)
print("median θ per CDR:", {c: round(float(np.median(TH[:, j])), 1) for j, c in enumerate(CDRS)})
print("median pivot frac:", {c: round(float(np.median(FR[:, j])), 2) for j, c in enumerate(CDRS)})
