#!/usr/bin/env python
"""Why each CDR loop is clamped on one stem: the V(D)J-recombination / germline origin, plus the data.

Top panel  : schematic of the TCR V-domain gene structure -> where each CDR comes from and why one stem
             of each loop is anchored (conserved germline landmark) while the other / the apex is free.
Bottom panel: the measured N-stem vs C-stem fluctuation-to-rigid for all 6 CDRs (both chains), showing the
             constrained side is consistent between alpha and beta.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc
from loop_flank_analysis import load_all, rigid_definition, dev_to_rigid, profile
from graph_build import CDR_RANGES, HERE

# ---- data: N-stem vs C-stem fluctuation-to-rigid per CDR, per chain ----
data = load_all()
stem = {}
for ch in "AB":
    _, ultra = rigid_definition(data, ch); rigid = set(ultra)
    present = set(profile(data, ch))
    for cdr, (lo, hi) in CDR_RANGES.items():
        Ns = [r for r in (lo, lo+1, lo+2) if r in present]
        Cs = [r for r in (hi, hi-1, hi-2) if r in present]
        mm, _ = dev_to_rigid(data, ch, Ns + Cs, rigid)
        stem[(ch, cdr)] = (float(np.mean([mm[r] for r in Ns])), float(np.mean([mm[r] for r in Cs])))

# constrained side + the germline reason for each CDR (same for both chains)
CLAMP = {
    "CDR1": ("C", "C-term clamp: Trp41 / FR2 β-strand"),
    "CDR2": ("N", "N-term clamp: FR2 end β-strand"),
    "CDR3": ("N", "N-term clamp: Cys104 disulfide\n+ V-gene C-A-x anchor"),
}
GREEN, ORANGE, BLUE, RED, GREY = "#4C9A5B", "#E1943B", "#4E79A7", "#C0504D", "#B8B8B8"

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.32)

# ================= TOP: recombination schematic =================
axS = fig.add_subplot(gs[0]); axS.set_xlim(0, 132); axS.set_ylim(-2.6, 5.2); axS.axis("off")
axS.set_title("Where the clamps come from: TCR V-domain gene structure (V(D)J recombination)", fontsize=12, pad=8)

# row 1: gene segments (y ~ 3.7)
axS.add_patch(Rectangle((3, 3.6), 104 - 3, 0.7, color=GREEN, alpha=.85))
axS.add_patch(Rectangle((104, 3.6), 116 - 104, 0.7, color=ORANGE, alpha=.9, hatch="///", ec="white"))
axS.add_patch(Rectangle((116, 3.6), 129 - 116, 0.7, color=BLUE, alpha=.85))
axS.text(53, 3.95, "V gene segment  (germline)", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
axS.text(110, 4.35, "V–(D)–J junction\nN/D/N (random, non-germline)", ha="center", va="bottom", color=ORANGE, fontsize=8.5, fontweight="bold")
axS.text(122.5, 3.95, "J gene\n(germline)", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

# row 2: protein domain regions (y ~ 1.9)
REGIONS = [("FR1", 3, 26, GREY), ("CDR1", 27, 38, RED), ("FR2", 39, 55, GREY), ("CDR2", 56, 65, RED),
           ("FR3", 66, 104, GREY), ("CDR3", 105, 117, RED), ("FR4", 118, 128, GREY)]
for name, lo, hi, col in REGIONS:
    axS.add_patch(Rectangle((lo - .5, 1.7), hi - lo + 1, 0.8, color=col, alpha=.55 if col == GREY else .8, ec="k", lw=.4))
    axS.text((lo + hi) / 2, 2.1, name, ha="center", va="center", fontsize=8, fontweight="bold" if col == RED else "normal")
axS.text(-1, 2.1, "protein\ndomain", ha="right", va="center", fontsize=8, color="0.4")
axS.text(-1, 3.95, "gene", ha="right", va="center", fontsize=8, color="0.4")

# disulfide arc Cys23 - Cys104
axS.add_patch(Arc(((23 + 104) / 2, 2.5), 104 - 23, 2.2, angle=0, theta1=0, theta2=180, color="#B8860B", lw=1.6, ls="-"))
for r in (23, 104):
    axS.plot([r], [2.5], "o", ms=5, color="#B8860B")
axS.text((23 + 104) / 2, 3.55, "Cys23–Cys104 disulfide", ha="center", va="bottom", fontsize=8.5, color="#8B6508")

# clamp arrows (point up to the constrained stem) + reasons
for name, lo, hi, col in REGIONS:
    if name not in CLAMP:
        continue
    side, reason = CLAMP[name]
    xpos = (hi if side == "C" else lo)
    axS.add_patch(FancyArrowPatch((xpos, 0.2), (xpos, 1.6), arrowstyle="-|>", mutation_scale=14, color="#1a1a1a", lw=1.6))
    axS.text(xpos, -0.15, f"{name}\n{reason}", ha="center", va="top", fontsize=7.6,
             bbox=dict(boxstyle="round,pad=0.25", fc="#FFF7E6", ec="#C9A227", lw=.8))
axS.annotate("CDR3 apex = junctional diversity\n(N/D/N, no germline) → free / floppy",
             xy=(111, 1.7), xytext=(111, 4.9), ha="center", fontsize=8, color=ORANGE, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))

# ================= BOTTOM: the data =================
axB = fig.add_subplot(gs[1])
order = [(cdr, ch) for cdr in ["CDR1", "CDR2", "CDR3"] for ch in "AB"]
x = np.arange(len(order)); w = 0.38
for k, (cdr, ch) in enumerate(order):
    Nf, Cf = stem[(ch, cdr)]
    side = CLAMP[cdr][0]
    bN = axB.bar(x[k] - w/2, Nf, w, color="#8FB8DE", ec="k", lw=.5)
    bC = axB.bar(x[k] + w/2, Cf, w, color="#2E5A88", ec="k", lw=.5)
    # star the constrained (lower) stem
    lowx, lowy = (x[k] - w/2, Nf) if side == "N" else (x[k] + w/2, Cf)
    axB.plot(lowx, lowy + .02, "*", ms=15, color="#C0392B", zorder=5)
axB.set_xticks(x); axB.set_xticklabels([f"{ch}_{cdr}" for cdr, ch in order])
axB.set_ylabel("stem fluctuation to RIGID set (Å)\n(22-TCR average)")
axB.set_title("Measured: one stem of each loop is clamped (★ = constrained side) — and it is the SAME side in α and β\n"
              "CDR1 → C-terminal · CDR2 → N-terminal · CDR3 → N-terminal", fontsize=11)
from matplotlib.patches import Patch
axB.legend(handles=[Patch(fc="#8FB8DE", ec="k", label="N-terminal stem"),
                    Patch(fc="#2E5A88", ec="k", label="C-terminal stem"),
                    plt.Line2D([], [], ls="", marker="*", ms=13, color="#C0392B", label="constrained (clamped) stem")],
           loc="upper left", fontsize=9)
for k in (1.5, 3.5):
    axB.axvline(k, color="0.85", lw=1)
axB.margins(y=.18)

fig.suptitle("Each CDR loop is asymmetrically clamped — the anchored stem is fixed by germline/recombination structure",
             y=0.995, fontsize=13)
out = f"{HERE}/figures/clamp_recombination.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("fig ->", out)
for k in order:
    print(f"  {k[1]}_{k[0]}: N={stem[(k[1],k[0])][0]:.2f}  C={stem[(k[1],k[0])][1]:.2f}  clamp={CLAMP[k[0]][0]}")
