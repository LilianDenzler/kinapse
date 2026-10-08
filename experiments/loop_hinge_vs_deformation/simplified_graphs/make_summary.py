#!/usr/bin/env python
"""One-page SUMMARY of the alignment-free decomposition of TCR CDR-loop motion (22 TCRs):
 (A) the rigid axes + measured angles (embedded PyMOL render)
 (B) how we measure it, alignment-free: d_LL -> deformation, d_LR -> rigid pose
 (C) rigid pose vs deformation per CDR
 (D) which TCRs are more rigid vs more deforming
 (E) the honest twist caveat: most 'twist' is corkscrew deformation, not rigid
 (F) method -> conclusion
-> figures/SUMMARY_alignfree.png"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from graph_build import HERE

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
G, R_, B_ = "#2E7D32", "#C0392B", "#0072B2"
res = json.load(open(f"{HERE}/results_alignfree.json"))
tcrs = list(res)
med = lambda key, cdr: float(np.median([res[t][cdr][key] for t in tcrs]))

fig = plt.figure(figsize=(18, 10.5))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.33, wspace=0.26)

# (A) axes render
axA = fig.add_subplot(gs[0, 0]); axA.axis("off")
img = f"{HERE}/figures/axes_3SKN_B_CDR1.png"
if os.path.exists(img):
    axA.imshow(plt.imread(img))
axA.set_title("A  Rigid axes through the clamp + measured swing angles\n"
              "ĉ green = tip-elevation (hinge) · ĥ purple = twist · n̂ orange = sway", fontsize=10, fontweight="bold")

# (B) method schematic
axB = fig.add_subplot(gs[0, 1]); axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
axB.set_title("B  How we measure it — alignment-free (distances only)", fontsize=10, fontweight="bold")
th = np.linspace(0.2*np.pi, 0.8*np.pi, 7); lx, ly = 3 + 2.6*np.cos(th), 3.2 + 3.3*np.sin(th)
axB.plot(lx, ly, "-o", color="#333", ms=7, lw=2, zorder=3)                 # loop arch
fw = np.array([[6.8, 2.0], [7.8, 2.6], [8.6, 1.7], [7.4, 1.1]])            # ultra-rigid framework
axB.scatter(fw[:, 0], fw[:, 1], s=90, marker="s", color="#888", zorder=3)
axB.text(7.7, 0.4, "ultra-rigid\nframework", fontsize=8, ha="center", color="#555")
axB.text(3, 7.2, "CDR loop", fontsize=8, ha="center", color="#333")
for a in range(len(lx)):                                                    # d_LL (deformation)
    for b in range(a+1, len(lx)):
        axB.plot([lx[a], lx[b]], [ly[a], ly[b]], color=R_, lw=0.5, alpha=0.35, zorder=1)
for a in [0, 3, 6]:                                                         # d_LR (rigid pose)
    for f in fw:
        axB.plot([lx[a], f[0]], [ly[a], f[1]], color=B_, lw=0.7, alpha=0.5, zorder=1)
axB.text(0.3, 5.0, "d_LL  (loop–loop)\n→ Var = DEFORMATION\n   (shape, pose-blind)", color=R_, fontsize=9, fontweight="bold", va="center")
axB.text(0.3, 1.8, "d_LR  (loop → framework)\n→ Var = RIGID POSE\n   (SE(3)-invariant, no fit)", color=B_, fontsize=9, fontweight="bold", va="center")

# (C) rigid pose vs deformation per CDR
axC = fig.add_subplot(gs[0, 2]); x = np.arange(6); w = 0.4
axC.bar(x - w/2, [med("eps_pose", c) for c in CDRS], w, color=B_, label="ε_pose  (rigid, d_LR)")
axC.bar(x + w/2, [med("eps_deform", c) for c in CDRS], w, color=R_, label="ε_deform  (d_LL)")
axC.set_xticks(x); axC.set_xticklabels(CDRS, rotation=20, fontsize=8)
axC.set_ylabel("Å (median over 22 TCRs)"); axC.legend(fontsize=8)
axC.set_title("C  Rigid pose vs deformation, per CDR", fontsize=10, fontweight="bold")

# (D) which TCRs more rigid vs deform
axD = fig.add_subplot(gs[1, 0])
mdf = np.array([np.mean([res[t][c]["eps_deform"] for c in CDRS]) for t in tcrs])
mpo = np.array([np.mean([res[t][c]["eps_pose"] for c in CDRS]) for t in tcrs])
ratio = mpo / mdf
axD.scatter(mdf, mpo, s=46, c=ratio, cmap="coolwarm_r", edgecolor="k", lw=0.4, zorder=3)
order = np.argsort(ratio)
for i in list(order[:3]) + list(order[-3:]):                               # label extremes
    axD.annotate(tcrs[i], (mdf[i], mpo[i]), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
lims = [min(mdf.min(), mpo.min())*0.9, max(mdf.max(), mpo.max())*1.05]
axD.plot(lims, lims, "0.7", ls="--", lw=1)
axD.set_xlabel("deformation  ε_deform (Å)"); axD.set_ylabel("rigid pose  ε_pose (Å)")
axD.set_title("D  Which TCRs are more rigid vs deforming\n(blue = rigid-dominated, red = deform-dominated)", fontsize=10, fontweight="bold")

# (E) twist honesty caveat
axE = fig.add_subplot(gs[1, 1])
axE.bar(x - w/2, [med("twist_rigid", c) for c in CDRS], w, color="#7B4FA3", label="RIGID twist (coherent)")
axE.bar(x + w/2, [med("twist_deform", c) for c in CDRS], w, color=R_, label="DEFORM twist (corkscrew)")
axE.set_xticks(x); axE.set_xticklabels(CDRS, rotation=20, fontsize=8)
axE.set_ylabel("twist amplitude (deg, median)"); axE.legend(fontsize=8)
axE.set_title("E  The 'twist' is mostly deformation, not rigid\n(corkscrew > coherent for CDR2/CDR3)", fontsize=10, fontweight="bold")

# (F) method -> conclusion
axF = fig.add_subplot(gs[1, 2]); axF.axis("off")
rk = [tcrs[i] for i in order]
txt = (
 "F   Method → conclusion\n\n"
 "MEASURE (no superposition; CA–CA distances,\n"
 "ultra-rigid framework as reference):\n"
 "  • d_LL variance  = DEFORMATION (shape change)\n"
 "  • d_LR variance  = RIGID POSE vs the framework\n"
 "  • rigid axes ĉ/ĥ/n̂ from the loop's own frame;\n"
 "    multilateration reproduces the typing (r≈0.98)\n"
 "    → typing is intrinsic to the distances.\n\n"
 "CONCLUDE:\n"
 "  • CDR1 = rigid HINGE (tip-elevation, ĉ); least deform\n"
 "  • CDR3 = DEFORMATION-heavy; its 'twist' is mostly\n"
 "    corkscrew (shape change), not rigid rotation\n"
 f"  • most RIGID-dominated TCRs: {', '.join(rk[-3:])}\n"
 f"  • most DEFORM-dominated TCRs: {', '.join(rk[:3])}\n"
 "  (rigid/deform ranked by ε_pose/ε_deform)"
)
axF.text(0.0, 1.0, txt, fontsize=9.2, va="top", family="monospace")

fig.suptitle("Alignment-free decomposition of TCR CDR-loop motion — rigid pose vs deformation (22 unbound TCRs)",
             y=0.99, fontsize=14, fontweight="bold")
out = f"{HERE}/figures/SUMMARY_alignfree.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("fig ->", out)
print("rigid-dominated (high ε_pose/ε_deform):", [rk[-1], rk[-2], rk[-3]])
print("deform-dominated:", [rk[0], rk[1], rk[2]])
