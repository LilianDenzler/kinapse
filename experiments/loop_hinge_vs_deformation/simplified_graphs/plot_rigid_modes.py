#!/usr/bin/env python
"""RIGID motion of the loop, separated from DEFORMATION — the clean version.
Rigid motion = the Kabsch best-rigid-body fit (deformation is its orthogonal residual, so it is EXCLUDED).
LEFT : the rigid ROTATION broken into its modes — hinge(ĉ) / twist(ĥ) / sway(n̂) angular amplitude (deg).
       These come from the rigid rotation vector only (no translation carving, no lever-arm/Å artifact).
RIGHT: magnitude check — rigid motion vs deformation, both as per-atom RMS (Å), deformation EXCLUDED from rigid.
Reuses results_rotframe.json (rotation modes) + results_pivot.json (rigid/deform magnitude). -> figures/rigid_modes_vs_deform.png"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from graph_build import HERE

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
rf = json.load(open(f"{HERE}/results_rotframe.json"))
pv = json.load(open(f"{HERE}/results_pivot.json"))
tcrs = [t for t in rf if t in pv]
x = np.arange(6)


def med(src, cdr, key):
    return np.median([src[t][cdr][key] for t in tcrs])


fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.15, 1]})

# LEFT: rigid rotation modes (deg) — the different rigid mode movements, deformation excluded
w = 0.26
for j, (k, col, lab) in enumerate([("amp_c", "#2E7D32", "hinge (ĉ, tip-elev)"), ("amp_h", "#7B4FA3", "twist (ĥ)"), ("amp_n", "#F0A030", "sway (n̂)")]):
    ax[0].bar(x + (j - 1) * w, [med(rf, c, k) for c in CDRS], w, color=col, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels(CDRS, fontsize=10)
ax[0].set_ylabel("rigid rotation amplitude (deg, median over 22 TCRs)")
ax[0].legend(fontsize=9)
ax[0].set_title("a  RIGID rotation modes (deformation excluded)\nfrom the Kabsch rigid rotation vector — no translation/Å carving", loc="left", fontsize=10, fontweight="bold")

# RIGHT: rigid vs deformation magnitude, both Å per-atom RMS, deformation EXCLUDED from rigid
rigid = np.array([np.median([np.sqrt((pv[t][c]["tot"] - pv[t][c]["deform"]) / pv[t][c]["nres"]) for t in tcrs]) for c in CDRS])
deform = np.array([np.median([np.sqrt(pv[t][c]["deform"] / pv[t][c]["nres"]) for t in tcrs]) for c in CDRS])
ax[1].bar(x - 0.2, rigid, 0.4, color="#3B6EA5", label="RIGID motion (deform excluded)")
ax[1].bar(x + 0.2, deform, 0.4, color="#C0392B", label="DEFORMATION")
ax[1].set_xticks(x); ax[1].set_xticklabels(CDRS, fontsize=10)
ax[1].set_ylabel("per-atom RMS displacement (Å, median over 22 TCRs)")
ax[1].legend(fontsize=9)
ax[1].set_title("b  Rigid motion vs deformation (magnitude, Å)\nseparated by Kabsch; deform cross-validated by d_LL (r=0.97)", loc="left", fontsize=10, fontweight="bold")

fig.suptitle("Loop RIGID motion (its modes + magnitude) vs DEFORMATION — separated.  "
             "Rigid = best rigid-body fit; deformation = the residual it can't explain.", y=1.02, fontsize=11)
out = f"{HERE}/figures/rigid_modes_vs_deform.png"
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("fig ->", out)
print(f"\n{'CDR':8}{'hinge°':>7}{'twist°':>7}{'sway°':>7} | {'rigid(Å)':>9}{'deform(Å)':>10}")
for i, c in enumerate(CDRS):
    print(f"{c:8}{med(rf,c,'amp_c'):7.1f}{med(rf,c,'amp_h'):7.1f}{med(rf,c,'amp_n'):7.1f} | {rigid[i]:9.2f}{deform[i]:10.2f}")
