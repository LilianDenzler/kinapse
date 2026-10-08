#!/usr/bin/env python
"""Typed rigid-vs-deformation classification of each CDR using the loop-frame axes.
(A) scatter: deformation (A) vs rigid rotation (deg), each TCRxCDR coloured by DOMINANT axis
    (green=tip-elevation hinge, orange=in-plane sway, purple=twist); per-CDR medians as big labels.
(B) per CDR, the fraction of the 22 TCRs whose rigid rotation is hinge- / sway- / twist-dominant.
Reads results_rotframe.json -> figures/classify_rigid_deform.png"""
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
RGB = {"c": "#2E7D32", "n": "#F0A030", "h": "#7B4FA3"}
LAB = {"c": "hinge (tip-elev, ĉ)", "n": "sway (n̂)", "h": "twist (ĥ)"}
res = json.load(open(f"{HERE}/results_rotframe.json"))
tcrs = list(res)


def dom(r):
    return max("cnh", key=lambda k: r[f"f_{k}"])


fig, ax = plt.subplots(1, 2, figsize=(14.5, 6), gridspec_kw={"width_ratios": [1.35, 1]})
# (A) rigid rotation vs deformation, coloured by dominant axis
for t in tcrs:
    for cdr in CDRS:
        r = res[t][cdr]
        rot = np.sqrt(r["amp_c"] ** 2 + r["amp_n"] ** 2 + r["amp_h"] ** 2)
        ax[0].scatter(r["deform"], rot, s=22, color=RGB[dom(r)], alpha=0.55, edgecolor="none", zorder=2)
for cdr in CDRS:
    rr = [np.sqrt(res[t][cdr]["amp_c"] ** 2 + res[t][cdr]["amp_n"] ** 2 + res[t][cdr]["amp_h"] ** 2) for t in tcrs]
    dd = [res[t][cdr]["deform"] for t in tcrs]
    mx, my = np.median(dd), np.median(rr)
    ax[0].scatter(mx, my, s=160, marker="D", color="white", edgecolor="k", lw=1.4, zorder=4)
    ax[0].annotate(cdr, (mx, my), fontsize=8.5, fontweight="bold", xytext=(6, 4), textcoords="offset points", zorder=5)
ax[0].axvline(np.median([res[t][c]["deform"] for t in tcrs for c in CDRS]), color="0.7", lw=.7, ls=":")
ax[0].set_xlabel("deformation (Å)   →  shape change"); ax[0].set_ylabel("rigid rotation amplitude (deg)   →  pose change")
ax[0].set_title("a  Typed rigid↔deform map (each dot = TCR×CDR; colour = dominant rigid axis)", loc="left", fontsize=10, fontweight="bold")
hs = [plt.Line2D([], [], marker="o", ls="", color=RGB[k], label=LAB[k]) for k in "cnh"]
hs.append(plt.Line2D([], [], marker="D", ls="", mfc="white", mec="k", label="per-CDR median"))
ax[0].legend(handles=hs, fontsize=8.5, loc="upper right")
# (B) mode-dominance classification per CDR
bot = np.zeros(6)
for k in "cnh":
    frac = [np.mean([dom(res[t][c]) == k for t in tcrs]) for c in CDRS]
    ax[1].bar(range(6), frac, bottom=bot, color=RGB[k], label=LAB[k]); bot += np.array(frac)
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(CDRS, rotation=20, fontsize=9); ax[1].set_ylim(0, 1)
ax[1].set_ylabel("fraction of 22 TCRs"); ax[1].legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3)
ax[1].set_title("b  Which rigid mode dominates, per CDR", loc="left", fontsize=10, fontweight="bold", y=1.06)
fig.suptitle("Classifying each CDR's motion: rigid rotation (typed by axis) vs deformation — 22 TCRs", y=1.02, fontsize=12)
out = f"{HERE}/figures/classify_rigid_deform.png"
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("fig ->", out)
for cdr in CDRS:
    cnt = {k: int(sum(dom(res[t][cdr]) == k for t in tcrs)) for k in "cnh"}
    print(f"  {cdr}: dominant-axis counts (hinge/sway/twist) = {cnt['c']}/{cnt['n']}/{cnt['h']}")
