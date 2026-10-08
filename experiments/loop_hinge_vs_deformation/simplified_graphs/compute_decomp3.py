#!/usr/bin/env python
"""Batch the 3-way motion split (hinge / off-axis rigid / deformation) across all TCRs; save + plot."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob, traceback
import numpy as np
from graph_build import load_md, CDR_RANGES, HERE
from decompose3 import decompose, CDRS

tcrs = [os.path.basename(f)[:4] for f in sorted(glob.glob(f"{HERE}/results_swing/*.npz"))]
rig = json.load(open(f"{HERE}/rigid_framework.json"))
res = {}
for t in tcrs:
    try:
        tv, xyz, imap = load_md(t)
        row = {}
        for cdr in CDRS:
            ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
            lk = sorted(k for k in imap[ch] if lo <= k <= hi)
            loop = xyz[:, np.array([imap[ch][k] for k in lk])]
            R = xyz[:, np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])]
            fh, fo, fd, dt = decompose(loop, R)
            row[cdr] = [fh, fo, fd, dt]
        res[t] = row
        print(f"{t} ok", flush=True)
    except Exception as e:
        print(f"{t} ERR {type(e).__name__}: {str(e)[:100]}", flush=True)
json.dump(res, open(f"{HERE}/results_decomp3.json", "w"))
print(f"\n{len(res)} TCRs -> results_decomp3.json")

# summary table (medians across TCRs)
print(f"\n{'CDR':8}{'hinge':>8}{'off-axis':>10}{'deform':>8}{'D_tot':>8}   (median over TCRs)")
M = {}
for cdr in CDRS:
    A = np.array([res[t][cdr] for t in res])
    m = np.median(A, 0); M[cdr] = A
    print(f"{cdr:8}{m[0]:7.0%}{m[1]:9.0%}{m[2]:7.0%}{m[3]:8.2f}")

# plot: stacked medians + per-TCR deform-fraction spread
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"hinge": "#2E7D32", "off": "#F0A030", "deform": "#C0392B"}
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.1, 1.3]})
x = np.arange(6)
med = np.array([np.median(M[c], 0)[:3] for c in CDRS])       # (6,3) hinge/off/deform
ax[0].bar(x, med[:, 0], color=COL["hinge"], label="hinge (along axis)")
ax[0].bar(x, med[:, 1], bottom=med[:, 0], color=COL["off"], label="off-axis rigid")
ax[0].bar(x, med[:, 2], bottom=med[:, 0] + med[:, 1], color=COL["deform"], label="deformation")
ax[0].set_xticks(x); ax[0].set_xticklabels(CDRS, rotation=20, fontsize=9)
ax[0].set_ylabel("fraction of total motion (median)"); ax[0].set_ylim(0, 1)
ax[0].axhline(0, color="k", lw=.5); ax[0].legend(fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(.5, 1.01))
ax[0].set_title("Median 3-way split per CDR", y=1.08)

rng = np.random.default_rng(0)
for j, c in enumerate(CDRS):
    df = M[c][:, 2]; rg = M[c][:, 0] + M[c][:, 1]     # deform frac ; rigid frac (hinge+off)
    ax[1].scatter(np.full(len(df), j) + rng.uniform(-.16, .16, len(df)), df, c=COL["deform"], s=26, edgecolor="k", linewidths=.3, zorder=3)
bp = ax[1].boxplot([M[c][:, 2] for c in CDRS], positions=x, widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.4))
for p in bp["boxes"]:
    p.set_facecolor(COL["deform"]); p.set_alpha(.18)
ax[1].axhline(0.5, color="0.6", lw=.8, ls="--"); ax[1].text(5.6, 0.51, "half deform", fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels(CDRS, rotation=20, fontsize=9)
ax[1].set_ylabel("deformation fraction"); ax[1].set_ylim(0, 1)
ax[1].set_title("Deformation fraction per CDR across 22 TCRs")
fig.suptitle("How much of each CDR's motion is rigid hinge vs off-axis rigid vs deformation", y=1.04, fontsize=13)
out = f"{HERE}/figures/decomp3_spread.png"
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("fig ->", out)
print("median rigid (hinge+off) frac:", {c: round(float(np.median(M[c][:, 0] + M[c][:, 1])), 2) for c in CDRS})
