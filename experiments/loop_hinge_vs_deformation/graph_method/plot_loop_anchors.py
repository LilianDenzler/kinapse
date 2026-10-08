#!/usr/bin/env python
"""Per-CDR LOOP-ANCHOR stability graphs (alignment-free, pairwise Ca distances).
Anchors per CDR: CDR1=26/39, CDR2=55/66, CDR3=104/118 (all consensus rigid-framework residues).

fig1 (each CDR's two anchors):
   A: within-MD  -> std over frames of the anchor-anchor distance, per CDR, one point per TCR  (do the 2 anchors of a loop move apart during one MD?)
   B: across-TCR -> the anchor-anchor distance itself, per CDR, one point per TCR               (are the 2 anchors placed the same in every TCR?)
fig2 (the three anchor pairs vs each other, per chain, via pair MIDPOINTS):
   A: within-MD  -> std over frames of the 3 midpoint-midpoint distances                        (do the 3 loop bases move relative to each other during one MD?)
   B: across-TCR -> those midpoint-midpoint distances                                            (is the arrangement of the 3 loop bases conserved across TCRs?)"""
from __future__ import annotations
import os, sys, glob
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__)); RES = f"{ROOT}/results_loopanchor"; FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 130, "axes.axisbelow": True})
PAIRS = {"CDR1": (26, 39), "CDR2": (55, 66), "CDR3": (104, 118)}
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
COL = {"A_CDR1": "#4C72B0", "A_CDR2": "#5A8FBF", "A_CDR3": "#2E5A88",
       "B_CDR1": "#C44E52", "B_CDR2": "#D07A6B", "B_CDR3": "#8C2D2F"}

data = {os.path.basename(f)[:4]: dict(np.load(f, allow_pickle=True)) for f in sorted(glob.glob(f"{RES}/*.npz"))}
tcrs = list(data)
print(f"{len(tcrs)} TCRs")
rng = np.random.default_rng(0)


def strip(ax, xi, vals, col, w=0.11):
    xj = xi + rng.uniform(-w, w, len(vals))
    ax.scatter(xj, vals, s=16, c=col, alpha=.8, edgecolor="none")
    ax.hlines(np.median(vals), xi - 0.28, xi + 0.28, color="k", lw=2, zorder=5)


# ---------- fig1: each CDR's two anchors ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5))
for k, cdr in enumerate(CDRS):
    ch = cdr[0]; n, c = PAIRS[cdr[2:]]
    within, means = [], []
    for s in tcrs:
        z = data[s]; im = z.get(f"{ch}_imgt")
        if im is None:
            continue
        im = im.tolist()
        if n in im and c in im:
            i, j = im.index(n), im.index(c)
            within.append(float(z[f"{ch}_std"][i, j])); means.append(float(z[f"{ch}_mean"][i, j]))
    strip(axA, k, within, COL[cdr]); strip(axB, k, means, COL[cdr])
    cv = 100 * np.std(means) / np.mean(means)
    axB.text(k, max(means) + 0.15, f"CV\n{cv:.1f}%", ha="center", va="bottom", fontsize=7, color="0.3")
for ax, ttl, yl in [(axA, "A · within one MD: do the two anchors of a loop move apart?\n(std over frames of the anchor–anchor distance)", "anchor–anchor distance std over MD (Å)"),
                    (axB, "B · across TCRs: are the two anchors placed the same?\n(the anchor–anchor distance, one point per TCR)", "anchor–anchor distance (Å)")]:
    ax.set_xticks(range(len(CDRS))); ax.set_xticklabels(CDRS, rotation=20); ax.set_title(ttl); ax.set_ylabel(yl)
axA.set_ylim(0, None)
fig.suptitle("Loop anchors per CDR — CDR1=26/39, CDR2=55/66, CDR3=104/118  (alignment-free)", y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_loopanchor_pair.png", bbox_inches="tight"); plt.close(fig)

# ---------- fig2: the three anchor pairs vs each other (per chain, midpoints) ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5))
MIDPAIRS = [("CDR1", "CDR2"), ("CDR1", "CDR3"), ("CDR2", "CDR3")]
labels = [f"{ch}:{a[-1]}-{b[-1]}" for ch in "AB" for a, b in MIDPAIRS]     # e.g. A:1-2
xk = 0; ticks = []
for ch in "AB":
    for (a, b) in MIDPAIRS:
        within, means = [], []
        for s in tcrs:
            z = data[s]; lab = z.get(f"{ch}_mid_labels")
            if lab is None:
                continue
            lab = lab.tolist()
            if a in lab and b in lab:
                i, j = lab.index(a), lab.index(b)
                within.append(float(z[f"{ch}_mid_std"][i, j])); means.append(float(z[f"{ch}_mid_mean"][i, j]))
        col = "#4C72B0" if ch == "A" else "#C44E52"
        strip(axA, xk, within, col); strip(axB, xk, means, col)
        ticks.append(f"{ch}: {a[-1]}–{b[-1]}"); xk += 1
for ax, ttl, yl in [(axA, "A · within one MD: do the 3 loop bases move relative to each other?\n(std over frames of midpoint–midpoint distance)", "midpoint–midpoint distance std over MD (Å)"),
                    (axB, "B · across TCRs: is the arrangement of the 3 loop bases conserved?\n(midpoint–midpoint distance, one point per TCR)", "midpoint–midpoint distance (Å)")]:
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, rotation=20); ax.set_title(ttl); ax.set_ylabel(yl)
axA.set_ylim(0, None)
fig.suptitle("Relation of the three CDR anchor pairs to each other, per chain  (loop-base midpoints, alignment-free)",
             y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_loopanchor_triad.png", bbox_inches="tight"); plt.close(fig)
print(f"figs -> {FIG}/fig_loopanchor_pair.png , {FIG}/fig_loopanchor_triad.png")
