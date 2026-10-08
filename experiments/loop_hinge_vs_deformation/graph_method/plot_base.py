#!/usr/bin/env python
"""Local-base results: (1) base stability (patch-centroid base width + axis-direction, per CDR);
(2) the single-hinge test: f_fixed_global vs f_local_base vs f_rigid_free per CDR, across TCRs."""
from __future__ import annotations
import os, sys, glob
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__)); RES = f"{ROOT}/results_base"; FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 130, "axes.axisbelow": True})
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/*_base.csv"))], ignore_index=True)
print(f"{m.system.nunique()} TCRs")
rng = np.random.default_rng(0)


def strip(ax, xi, v, col, w=0.1):
    ax.scatter(xi + rng.uniform(-w, w, len(v)), v, s=15, c=col, alpha=.8, edgecolor="none")
    ax.hlines(np.median(v), xi - 0.3, xi + 0.3, color="k", lw=2, zorder=5)


# ---- fig1: base stability (patch centroids) --------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k, cdr in enumerate(CDRS):
    d = m[m.cdr == cdr]
    strip(ax[0], k, d.base_width_std.values, "#4C72B0")
    strip(ax[1], k, d.axis_dir_std_deg.values, "#DD8452")
ax[0].set_ylabel("base width std over MD (Å)"); ax[0].set_title("A · local-base width stability\n(3-residue patch centroids; do the two bases keep a fixed separation?)")
ax[1].set_ylabel("base-axis direction std over MD (deg)"); ax[1].set_title("B · local-base AXIS-direction stability\n(std of angle(u_base(t), mean) in the framework frame)")
for a in ax:
    a.set_xticks(range(len(CDRS))); a.set_xticklabels(CDRS, rotation=20); a.set_ylim(0, None)
fig.suptitle("Local CDR base defined by 3-residue β-strand patch centroids (alignment-free width; framework-frame axis)", y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_base_stability.png", bbox_inches="tight"); plt.close(fig)

# ---- fig2: the single-hinge test ------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.4))
x = np.arange(len(CDRS)); w = 0.26
for j, (col, key, lab) in enumerate([("#B0B0B0", "f_fixed_global", "fixed global axis"),
                                     ("#4C72B0", "f_local_base", "local dynamic base"),
                                     ("#55A868", "f_rigid_free", "any-axis rigid (ceiling)")]):
    med = [m[m.cdr == c][key].median() for c in CDRS]
    for k, c in enumerate(CDRS):
        strip(ax, k + (j - 1) * w, m[m.cdr == c][key].values, col, w=0.07)
    ax.plot(x + (j - 1) * w, med, "_", ms=1)
    ax.bar(x + (j - 1) * w, med, w, color=col, alpha=.35, label=lab, zorder=0)
ax.set_xticks(x); ax.set_xticklabels(CDRS, rotation=20)
ax.set_ylabel("hinge fraction of loop→framework distance variance"); ax.set_ylim(0, 1)
ax.set_title("Single-hinge test per CDR: fixed-global vs local-dynamic-base vs any-axis-rigid\n"
             "local base ≈ fixed global ≪ any-axis rigid  ⇒  the CDR reorientation is multi-axis, not a 1-D hinge")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig_single_hinge_test.png", bbox_inches="tight"); plt.close(fig)

# ---- fig3: UPDATED loop-anchor PAIR plot (patch-centroid base) -------------
COL = {"A_CDR1": "#4C72B0", "A_CDR2": "#5A8FBF", "A_CDR3": "#2E5A88",
       "B_CDR1": "#C44E52", "B_CDR2": "#D07A6B", "B_CDR3": "#8C2D2F"}
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5))
for k, cdr in enumerate(CDRS):
    d = m[m.cdr == cdr]
    strip(axA, k, d.base_width_std.values, COL[cdr])
    strip(axB, k, d.base_width_mean.values, COL[cdr])
    cv = 100 * d.base_width_mean.std() / d.base_width_mean.mean()
    axB.text(k, d.base_width_mean.max() + 0.1, f"CV\n{cv:.1f}%", ha="center", va="bottom", fontsize=7, color="0.3")
axA.set_ylabel("base-width std over MD (Å)"); axA.set_title("A · within one MD: do the loop's two BASES move apart?\n(3-residue patch centroids — tighter than single anchors)"); axA.set_ylim(0, None)
axB.set_ylabel("base width (Å)"); axB.set_title("B · across TCRs: is the base width the same?\n(patch-centroid separation, one point per TCR)")
for a in (axA, axB):
    a.set_xticks(range(len(CDRS))); a.set_xticklabels(CDRS, rotation=20)
fig.suptitle("Loop anchors per CDR — 3-residue β-strand patch centroids (updated from single residues)", y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_loopanchor_pair.png", bbox_inches="tight"); plt.close(fig)

# ---- fig4: UPDATED triad plot (patch-base midpoints' mutual distances) ------
mids = {os.path.basename(f)[:4]: dict(np.load(f, allow_pickle=True)) for f in sorted(glob.glob(f"{RES}/*_mid.npz"))}
MIDP = [("CDR1", "CDR2"), ("CDR1", "CDR3"), ("CDR2", "CDR3")]
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5)); xk = 0; ticks = []
for ch in "AB":
    for a, b in MIDP:
        within, means = [], []
        for s, z in mids.items():
            lab = z.get(f"{ch}_labels")
            if lab is None:
                continue
            lab = [str(x) for x in lab]
            if a in lab and b in lab:
                i, j = lab.index(a), lab.index(b)
                within.append(float(z[f"{ch}_mid_std"][i, j])); means.append(float(z[f"{ch}_mid_mean"][i, j]))
        col = "#4C72B0" if ch == "A" else "#C44E52"
        strip(axA, xk, within, col); strip(axB, xk, means, col)
        ticks.append(f"{ch}: {a[-1]}–{b[-1]}"); xk += 1
axA.set_ylabel("base-midpoint distance std over MD (Å)"); axA.set_title("A · within one MD: do the 3 loop BASES move relative to each other?\n(patch-centroid midpoints)"); axA.set_ylim(0, None)
axB.set_ylabel("base-midpoint distance (Å)"); axB.set_title("B · across TCRs: is the 3-base arrangement conserved?")
for a in (axA, axB):
    a.set_xticks(range(len(ticks))); a.set_xticklabels(ticks, rotation=20)
fig.suptitle("Relation of the three CDR loop bases (patch-centroid midpoints, per chain)", y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_loopanchor_triad.png", bbox_inches="tight"); plt.close(fig)

print("SUMMARY (median over TCRs):")
g = m.groupby("cdr")[["base_width_std", "axis_dir_std_deg", "f_fixed_global", "f_local_base", "f_rigid_free"]].median().reindex(CDRS)
print(g.round(3).to_string())
print(f"\nfigs -> {FIG}/fig_base_stability.png , {FIG}/fig_single_hinge_test.png")
