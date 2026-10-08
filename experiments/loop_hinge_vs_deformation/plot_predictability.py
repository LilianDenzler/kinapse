#!/usr/bin/env python
"""How predictable is DEFORMATION vs HINGE from CDR identity + loop length alone?
Left  : per-CDR predictability from length (R^2 of metric~length across the 21 TCRs of that CDR;
        within a CDR, identity is constant so length is the only varying static feature).
Right : pooled feature attribution -- R^2 from {CDR identity only, length only, CDR+length}
        for deformation (D_deform) vs hinge angle (theta).
Run after plot_hinge_deform.py."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

FIG = os.path.join(C.ROOT, "figures")
CDRS = C.CDRS
plt.rcParams.update({"font.size": 9, "figure.dpi": 130, "axes.axisbelow": True})
C_DEF, C_HIN = "#DD8452", "#4C72B0"      # deformation / hinge

m = pd.read_csv(f"{C.RESULTS}/master_percdr.csv")
m = m[m.system != "1KGC"].copy()          # drop 10x breadth outlier


def r2(y, X):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_res = ((y - X @ beta) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    return max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def within_r2(d, metric):
    """R^2 of metric ~ length within one CDR (0 if length has no variation)."""
    if d.length.nunique() < 2:
        return 0.0, False
    return r2(d[metric].values, d[["length"]].values), True


# ---- per-CDR predictability from length -----------------------------------
rows = []
for cdr in CDRS:
    d = m[m.cdr == cdr]
    r2d, okd = within_r2(d, "D_deform")
    r2h, okh = within_r2(d, "theta")
    rows.append(dict(cdr=cdr, n=len(d), len_min=int(d.length.min()), len_max=int(d.length.max()),
                     len_unique=int(d.length.nunique()),
                     r2_deform=round(r2d, 3), r2_hinge_theta=round(r2h, 3),
                     length_varies=okd))
pc = pd.DataFrame(rows)
pc.to_csv(f"{C.RESULTS}/predictability_per_cdr.csv", index=False)
print(pc.to_string(index=False))

# ---- pooled feature attribution -------------------------------------------
cdr_oh = pd.get_dummies(m.cdr, drop_first=True).values.astype(float)
length = m[["length"]].values.astype(float)
both = np.column_stack([length, cdr_oh])
pooled = {}
for name, met in [("Deformation\n$D_{deform}$", "D_deform"), ("Hinge angle\n" + r"$\theta$", "theta")]:
    y = m[met].values
    pooled[name] = dict(cdr=r2(y, cdr_oh), length=r2(y, length), both=r2(y, both))
print("\npooled R^2 (feature attribution):")
for k, v in pooled.items():
    print(f"  {k.splitlines()[0]:12s}  CDR-only={v['cdr']:.2f}  length-only={v['length']:.2f}  CDR+length={v['both']:.2f}")

# ---- figure ---------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))

x = np.arange(len(CDRS)); w = 0.38
axL.bar(x - w/2, pc.r2_deform, w, color=C_DEF, label="deformation ($D_{deform}$)")
axL.bar(x + w/2, pc.r2_hinge_theta, w, color=C_HIN, label=r"hinge angle ($\theta$)")
for i, r in pc.iterrows():
    tag = f"len {r.len_min}" if r.len_min == r.len_max else f"len {r.len_min}–{r.len_max}"
    axL.text(i, -0.045, tag, ha="center", va="top", fontsize=7, color="0.35")
    if not r.length_varies:
        axL.text(i, 0.02, "no len\nvariation", ha="center", fontsize=6.5, color="0.5")
axL.set_xticks(x); axL.set_xticklabels(CDRS, rotation=20)
axL.set_ylabel("$R^2$  (predict from loop length, within CDR)")
axL.set_ylim(-0.08, 1.0)
axL.set_title("A · per-CDR predictability from length\n(across the 21 TCRs of each CDR)")
axL.legend(fontsize=8, loc="upper left"); axL.grid(axis="y", alpha=.25)
axL.axhline(0, color="k", lw=.6)

groups = list(pooled); gx = np.arange(len(groups)); bw = 0.26
feats = [("CDR identity", "cdr", "#8C8C8C"), ("length", "#B0B0B0", None)]
labels = [("CDR identity", "cdr", "#9C6B4F"), ("length", "length", "#6B8CBF"), ("CDR + length", "both", "#2E5A88")]
for j, (lab, key, col) in enumerate(labels):
    vals = [pooled[g][key] for g in groups]
    axR.bar(gx + (j-1)*bw, vals, bw, color=col, label=lab)
    for gi, v in enumerate(vals):
        axR.text(gx[gi] + (j-1)*bw, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
axR.set_xticks(gx); axR.set_xticklabels(groups)
axR.set_ylabel("$R^2$  (pooled, all 126 loops)")
axR.set_ylim(0, 0.65)
axR.set_title("B · pooled: what carries the signal?\nCDR identity vs length vs both")
axR.legend(fontsize=8, loc="upper right"); axR.grid(axis="y", alpha=.25)

fig.suptitle("Deformation is predictable from CDR identity + length; the hinge angle is not",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(f"{FIG}/fig7_predictability.png", bbox_inches="tight"); plt.close(fig)
print(f"\nfig -> {FIG}/fig7_predictability.png")
