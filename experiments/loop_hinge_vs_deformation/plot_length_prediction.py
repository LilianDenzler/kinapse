#!/usr/bin/env python
"""Does loop length predict hinge extent? Can we predict hinge from the TCR?
- correlations of length vs hinge/deform metrics: pooled AND within-CDR (partialling out CDR identity)
- OLS R^2 for predicting each metric from [length] and [length + CDR identity]
- fig6: length vs theta / D_framework / D_deform, coloured by CDR, with fits.
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
from scipy import stats
import config as C

FIG = os.path.join(C.ROOT, "figures")
CDRS = C.CDRS
COL = {"A_CDR1": "#4C72B0", "A_CDR2": "#5A8FBF", "A_CDR3": "#2E5A88",
       "B_CDR1": "#C44E52", "B_CDR2": "#D07A6B", "B_CDR3": "#8C2D2F"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 130, "axes.axisbelow": True})

m = pd.read_csv(f"{C.RESULTS}/master_percdr.csv")
m = m[m.system != "1KGC"].copy()                        # drop the 10x breadth outlier


def within_cdr_resid(df, col):
    """value minus its per-CDR mean -> removes the universal CDR effect (partial correlation)."""
    return df[col] - df.groupby("cdr")[col].transform("mean")


def ols_r2(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


# ---------- correlations ---------------------------------------------------
TARGETS = ["theta", "theta_p95", "D_framework", "total_disp", "D_deform", "hinge_frac"]
print("length vs metric  (n=%d, 1KGC excluded)" % len(m))
print(f"{'metric':12s} {'pooled r':>9} {'rho':>6} {'p':>9} | {'within-CDR r':>12} {'p':>9}")
rows = []
for t in TARGETS:
    r, p = stats.pearsonr(m.length, m[t])
    rho, _ = stats.spearmanr(m.length, m[t])
    rw, pw = stats.pearsonr(within_cdr_resid(m, "length"), within_cdr_resid(m, t))
    print(f"{t:12s} {r:9.2f} {rho:6.2f} {p:9.1e} | {rw:12.2f} {pw:9.1e}")
    rows.append(dict(metric=t, pooled_r=round(r, 2), spearman=round(rho, 2),
                     within_cdr_r=round(rw, 2), within_cdr_p=pw))
pd.DataFrame(rows).to_csv(f"{C.RESULTS}/length_correlations.csv", index=False)

# CDR3-only length effect (as in MD_flexibility)
print("\nCDR3-only (A_CDR3 + B_CDR3) length vs metric:")
c3 = m[m.cdr.isin(["A_CDR3", "B_CDR3"])]
for t in TARGETS:
    r, p = stats.pearsonr(c3.length, c3[t])
    print(f"  {t:12s} r={r:5.2f}  p={p:.1e}")

# ---------- predictability: OLS R^2 ---------------------------------------
print("\npredictive R^2  (how well can we predict the metric?)")
print(f"{'metric':12s} {'~length':>9} {'~length+CDR':>12} {'~CDR only':>10}")
cdr_dum = pd.get_dummies(m.cdr, drop_first=True).values.astype(float)
for t in TARGETS:
    y = m[t].values
    r2_len = ols_r2(y, m[["length"]].values.astype(float))
    r2_both = ols_r2(y, np.column_stack([m.length.values.astype(float), cdr_dum]))
    r2_cdr = ols_r2(y, cdr_dum)
    print(f"{t:12s} {r2_len:9.2f} {r2_both:12.2f} {r2_cdr:10.2f}")

# ---------- fig6 -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, t, lab in zip(axes, ["theta", "D_framework", "D_deform"],
                      ["hinge ANGLE θ (deg)  — scale-free",
                       "hinge DISPLACEMENT $D_{framework}$ (Å)",
                       "deformation $D_{deform}$ (Å)"]):
    for cdr in CDRS:
        d = m[m.cdr == cdr]
        ax.scatter(d.length, d[t], s=26, c=COL[cdr], label=cdr, alpha=.85, edgecolor="none")
    r, p = stats.pearsonr(m.length, m[t])
    rw, _ = stats.pearsonr(within_cdr_resid(m, "length"), within_cdr_resid(m, t))
    b = np.polyfit(m.length, m[t], 1)
    xs = np.array([m.length.min(), m.length.max()])
    ax.plot(xs, np.polyval(b, xs), "k--", lw=1.2, alpha=.7)
    ax.set_xlabel("loop length (# Cα)"); ax.set_ylabel(lab)
    ax.set_title(f"{lab.split('(')[0].strip()}\npooled r={r:.2f}   within-CDR r={rw:.2f}", fontsize=9)
    ax.grid(alpha=.25)
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("Loop length vs. hinge extent and deformation (21 TCRs, 1KGC excluded)", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig6_length_vs_hinge.png", bbox_inches="tight"); plt.close(fig)
print(f"\nfig -> {FIG}/fig6_length_vs_hinge.png")
