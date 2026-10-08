#!/usr/bin/env python
"""How well can we predict an UNSEEN TCR's total/deform/hinge for each of the 6 CDRs, using the
other TCRs? Leave-one-TCR-out. Two legitimate predictors -- both are fair generalization:
  (M0) the CDR average of the training TCRs      (no length used)
  (M1) a linear fit in CDR length                (uses length if it helps)
Headline metric = held-out RMSE in physical units (Å or deg) + relative error (RMSE / mean value).
Length is only *useful* where M1's error < M0's error. (Q2 = 1 - RMSE_M1^2/RMSE_M0^2 is the
secondary 'does length beat the average' view, kept in the CSV.) Run after plot_hinge_deform.py."""
from __future__ import annotations
import os, sys, warnings
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

FIG = os.path.join(C.ROOT, "figures")
CDRS = C.CDRS
plt.rcParams.update({"font.size": 9, "figure.dpi": 130, "axes.axisbelow": True})
C_TOT, C_DEF, C_ANG = "#4C4C4C", "#DD8452", "#4C72B0"
RNG = np.random.default_rng(42)

m = pd.read_csv(f"{C.RESULTS}/master_percdr.csv")
m = m[m.system != "1KGC"].copy()                         # 10x breadth outlier
m["hinge_rmsd"] = np.sqrt(np.clip(m.total_disp**2 - m.E_nonrigid**2, 0, None))

# quantities we want to predict for an unseen TCR
TARGETS = [("total", "total_disp", C_TOT, "total flexibility", "Å"),
           ("deform", "D_deform", C_DEF, "deformation", "Å"),
           ("hinge", "theta", C_ANG, "hinge angle", "°")]


def loo_mean_rmse(y):
    """held-out RMSE of predicting each TCR by the mean of the OTHER TCRs (M0)."""
    n = len(y); pred = (y.sum() - y) / (n - 1)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def loo_len_rmse(x, y):
    """held-out RMSE of predicting from a length fit on the other TCRs (M1). None if length constant."""
    x = np.asarray(x, float); n = len(x)
    if np.unique(x).size < 2:
        return None
    pred = np.array([np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i]) for i in range(n)])
    return float(np.sqrt(np.mean((y - pred) ** 2)))


# ---------------- per-CDR predictability ----------------------------------
rows = []
for cdr in CDRS:
    d = m[m.cdr == cdr]
    for tag, met, col, name, unit in TARGETS:
        y = d[met].values.astype(float); x = d.length.values
        mean_y = float(y.mean())
        rmse_m0 = loo_mean_rmse(y)
        rmse_m1 = loo_len_rmse(x, y)
        best = rmse_m0 if (rmse_m1 is None or rmse_m1 >= rmse_m0) else rmse_m1
        length_helps = (rmse_m1 is not None) and (rmse_m1 < rmse_m0)
        rows.append(dict(cdr=cdr, metric=tag, unit=unit, n=len(d),
                         len_min=int(d.length.min()), len_max=int(d.length.max()),
                         mean=round(mean_y, 3), sd=round(float(y.std()), 3),
                         rmse_mean=round(rmse_m0, 3),
                         rmse_len=None if rmse_m1 is None else round(rmse_m1, 3),
                         best_rmse=round(best, 3), rel_err=round(best / mean_y, 3),
                         length_helps=length_helps,
                         Q2_len=None if rmse_m1 is None else round(1 - (rmse_m1 / rmse_m0) ** 2, 3)))
pred = pd.DataFrame(rows)
pred.to_csv(f"{C.RESULTS}/predictability_unseen_tcr.csv", index=False)

print("HELD-OUT PREDICTABILITY of an unseen TCR (leave-one-TCR-out), per CDR:")
print("  best_rmse = error using CDR-average, or length if it helps;  rel_err = best_rmse / mean value\n")
for tag, met, col, name, unit in TARGETS:
    sub = pred[pred.metric == tag].set_index("cdr").reindex(CDRS)
    print(f"{name:16s} ({unit}):")
    for cdr in CDRS:
        r = sub.loc[cdr]
        flag = "  <- length helps" if r.length_helps else ""
        print(f"   {cdr}: mean={r['mean']:6.3f}  held-out RMSE={r.best_rmse:6.3f}{unit}"
              f"  ({100*r.rel_err:4.0f}% rel err){flag}")
    print()

# ---------------- figure ---------------------------------------------------
# Native units -- total & deform are in A (share a panel), the hinge ANGLE is in degrees (own panel).
# Each: solid bar = held-out RMSE using length (falls back to the average where length is useless);
#       dashed outline = average-only RMSE -> the gap above the solid bar is what knowing length buys.
fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), gridspec_kw=dict(width_ratios=[1.5, 1.1, 1.0]))
x = np.arange(len(CDRS))


def bars(ax, targets, w, unit_label, title):
    off = np.linspace(-(len(targets)-1)/2, (len(targets)-1)/2, len(targets)) * w
    for (tag, met, col, name, unit), o in zip(targets, off):
        sub = pred[pred.metric == tag].set_index("cdr").reindex(CDRS)
        ax.bar(x + o, sub.rmse_mean.values, w, facecolor="none", edgecolor=col, lw=1.3, ls=(0, (3, 2)))
        ax.bar(x + o, sub.best_rmse.values, w, color=col, label=name)
        for i, c in enumerate(CDRS):                                   # % of the mean, for in-unit context
            ax.text(i + o, sub.loc[c, "best_rmse"] + ax.get_ylim()[1]*0.008,
                    f"{100*sub.loc[c,'rel_err']:.0f}%", ha="center", va="bottom", fontsize=6, color="0.3")
    ax.set_xticks(x); ax.set_xticklabels(CDRS, rotation=20)
    ax.set_ylabel(f"held-out RMSE ({unit_label})"); ax.set_title(title)
    ax.grid(axis="y", alpha=.25)


# A: the two Angstrom quantities
axA = axes[0]
bars(axA, [TARGETS[0], TARGETS[1]], 0.36, "Å", "A · total flexibility & deformation (Å)\n"
     "solid = with length · dashed = average only · %=RMSE/mean")
h = axA.plot([], [], color="0.4", lw=1.3, ls=(0, (3, 2)))[0]
leg1 = axA.legend(fontsize=8, title="predict:", loc="upper right")
axA.legend([h], ["average only (no length)"], fontsize=7.5, loc="upper left"); axA.add_artist(leg1)

# B: the hinge angle, in its own unit (degrees)
axB = axes[1]
bars(axB, [TARGETS[2]], 0.55, "degrees", "B · hinge angle (°)\nseparate axis — not comparable to Å")
axB.legend(fontsize=8, loc="upper left")

# C: predicted vs actual for deformation, CDR3 (clearest place length helps)
axC = axes[2]
for cdr, mk in [("A_CDR3", "o"), ("B_CDR3", "s")]:
    d = m[m.cdr == cdr]; xx = d.length.values; yy = d.D_deform.values
    p = np.array([np.polyval(np.polyfit(np.delete(xx, i), np.delete(yy, i), 1), xx[i]) for i in range(len(xx))])
    axC.scatter(yy, p, marker=mk, s=34, color=C_DEF, alpha=.85,
                label=f"{cdr}  {100*loo_len_rmse(xx,yy)/yy.mean():.0f}% err")
lim = np.array(axC.get_xlim()); axC.plot(lim, lim, "k--", lw=1, alpha=.6)
axC.set_xlabel("actual deformation (Å)"); axC.set_ylabel("predicted from length (held-out)")
axC.set_title("C · held-out CDR3 deformation\n(length prediction)"); axC.legend(fontsize=7); axC.grid(alpha=.25)

fig.suptitle("Predicting an unseen TCR per CDR (leave-one-TCR-out, native units): length lowers the held-out "
             "error mainly for deformation; %=RMSE/mean is the only cross-quantity comparison", fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig8_cv_generalizability.png", bbox_inches="tight"); plt.close(fig)
print(f"fig -> {FIG}/fig8_cv_generalizability.png")
