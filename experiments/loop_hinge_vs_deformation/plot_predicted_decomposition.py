#!/usr/bin/env python
"""Predicting the hinge & deformation PARTS of an unseen TCR's total flexibility (leave-one-TCR-out).
total flexibility (A) splits additively into c_deform + c_hinge (variance shares of total^2 = hinge^2 + deform^2).
For each part and for total, we predict a held-out TCR two ways and compare the held-out error (RMSE, A):
   - CDR average of the other TCRs  (no length used)
   - a linear fit in CDR length     (length used)
One panel per quantity (deform / hinge / total); each: per-CDR bars, average-only vs +length.
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
C_GREY = "#B8B8B8"                                    # no-length (CDR average)

m = pd.read_csv(f"{C.RESULTS}/master_percdr.csv")
m = m[m.system != "1KGC"].copy()
m["hinge_rmsd"] = np.sqrt(np.clip(m.total_disp**2 - m.E_nonrigid**2, 0, None))
m["c_deform"] = m.E_nonrigid**2 / m.total_disp        # parts that ADD to total (A)
m["c_hinge"] = m.hinge_rmsd**2 / m.total_disp

# quantity, column, colour-when-length-used, title
QTY = [("deformation part", "c_deform", "#DD8452"),
       ("hinge part", "c_hinge", "#4C72B0"),
       ("total flexibility", "total_disp", "#4C4C4C")]


def loo_rmse(x, y, model):
    """held-out RMSE (A) predicting each TCR from the others: model='mean' or 'length'."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(y)
    if model == "mean":
        pred = (y.sum() - y) / (n - 1)
    else:
        if np.unique(x).size < 2:
            return np.nan
        pred = np.array([np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i]) for i in range(n)])
    return float(np.sqrt(np.mean((y - pred) ** 2)))


rows = []
for cdr in CDRS:
    d = m[m.cdr == cdr]; x = d.length.values
    for qname, col, _ in QTY:
        rows.append(dict(cdr=cdr, quantity=qname,
                         rmse_noLength=loo_rmse(x, d[col].values, "mean"),
                         rmse_withLength=loo_rmse(x, d[col].values, "length"),
                         mean=float(d[col].mean())))
r = pd.DataFrame(rows)
r["helps"] = r.rmse_withLength < r.rmse_noLength
r.round(3).to_csv(f"{C.RESULTS}/predicted_decomposition.csv", index=False)

fig, ax = plt.subplots(figsize=(13.5, 5.6))
x = np.arange(len(CDRS)); w = 0.27
for j, (qname, col, ccol) in enumerate(QTY):
    sub = r[r.quantity == qname].set_index("cdr").reindex(CDRS)
    off = (j - 1) * w
    ax.bar(x + off, sub.rmse_noLength, w, facecolor="none", edgecolor=ccol, lw=1.3, ls=(0, (3, 2)))  # no length
    ax.bar(x + off, sub.rmse_withLength, w, color=ccol, label=qname)                                 # with length
    for i, c in enumerate(CDRS):
        if sub.loc[c, "helps"]:                       # length lowers the error -> gap above the solid bar
            ax.annotate("", xy=(i + off, sub.loc[c, "rmse_withLength"]),
                        xytext=(i + off, sub.loc[c, "rmse_noLength"]),
                        arrowprops=dict(arrowstyle="->", color="green", lw=1.3))
ax.set_xticks(x); ax.set_xticklabels(CDRS, rotation=20)
ax.set_ylabel("held-out prediction error  RMSE (Å)  —  lower = better")
ax.set_title("Predicting an unseen TCR per CDR: deformation vs hinge vs total, on one Å axis\n"
             "solid = with length · dashed outline = CDR average (no length) · green ↓ = length lowers the error")
hleg = ax.legend(fontsize=8.5, title="held-out error of:", loc="upper left")
proxy = ax.plot([], [], color="0.4", lw=1.3, ls=(0, (3, 2)))[0]
ax.legend([proxy], ["without length (average)"], fontsize=8, loc="upper center"); ax.add_artist(hleg)
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(f"{FIG}/fig10_predicted_decomposition.png", bbox_inches="tight"); plt.close(fig)

print("held-out RMSE (Å): no-length (CDR average)  vs  +length")
for qname, col, _ in QTY:
    sub = r[r.quantity == qname].set_index("cdr").reindex(CDRS)
    print(f"\n{qname}:")
    for c in CDRS:
        s = sub.loc[c]
        print(f"   {c}: no-length {s.rmse_noLength:.3f}  +length {s.rmse_withLength:.3f}"
              f"   {'<- length helps' if s.helps else ''}")
print(f"\nfig -> {FIG}/fig10_predicted_decomposition.png")
