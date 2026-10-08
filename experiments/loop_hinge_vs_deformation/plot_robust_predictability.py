#!/usr/bin/env python
"""Robust predictability: RMSE (dominated by a couple of outlier TCRs like 3SKN/4JFH in B_CDR2) vs
the MEDIAN absolute held-out error (reflects the TYPICAL TCR). Leave-one-TCR-out, CDR-average predictor.
Shows B_CDR2 is well-behaved for the typical TCR; its big RMSE is two tails, not baseline flexibility."""
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

m = pd.read_csv(f"{C.RESULTS}/master_percdr.csv")
m = m[m.system != "1KGC"].copy()
m["hinge_rmsd"] = np.sqrt(np.clip(m.total_disp**2 - m.E_nonrigid**2, 0, None))
m["c_deform"] = m.E_nonrigid**2 / m.total_disp
m["c_hinge"] = m.hinge_rmsd**2 / m.total_disp
QTY = [("deformation part", "c_deform", "#DD8452"),
       ("hinge part", "c_hinge", "#4C72B0"),
       ("total flexibility", "total_disp", "#4C4C4C")]


def loo_errors(y):
    n = len(y); pred = (y.sum() - y) / (n - 1)     # leave-one-TCR-out CDR average
    return np.abs(y - pred)


rows = []
for cdr in CDRS:
    d = m[m.cdr == cdr]
    for qname, col, _ in QTY:
        e = loo_errors(d[col].values)
        rows.append(dict(cdr=cdr, quantity=qname,
                         rmse=float(np.sqrt(np.mean(e**2))), medae=float(np.median(e)),
                         mean=float(d[col].mean())))
r = pd.DataFrame(rows)
r.round(3).to_csv(f"{C.RESULTS}/robust_predictability.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(len(CDRS)); w = 0.38
for ax, (qname, col, ccol) in zip(axes, QTY):
    sub = r[r.quantity == qname].set_index("cdr").reindex(CDRS)
    ax.bar(x - w/2, sub.rmse, w, color="#C0C0C0", label="RMSE (outlier-sensitive)")
    ax.bar(x + w/2, sub.medae, w, color=ccol, label="median |error| (robust)")
    ax.set_xticks(x); ax.set_xticklabels(CDRS, rotation=20)
    ax.set_ylabel("held-out prediction error (Å)")
    ax.set_title(qname)
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(axis="y", alpha=.25)
fig.suptitle("Robust vs outlier-sensitive predictability (leave-one-TCR-out): B_CDR2's big RMSE is 2 tails "
             "(3SKN, 4JFH) — its TYPICAL error (median) is small", fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig11_robust_predictability.png", bbox_inches="tight"); plt.close(fig)

print("held-out error (Å): RMSE (all 21)  vs  median|error| (typical TCR)")
for qname, col, _ in QTY:
    sub = r[r.quantity == qname].set_index("cdr").reindex(CDRS)
    print(f"\n{qname}:")
    for c in CDRS:
        print(f"   {c}: RMSE {sub.loc[c,'rmse']:.3f}   median {sub.loc[c,'medae']:.3f}"
              f"   ({'OUTLIER-INFLATED' if sub.loc[c,'rmse']>2.2*sub.loc[c,'medae'] else 'consistent'})")
print(f"\nfig -> {FIG}/fig11_robust_predictability.png")
