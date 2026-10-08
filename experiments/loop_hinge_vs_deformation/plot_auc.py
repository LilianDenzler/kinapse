#!/usr/bin/env python
"""'AUC' for the (regression) predictability graphs = concordance index (C-index): the probability
that, given two TCRs, the length-based prediction ranks their loop flexibility in the correct order.
This is ROC-AUC generalized to a continuous outcome (0.5 = random, 1 = perfect) and is rank-based, so
it is OUTLIER-ROBUST (3SKN/4JFH big swings just rank high). Leave-one-TCR-out length predictions.
The CDR-average predictor is a constant -> AUC = 0.5 by construction; length is the discriminator."""
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


def loo_len(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(y)
    if np.unique(x).size < 2:
        return np.full(n, y.mean())
    return np.array([np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i]) for i in range(n)])


def cindex(yt, yp):
    """P(prediction ranks a random discordant-in-truth pair correctly); ties in pred count 0.5."""
    yt = np.asarray(yt); yp = np.asarray(yp); n = len(yt); c = d = t = 0
    for i in range(n):
        for j in range(i + 1, n):
            if yt[i] == yt[j]:
                continue
            dp = yp[i] - yp[j]
            if dp == 0:
                t += 1
            elif (yt[i] > yt[j]) == (dp > 0):
                c += 1
            else:
                d += 1
    tot = c + d + t
    return (c + 0.5 * t) / tot if tot else np.nan


rows = []
for cdr in CDRS:
    d = m[m.cdr == cdr]; x = d.length.values
    for qname, col, _ in QTY:
        rows.append(dict(cdr=cdr, quantity=qname, auc=cindex(d[col].values, loo_len(x, d[col].values))))
a = pd.DataFrame(rows)
a.round(3).to_csv(f"{C.RESULTS}/auc_concordance.csv", index=False)

# length variation per CDR: AUC is only meaningful where length actually varies
lenvar = m.groupby("cdr").length.nunique().reindex(CDRS)
print("AUC (concordance) of predicting an unseen TCR from LENGTH  [0.5 = no ranking power]:")
piv = a.pivot(index="cdr", columns="quantity", values="auc").reindex(CDRS)[[q[0] for q in QTY]]
piv["len_vals"] = lenvar
print(piv.round(2).to_string())

# POOLED AUC over all 126 loops: rank ANY loop's flexibility using CDR identity (leave-one-TCR-out
# CDR average) or CDR+length. This is the well-defined 'overall' AUC (CDR identity separates loops).
print("\nPOOLED AUC across all 126 loops (rank any loop's flexibility):")
for qname, col, _ in QTY:
    y = m[col].values
    # leave-one-TCR-out CDR-average prediction (varies by CDR) and +length
    predC = np.empty(len(m)); predCL = np.empty(len(m))
    for t in m.system.unique():
        tr = m.system != t; te = ~tr
        for cdr in CDRS:
            mask_tr = tr & (m.cdr == cdr); mask_te = te & (m.cdr == cdr)
            if mask_te.sum() == 0:
                continue
            ytr = m.loc[mask_tr, col].values; xtr = m.loc[mask_tr, "length"].values
            predC[mask_te.values] = ytr.mean()
            xte = m.loc[mask_te, "length"].values
            predCL[mask_te.values] = (np.polyval(np.polyfit(xtr, ytr, 1), xte)
                                      if np.unique(xtr).size >= 2 else ytr.mean())
    print(f"   {qname:18s}  CDR-identity AUC={cindex(y, predC):.2f}   CDR+length AUC={cindex(y, predCL):.2f}")

fig, ax = plt.subplots(figsize=(11, 5.2))
x = np.arange(len(CDRS)); w = 0.26
ax.axhspan(0.4, 0.6, color="0.9", zorder=0)                        # 'chance' band
for j, (qname, col, ccol) in enumerate(QTY):
    sub = a[a.quantity == qname].set_index("cdr").reindex(CDRS)
    b = ax.bar(x + (j - 1) * w, sub.auc, w, color=ccol, label=qname)
    ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
ax.axhline(0.5, color="k", lw=1, ls="--")
for i, c in enumerate(CDRS):                                       # flag near-constant-length CDRs
    if lenvar[c] <= 3:
        ax.text(i, 0.02, "length\n~constant\n→ AUC=chance", ha="center", va="bottom", fontsize=6, color="0.45")
ax.set_xticks(x); ax.set_xticklabels(CDRS, rotation=20)
ax.set_ylabel("AUC / concordance  (rank a held-out TCR from length)")
ax.set_ylim(0.0, 1.0)
ax.set_title("Ranking power (AUC) of predicting an unseen TCR's flexibility from loop length\n"
             "meaningful only where length varies (CDR3): length ranks DEFORMATION (0.60/0.71), not the hinge; grey band = chance")
ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(f"{FIG}/fig12_auc.png", bbox_inches="tight"); plt.close(fig)
print(f"\nfig -> {FIG}/fig12_auc.png")
