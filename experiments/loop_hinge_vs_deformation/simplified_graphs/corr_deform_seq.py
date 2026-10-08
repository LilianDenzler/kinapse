#!/usr/bin/env python
"""Does CDR composition (Gly, Pro, ...) explain internal flexibility (eps_deform) beyond loop length?"""
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from graph_build import HERE

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
METRICS = {"Gly": "G", "Pro": "P", "aromatic": "FWY", "charged": "DEKR", "small(GAS)": "GAS", "β-branch(VIT)": "VIT"}
seqs = json.load(open(f"{HERE}/cdr_sequences.json"))

rows = {c: [] for c in CDRS}      # per CDR: (sysid, eps, N, seq)
for f in sorted(glob.glob(f"{HERE}/results_deform/*.npz")):
    s = os.path.basename(f)[:4]; z = np.load(f)
    for c in CDRS:
        rows[c].append((s, float(z[f"{c}_eps"]), int(z[f"{c}_N"]), seqs[s][c]))


def frac(seq, aas):
    return sum(seq.count(a) for a in aas) / len(seq) if seq else np.nan


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan


# ---- correlations of eps with each composition metric ----
print("corr(eps_deform, composition)  — CDR3 pooled (A+B, n=44) and ALL-CDR pooled (n=132)")
cdr3 = rows["A_CDR3"] + rows["B_CDR3"]
allc = [r for c in CDRS for r in rows[c]]
print(f"{'metric':14}{'CDR3 r':>9}{'ALL r':>9}")
eps3 = [r[1] for r in cdr3]; epsA = [r[1] for r in allc]
print(f"{'length':14}{corr([r[2] for r in cdr3], eps3):9.2f}{corr([r[2] for r in allc], epsA):9.2f}")
for name, aas in METRICS.items():
    print(f"{name:14}{corr([frac(r[3], aas) for r in cdr3], eps3):9.2f}{corr([frac(r[3], aas) for r in allc], epsA):9.2f}")

# ---- does Gly/Pro add beyond length? (partial: regress out length, correlate residual with comp) ----
print("\nPARTIAL corr with eps after removing CDR3-length effect (CDR3 pooled):")
L = np.array([r[2] for r in cdr3], float); E = np.array(eps3)
Eres = E - np.polyval(np.polyfit(L, E, 1), L)                      # eps residual after length
for name, aas in METRICS.items():
    g = np.array([frac(r[3], aas) for r in cdr3])
    gres = g - np.polyval(np.polyfit(L, g, 1), L)                  # composition residual after length
    print(f"  {name:14} partial r = {corr(gres, Eres):+.2f}")

# ---- plot: eps vs Gly and vs Pro (CDR3), + correlation bars ----
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for k, (name, aas) in enumerate([("Gly", "G"), ("Pro", "P")]):
    for c, col in [("A_CDR3", "#2E5A88"), ("B_CDR3", "#8C2D2F")]:
        g = [frac(r[3], aas) for r in rows[c]]; e = [r[1] for r in rows[c]]
        ax[k].scatter(g, e, s=32, c=col, edgecolor="k", linewidths=.3, label=f"{c} r={corr(g,e):.2f}")
    ax[k].set_xlabel(f"{name} fraction of CDR3"); ax[k].set_ylabel("ε_deform"); ax[k].legend(fontsize=8)
    ax[k].set_title(f"CDR3 flexibility vs {name} content")
# correlation bar chart
mets = ["length"] + list(METRICS)
rs3 = [corr([r[2] for r in cdr3], eps3)] + [corr([frac(r[3], a) for r in cdr3], eps3) for a in METRICS.values()]
ax[2].barh(range(len(mets)), rs3, color=["#777"] + ["#4C9A5B"] * len(METRICS))
ax[2].set_yticks(range(len(mets))); ax[2].set_yticklabels(mets, fontsize=9); ax[2].invert_yaxis()
ax[2].axvline(0, color="k", lw=.8); ax[2].set_xlabel("corr with ε_deform (CDR3 pooled)")
ax[2].set_title("what predicts CDR3 flexibility")
fig.suptitle("CDR3 internal flexibility (ε_deform) vs sequence composition — 22 TCRs (α+β)", y=1.02, fontsize=13)
out = f"{HERE}/figures/deform_vs_composition.png"
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("\nfig ->", out)
