#!/usr/bin/env python
"""Total loop flexibility split into hinge + deformation PARTS that add up to total, same unit (Å).
Decomposition is Pythagorean: total² = hinge_rmsd² + E_nonrigid².  To stack additively to `total`
in Å, each gets its variance share:  hinge²/total + deform²/total = total.
Left  = absolute (Å, bars sum to total flexibility); Right = normalized to 100% (the hinge/deform balance).
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
C_DEF, C_HIN = "#DD8452", "#4C72B0"          # deformation / hinge

m = pd.read_csv(f"{C.RESULTS}/master_percdr.csv")
m = m[m.system != "1KGC"].copy()             # 10x breadth outlier
# hinge displacement (A) and its variance-share split of total flexibility
m["hinge_rmsd"] = np.sqrt(np.clip(m.total_disp**2 - m.E_nonrigid**2, 0, None))
m["c_deform"] = m.E_nonrigid**2 / m.total_disp        # deform contribution to total (Å)
m["c_hinge"] = m.hinge_rmsd**2 / m.total_disp         # hinge  contribution to total (Å)
# per-CDR means (c_deform + c_hinge == total_disp by construction)
g = m.groupby("cdr")[["c_deform", "c_hinge", "total_disp"]].mean().reindex(CDRS)
sd_tot = m.groupby("cdr")["total_disp"].std().reindex(CDRS)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(CDRS))

# ---- A: absolute, stacked to total flexibility (Å) ------------------------
axA.bar(x, g.c_deform, color=C_DEF, label="deformation")
axA.bar(x, g.c_hinge, bottom=g.c_deform, color=C_HIN, label="hinge (reorientation)")
axA.errorbar(x, g.total_disp, yerr=sd_tot, fmt="none", ecolor="0.35", capsize=3, lw=1)
for i, c in enumerate(CDRS):
    axA.text(i, g.total_disp[c] + sd_tot[c] + 0.03, f"{g.total_disp[c]:.2f}", ha="center", fontsize=7.5)
axA.set_xticks(x); axA.set_xticklabels(CDRS, rotation=20)
axA.set_ylabel("total loop flexibility (Å)")
axA.set_title("A · total flexibility = hinge + deformation  (Å; parts sum to total)\n"
              "split by variance share: hinge²/total + deform²/total = total")
axA.legend(fontsize=8, loc="upper left"); axA.grid(axis="y", alpha=.25)

# ---- B: normalized to 100% (the balance) ----------------------------------
fd = 100 * g.c_deform / g.total_disp
fh = 100 * g.c_hinge / g.total_disp
axB.bar(x, fd, color=C_DEF, label="deformation")
axB.bar(x, fh, bottom=fd, color=C_HIN, label="hinge (reorientation)")
for i, c in enumerate(CDRS):
    axB.text(i, fd[c] / 2, f"{fd[c]:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
    axB.text(i, fd[c] + fh[c] / 2, f"{fh[c]:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
axB.set_xticks(x); axB.set_xticklabels(CDRS, rotation=20)
axB.set_ylabel("share of total flexibility (%)"); axB.set_ylim(0, 100)
axB.set_title("B · hinge / deformation balance (normalized to 100%)")
axB.legend(fontsize=8, loc="lower left"); axB.grid(axis="y", alpha=.25)

fig.suptitle("Total CDR-loop flexibility split into its hinge and deformation parts (21 TCRs, 1KGC excluded)",
             fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig9_flexibility_decomposition.png", bbox_inches="tight"); plt.close(fig)

print("per-CDR contributions to total flexibility (Å; deform + hinge = total):")
print(g.round(3).assign(hinge_pct=(100*g.c_hinge/g.total_disp).round(1)).to_string())
print(f"\nfig -> {FIG}/fig9_flexibility_decomposition.png")
