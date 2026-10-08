#!/usr/bin/env python
"""Do the 22 TCRs share one hinge-vs-deformation profile across the CDRs, or differ?
Panels: (A) each TCR's hinge_frac profile across the 6 CDRs; (B) TCR x CDR heatmap of the
deviation from each CDR's mean (isolates TCR-specificity); (C) variance partition
hinge_frac ~ CDR + TCR + interaction. Run after plot_hinge_deform.py."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
import config as C

FIG = os.path.join(C.ROOT, "figures")
CDRS = C.CDRS
plt.rcParams.update({"font.size": 9, "figure.dpi": 130, "axes.axisbelow": True})

m = pd.read_csv(f"{C.RESULTS}/master_percdr.csv")
METRIC = "hinge_frac"                                   # the hinge-vs-deform balance in [0,1]
P = m.pivot(index="system", columns="cdr", values=METRIC)[CDRS]     # (22 TCR, 6 CDR)
systems = P.index.tolist()


def variance_partition(df_long, value):
    """Two-way (CDR + TCR) SS decomposition of `value`. Returns %variance for each term."""
    x = df_long[value].values
    gm = x.mean()
    ss_tot = ((x - gm) ** 2).sum()
    cdr_mean = df_long.groupby("cdr")[value].transform("mean")
    tcr_mean = df_long.groupby("system")[value].transform("mean")
    ss_cdr = ((cdr_mean - gm) ** 2).sum()               # CDR identity (universal loop pattern)
    ss_tcr = ((tcr_mean - gm) ** 2).sum()               # TCR identity (some TCRs floppier overall)
    ss_res = ss_tot - ss_cdr - ss_tcr                   # TCR x CDR interaction + within noise
    return dict(CDR=100 * ss_cdr / ss_tot, TCR=100 * ss_tcr / ss_tot,
                interaction=100 * ss_res / ss_tot)


fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.5, 0.7], wspace=0.32)

# ---- A: profile lines -----------------------------------------------------
axA = fig.add_subplot(gs[0])
xpos = np.arange(len(CDRS))
mean_prof = P.mean(0).values
sd_prof = P.std(0).values
for s in systems:
    axA.plot(xpos, P.loc[s].values, color="0.7", lw=0.8, alpha=0.6, zorder=1)
axA.fill_between(xpos, mean_prof - sd_prof, mean_prof + sd_prof, color="#4C72B0", alpha=0.2, zorder=2)
axA.plot(xpos, mean_prof, color="#1f3d6b", lw=2.6, marker="o", zorder=4, label="mean of 22 TCRs")
lvl = P.mean(1).sort_values()
for s, c, lab in [(lvl.index[0], "#C44E52", f"{lvl.index[0]} (most deform)"),
                  (lvl.index[-1], "#2E8B57", f"{lvl.index[-1]} (most hinge)")]:
    axA.plot(xpos, P.loc[s].values, color=c, lw=1.8, marker="s", ms=4, zorder=5, label=lab)
axA.set_xticks(xpos); axA.set_xticklabels(CDRS, rotation=25)
axA.set_ylabel("hinge fraction  (1 − $E_{nonrigid}^2$/total$^2$)")
axA.set_ylim(0.4, 1.0); axA.grid(alpha=0.25)
axA.set_title("A · each grey line = one TCR's profile across the 6 CDRs")
axA.legend(fontsize=7, loc="lower left")

# ---- B: TCR x CDR heatmap of deviation from each CDR's mean ----------------
axB = fig.add_subplot(gs[1])
Dev = P - P.mean(0)                                     # remove the universal CDR pattern
order = leaves_list(linkage(Dev.values, method="ward"))
Dv = Dev.iloc[order]
vmax = np.abs(Dev.values).max()
im = axB.imshow(Dv.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axB.set_xticks(range(len(CDRS))); axB.set_xticklabels(CDRS, rotation=25)
axB.set_yticks(range(len(systems))); axB.set_yticklabels(Dv.index, fontsize=6.5)
axB.set_title("B · deviation from each CDR's mean (TCR-specificity; rows clustered)")
fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02, label="Δ hinge fraction")

# ---- C: variance partition ------------------------------------------------
axC = fig.add_subplot(gs[2])
vp = variance_partition(m, METRIC)
labels = ["CDR\nidentity", "TCR\nidentity", "TCR×CDR\n+ noise"]
vals = [vp["CDR"], vp["TCR"], vp["interaction"]]
cols = ["#4C72B0", "#DD8452", "#B0B0B0"]
axC.bar(labels, vals, color=cols)
for i, v in enumerate(vals):
    axC.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
axC.set_ylabel("% of variance in hinge fraction")
axC.set_ylim(0, max(vals) * 1.2)
axC.set_title("C · what explains the\nhinge-vs-deform balance?")

fig.suptitle("Do TCRs share one hinge-vs-deformation profile, or differ per CDR?", fontsize=12, y=1.02)
fig.savefig(f"{FIG}/fig5_tcr_profiles.png", bbox_inches="tight"); plt.close(fig)

print("variance partition (% of total variance):")
for met in ["hinge_frac", "H", "D_deform", "D_framework", "theta", "total_disp"]:
    vp = variance_partition(m, met)
    print(f"  {met:12s}  CDR={vp['CDR']:5.1f}  TCR={vp['TCR']:5.1f}  interaction={vp['interaction']:5.1f}")
print("\nper-CDR hinge_frac: mean +- SD across TCRs (min..max):")
for cdr in CDRS:
    v = P[cdr]
    print(f"  {cdr}: {v.mean():.2f} +- {v.std():.2f}  ({v.min():.2f}..{v.max():.2f})")
print(f"\nfig -> {FIG}/fig5_tcr_profiles.png")
