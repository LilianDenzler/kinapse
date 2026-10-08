#!/usr/bin/env python
"""Cross-TCR spread of the rigid (hinge-like) fraction per CDR, and vs CDR loop length."""
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from geom_hinge import CDRS
from graph_build import CDR_RANGES, HERE

files = sorted(glob.glob(f"{HERE}/results_geom/*.npz"))
tcrs = [os.path.basename(f)[:4] for f in files]
RF = np.zeros((len(files), 6)); EE = np.zeros((len(files), 6)); LEN = np.full((len(files), 6), np.nan)
for i, f in enumerate(files):
    z = np.load(f)
    for j, c in enumerate(CDRS):
        Dr, E = z[f"{c}_rig"], z[f"{c}_E"]
        RF[i, j] = np.median((Dr ** 2) / (Dr ** 2 + E ** 2)); EE[i, j] = np.median(E)
    # loop lengths from the fluctuation npz (residues present in each CDR range)
    fl = f"{HERE}/results_fluct/{tcrs[i]}_fluct.npz"
    if os.path.exists(fl):
        zf = np.load(fl)
        for j, c in enumerate(CDRS):
            ch = c[0]; lo, hi = CDR_RANGES[c[2:]]
            LEN[i, j] = np.sum((zf[f"{ch}_imgt"] >= lo) & (zf[f"{ch}_imgt"] <= hi))

COL = ["#4C72B0", "#4C72B0", "#2E5A88", "#C44E52", "#C44E52", "#8C2D2F"]
fig = plt.figure(figsize=(19, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.15, 1.0], wspace=.32)
rng = np.random.default_rng(0)

# (1) strip + box per CDR + combined
ax = fig.add_subplot(gs[0])
data = [RF[:, j] for j in range(6)] + [RF.ravel()]
labels = CDRS + ["ALL\ncombined"]
bp = ax.boxplot(data, positions=range(7), widths=.55, patch_artist=True, showfliers=False,
                medianprops=dict(color="k", lw=1.5))
for k, patch in enumerate(bp["boxes"]):
    patch.set_facecolor((COL + ["#777777"])[k]); patch.set_alpha(.25)
for j in range(6):
    ax.scatter(np.full(len(files), j) + rng.uniform(-.16, .16, len(files)), RF[:, j],
               c=EE[:, j], cmap="inferno_r", vmin=0.1, vmax=0.4, s=26, edgecolor="k", linewidths=.3, zorder=3)
sc = ax.scatter(np.full(RF.size, 6) + rng.uniform(-.16, .16, RF.size), RF.ravel(),
                c=EE.ravel(), cmap="inferno_r", vmin=0.1, vmax=0.4, s=14, edgecolor="none", alpha=.5, zorder=3)
ax.set_xticks(range(7)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("rigid (hinge-like) fraction  D_rigid²/(D_rigid²+E²)"); ax.set_ylim(0, 1)
ax.set_title("Per-CDR spread across 22 TCRs (dot = one TCR; colour = E_nonrigid)")
fig.colorbar(sc, ax=ax, fraction=.03, pad=.01, label="E_nonrigid (trust)")

# (2) heatmap TCR x CDR (sorted by mean rigid fraction)
ax2 = fig.add_subplot(gs[1])
order = np.argsort(RF.mean(1))
im = ax2.imshow(RF[order], cmap="RdYlBu", vmin=0.3, vmax=0.9, aspect="auto")
ax2.set_xticks(range(6)); ax2.set_xticklabels(CDRS, rotation=90, fontsize=7)
ax2.set_yticks(range(len(files))); ax2.set_yticklabels([tcrs[o] for o in order], fontsize=6)
ax2.set_title("rigid fraction per TCR × CDR\n(TCRs sorted; blue = hinge-like, red = deform)")
fig.colorbar(im, ax=ax2, fraction=.046, pad=.02, label="rigid fraction")

# (3) rigid fraction vs CDR3 loop length
ax3 = fig.add_subplot(gs[2])
for j, c in [(2, "A_CDR3"), (5, "B_CDR3")]:
    m = ~np.isnan(LEN[:, j])
    ax3.scatter(LEN[m, j], RF[m, j], s=30, c=COL[j], edgecolor="k", linewidths=.3, label=c)
    if m.sum() > 3:
        r = np.corrcoef(LEN[m, j], RF[m, j])[0, 1]
        ax3.text(.03, .12 - .07 * (j > 3), f"{c}: r={r:.2f}", transform=ax3.transAxes, fontsize=9, color=COL[j])
ax3.set_xlabel("CDR3 loop length (residues)"); ax3.set_ylabel("rigid fraction"); ax3.set_ylim(0, 1)
ax3.set_title("hinge-fraction vs CDR3 length"); ax3.legend(fontsize=8)

fig.suptitle("Rigid (hinge-like) fraction of framework-relative motion — cross-TCR comparison", y=1.02, fontsize=13)
out = f"{HERE}/figures/rigidfraction_spread.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("fig ->", out)
# correlations of rigid fraction with loop length, all CDRs
print("corr(rigid fraction, loop length) per CDR:")
for j, c in enumerate(CDRS):
    m = ~np.isnan(LEN[:, j])
    if m.sum() > 3 and np.std(LEN[m, j]) > 0:
        print(f"  {c}: r={np.corrcoef(LEN[m, j], RF[m, j])[0,1]:+.2f}  (len range {int(LEN[m,j].min())}-{int(LEN[m,j].max())})")
    else:
        print(f"  {c}: constant length {int(np.nanmin(LEN[:,j]))}")
