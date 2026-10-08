#!/usr/bin/env python
"""Pool the geometric rigid-fit decomposition over all TCRs and plot per-CDR phenotype maps."""
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from geom_hinge import CDRS
from graph_build import HERE

RES = f"{HERE}/results_geom"
files = sorted(glob.glob(f"{RES}/*.npz"))
pool = {cdr: {"def": [], "rig": [], "E": []} for cdr in CDRS}
for f in files:
    z = np.load(f)
    for cdr in CDRS:
        for key in ("def", "rig", "E"):
            pool[cdr][key].append(z[f"{cdr}_{key}"])
for cdr in CDRS:
    for key in ("def", "rig", "E"):
        pool[cdr][key] = np.concatenate(pool[cdr][key])
print(f"{len(files)} TCRs pooled; ~{len(pool['A_CDR1']['def'])} frames/CDR")

rng = np.random.default_rng(0)
xm = np.percentile(np.concatenate([pool[c]["def"] for c in CDRS]), 99.5)
ym = np.percentile(np.concatenate([pool[c]["rig"] for c in CDRS]), 99.5)
em = np.percentile(np.concatenate([pool[c]["E"] for c in CDRS]), 98)
fig, axes = plt.subplots(1, 6, figsize=(24, 4.6), sharex=True, sharey=True)
for j, cdr in enumerate(CDRS):
    Dd, Dr, E = pool[cdr]["def"], pool[cdr]["rig"], pool[cdr]["E"]
    if len(Dd) > 20000:                                     # subsample for a legible cloud
        s = rng.choice(len(Dd), 20000, replace=False); Dd, Dr, E = Dd[s], Dr[s], E[s]
    ax = axes[j]
    sc = ax.scatter(Dd, Dr, c=E, s=4, cmap="inferno_r", vmin=0, vmax=em, alpha=.4, edgecolor="none", rasterized=True)
    ax.axvline(np.median(pool[cdr]["def"]), color="0.5", lw=.7, ls=":")
    ax.axhline(np.median(pool[cdr]["rig"]), color="0.5", lw=.7, ls=":")
    ax.set_xlim(0, xm); ax.set_ylim(0, ym)
    rigfrac = np.median(pool[cdr]["rig"]) / max(np.median(pool[cdr]["rig"]) + np.median(pool[cdr]["def"]), 1e-6)
    ax.set_title(f"{cdr}\nmed E={np.median(pool[cdr]['E']):.2f}  rigidfrac={rigfrac:.2f}", fontsize=9.5)
    ax.set_xlabel("D_deform (Å)", fontsize=9)
    if j == 0:
        ax.set_ylabel("D_rigid (Å)", fontsize=10)
fig.colorbar(sc, ax=axes, fraction=.012, pad=.01, label="E_nonrigid (Å) — rigid-fit error (dark = rigid model fails)")
fig.suptitle(f"Geometric rigid-fit decomposition, POOLED over {len(files)} TCRs (each dot = one MD frame; no PCA/regression)\n"
             "x=D_deform (internal) · y=D_rigid (rigid loop→framework motion) · colour=E_nonrigid (trust of the rigid description)",
             y=1.05, fontsize=13)
out = f"{HERE}/figures/geomhinge_pooled.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
print("fig ->", out)
