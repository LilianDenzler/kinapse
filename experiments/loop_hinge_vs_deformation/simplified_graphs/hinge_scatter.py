#!/usr/bin/env python
"""Per-frame, alignment-free hinge/deformation phenotype maps for each CDR of one TCR.

For each CDR loop L and the fluctuation-derived RIGID set R, per MD frame t (ref = mean over frames):
  D_LL(t) = sqrt(  mean_{ij in L,L} [d_ij(t) - d_ij^ref]^2 )     loop INTERNAL distance change (deformation)
  D_LR(t) = sqrt(  mean_{ia in L,R} [d_ia(t) - d_ia^ref]^2 )     loop -> rigid distance change (total relative)
Then remove the part of d_LR that loop deformation alone predicts:
  z_t = PCA scores of centered d_LL ;   B = lstsq(z, delta d_LR) ;   h_t = delta d_LR(t) - B z_t
  D_H(t) = sqrt( mean_{ia in L,R} h_t^2 )                          deformation-CORRECTED relative motion
All RMS-per-edge (not sums), so d_LR's larger edge count doesn't inflate it.

Plot 1 (x=D_LL, y=D_LR) = phenotype.  Plot 2 (x=D_LL, y=D_H) = hinge-like (D_H) vs deformation (D_LL).
CLI:  python hinge_scatter.py 3QH3
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from graph_build import load_md, CDR_RANGES, HERE

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]


def scatter_stats(loop, R, var=0.95, kmax=10):
    """loop (T,N,3), R (T,M,3). Returns D_LL, D_LR, D_H (each length T) and n deformation modes k."""
    T, N, _ = loop.shape
    iu = np.triu_indices(N, 1)
    dLL = np.linalg.norm(loop[:, iu[0]] - loop[:, iu[1]], axis=-1)                 # (T, N_LL)
    dLR = np.linalg.norm(loop[:, :, None, :] - R[:, None, :, :], axis=-1).reshape(T, -1)  # (T, N_LR)
    ddLL = dLL - dLL.mean(0); ddLR = dLR - dLR.mean(0)
    D_LL = np.sqrt((ddLL ** 2).mean(1)); D_LR = np.sqrt((ddLR ** 2).mean(1))
    U, S, _ = np.linalg.svd(ddLL, full_matrices=False)                            # deformation modes from d_LL
    cum = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-12)
    k = int(min(np.searchsorted(cum, var) + 1, kmax, len(S)))
    z = U[:, :k] * S[:k]
    B = np.linalg.lstsq(z, ddLR, rcond=None)[0]                                   # deformation's footprint on d_LR
    h = ddLR - z @ B                                                              # residual = rigid-relative
    D_H = np.sqrt((h ** 2).mean(1))
    return D_LL, D_LR, D_H, k


def main(sysid):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    RSET = {"A": rig["chain_A"]["ultra_rigid"], "B": rig["chain_B"]["ultra_rigid"]}
    stats = {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)                         # loop residues incl. insertions
        ri = np.array([imap[ch][r] for r in RSET[ch] if r in imap[ch]])
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        stats[cdr] = scatter_stats(loop, xyz[:, ri])

    xmax = max(s[0].max() for s in stats.values()) * 1.05
    ylr = max(s[1].max() for s in stats.values()) * 1.05
    yh = max(s[2].max() for s in stats.values()) * 1.05
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharex=True)
    for j, cdr in enumerate(CDRS):
        D_LL, D_LR, D_H, k = stats[cdr]
        for row, (y, ylab, ymax, col) in enumerate([(D_LR, "D_LR", ylr, "#4C72B0"), (D_H, "D_H", yh, "#8E44AD")]):
            ax = axes[row, j]
            ax.scatter(D_LL, y, s=5, c=col, alpha=.18, edgecolor="none")
            ax.axvline(np.median(D_LL), color="0.6", lw=.7, ls=":"); ax.axhline(np.median(y), color="0.6", lw=.7, ls=":")
            ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
            if row == 0:
                ax.set_title(f"{cdr}\n(k={k} deform modes)", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{ylab}  (Å, RMS/edge)", fontsize=9)
            if row == 1:
                ax.set_xlabel("D_LL  (Å, RMS/edge)", fontsize=8)
    axes[0, 0].text(.02, .96, "phenotype: y=D_LR", transform=axes[0, 0].transAxes, fontsize=8, va="top", color="#4C72B0")
    axes[1, 0].text(.02, .96, "deform-corrected: y=D_H", transform=axes[1, 0].transAxes, fontsize=8, va="top", color="#8E44AD")
    fig.suptitle(f"{sysid} — alignment-free hinge/deformation maps per CDR (each dot = one MD frame)\n"
                 "top: D_LL vs D_LR (total loop→rigid motion) · bottom: D_LL vs D_H (deformation-corrected = rigid-relative/hinge-like)",
                 y=1.02, fontsize=13)
    out = f"{HERE}/figures/hingemap_{sysid}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)
    print(f"{'CDR':8}{'medD_LL':>9}{'medD_LR':>9}{'medD_H':>9}{'D_H/D_LR':>10}")
    for cdr in CDRS:
        a, b, c, k = stats[cdr]
        print(f"{cdr:8}{np.median(a):9.2f}{np.median(b):9.2f}{np.median(c):9.2f}{np.median(c)/max(np.median(b),1e-6):10.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3QH3")
