#!/usr/bin/env python
"""Geometric per-frame rigid-fit decomposition of CDR motion — NO PCA, NO regression, NO assumed axis.

Reference = the ensemble MEDOID frame (loop shape closest to average): reference loop L0, framework R0.
For each frame t, fit the 6-DOF rigid placement T_t of the UNDEFORMED reference loop L0 that best matches
the observed loop->framework geometry (closed-form Kabsch onto the framework-superposed loop). Then, all in
DISTANCE space (alignment-free interpretation):

  D_deform(t)   = RMS[ d_LL(t)      - d_LL^0 ]                 internal deformation (direct; no model)
  D_rigid(t)    = RMS[ d_LR^rigid(t)- d_LR^0 ]                 how far the best RIGID loop moved vs framework
  E_nonrigid(t) = RMS[ d_LR^obs(t)  - d_LR^rigid(t) ]          how badly a rigid-loop model fits (trust)

Phenotype (D_deform vs D_rigid): small/small stable · small/large hinge-like · large/small deformation ·
large/large both.  E_nonrigid says how much to trust the rigid description of that frame.
CLI:  python geom_hinge.py 3QH3
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


def kabsch(P, Q):
    H = P.T @ Q; U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def geom_decomp(loop, R):
    """loop (T,N,3), R (T,M,3) rigid framework. Returns D_deform, D_rigid, E_nonrigid (length T)."""
    T, N, _ = loop.shape
    iu = np.triu_indices(N, 1)
    dLL = np.linalg.norm(loop[:, iu[0]] - loop[:, iu[1]], axis=-1)          # (T, N_LL)  rigid-invariant
    med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))                 # medoid = most average loop shape
    # superpose every frame's framework onto the medoid framework -> loop in the fixed R0 frame
    Rref = R[med] - R[med].mean(0); Rc = R[med].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Rot = kabsch(R[t] - R[t].mean(0), Rref)
        Lf[t] = (loop[t] - R[t].mean(0)) @ Rot.T + Rc
    L0 = Lf[med]; R0 = R[med]; L0c = L0 - L0.mean(0)
    dLL0 = dLL[med]
    dLR0 = np.linalg.norm(L0[:, None, :] - R0[None, :, :], axis=-1).ravel()
    D_deform = np.sqrt(((dLL - dLL0) ** 2).mean(1))
    D_rigid = np.empty(T); E = np.empty(T)
    for t in range(T):
        Tt = kabsch(L0c, Lf[t] - Lf[t].mean(0))                            # rigid placement of L0 at frame t
        Lrig = L0c @ Tt.T + Lf[t].mean(0)
        dR_rig = np.linalg.norm(Lrig[:, None, :] - R0[None, :, :], axis=-1).ravel()
        dR_obs = np.linalg.norm(Lf[t][:, None, :] - R0[None, :, :], axis=-1).ravel()
        D_rigid[t] = np.sqrt(((dR_rig - dLR0) ** 2).mean())
        E[t] = np.sqrt(((dR_obs - dR_rig) ** 2).mean())
    return D_deform, D_rigid, E


def main(sysid):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    RSET = {"A": rig["chain_A"]["ultra_rigid"], "B": rig["chain_B"]["ultra_rigid"]}
    stats = {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        ri = np.array([imap[ch][r] for r in RSET[ch] if r in imap[ch]])
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        stats[cdr] = geom_decomp(loop, xyz[:, ri])

    xm = max(s[0].max() for s in stats.values()) * 1.05
    ym = max(s[1].max() for s in stats.values()) * 1.05
    em = max(s[2].max() for s in stats.values())
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.4), sharex=True, sharey=True)
    for j, cdr in enumerate(CDRS):
        Dd, Dr, E = stats[cdr]; ax = axes[j]
        sc = ax.scatter(Dd, Dr, c=E, s=6, cmap="inferno_r", vmin=0, vmax=em, alpha=.6, edgecolor="none")
        ax.axvline(np.median(Dd), color="0.6", lw=.7, ls=":"); ax.axhline(np.median(Dr), color="0.6", lw=.7, ls=":")
        ax.set_xlim(0, xm); ax.set_ylim(0, ym)
        ax.set_title(cdr, fontsize=10); ax.set_xlabel("D_deform (Å)", fontsize=9)
        if j == 0:
            ax.set_ylabel("D_rigid (Å)", fontsize=10)
    fig.colorbar(sc, ax=axes, fraction=.012, pad=.01, label="E_nonrigid (Å) — rigid-fit error (trust)")
    fig.suptitle(f"{sysid} — geometric rigid-fit decomposition per CDR (each dot = one MD frame; no PCA/regression)\n"
                 "x=D_deform (internal), y=D_rigid (rigid loop→framework motion), colour=E_nonrigid (how well a rigid loop fits)",
                 y=1.04, fontsize=13)
    out = f"{HERE}/figures/geomhinge_{sysid}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)
    print(f"{'CDR':8}{'medDeform':>10}{'medRigid':>10}{'medE':>8}{'rigid frac':>11}")
    for cdr in CDRS:
        Dd, Dr, E = stats[cdr]
        frac = np.median(Dr) / max(np.median(Dr) + np.median(Dd), 1e-6)
        print(f"{cdr:8}{np.median(Dd):10.2f}{np.median(Dr):10.2f}{np.median(E):8.2f}{frac:11.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3QH3")
