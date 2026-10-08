#!/usr/bin/env python
"""FULLY alignment-free hinge/twist typing via multilateration, as a HEAD-TO-HEAD validation vs Kabsch.
Pipeline uses CA-CA distances only (no superposition anywhere):
  1. ultra-rigid framework reference R_can  <- classical MDS of the framework's MEAN internal distances (d_RR)
  2. each loop CA placed in that frame each frame  <- linear multilateration from d_LR (loop->anchor distances)
  3. decompose the reconstructed loop with the SAME loop-frame axes (ĉ/ĥ/n̂) -> f_c/f_h/f_n
Compare the resulting typing to the Kabsch-based results_rotframe.json. Agreement => the hinge/twist
classification is intrinsic to the distances, not an alignment artifact. -> figures/multilat_vs_kabsch_<sys>.png"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from graph_build import load_md, CDR_RANGES, HERE
from compute_rotframe import analyze as decompose      # same axis decomposition, reused

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]


def mds(D):
    """classical MDS of a distance matrix D (M,M) -> coords (M,3)."""
    n = D.shape[0]; J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:3]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def multilaterate(dLR, Rcan):
    """dLR (T,N,M) loop->anchor distances; Rcan (M,3) fixed anchors. Returns loop coords (T,N,3), align-free."""
    a0 = Rcan[0]; A = 2.0 * (Rcan[1:] - a0)                       # (M-1,3), same for all atoms/frames
    P = np.linalg.pinv(A)                                          # (3, M-1)
    const = (Rcan[1:] ** 2).sum(1) - (a0 ** 2).sum()              # (M-1,)
    b = const[None, None, :] - (dLR[:, :, 1:] ** 2 - dLR[:, :, 0:1] ** 2)   # (T,N,M-1)
    return b @ P.T                                                 # (T,N,3)


def main(sysid="3SKN"):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    kab = json.load(open(f"{HERE}/results_rotframe.json")).get(sysid, {})
    Rcan = {}
    for ch in "AB":
        ridx = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
        R = xyz[:, ridx]                                           # (T,M,3)
        dRR = np.linalg.norm(R[:, :, None, :] - R[:, None, :, :], axis=-1).mean(0)   # mean internal distances
        Rcan[ch] = mds(dRR)
    print(f"{sysid}  multilateration (align-free)  vs  Kabsch (align-based)")
    print(f"{'CDR':8}{'f_c ML/Kab':>16}{'f_h ML/Kab':>16}{'dominant ML/Kab':>18}{'agree?':>8}")
    rows = []
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        ridx = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
        R = xyz[:, ridx]
        dLR = np.linalg.norm(loop[:, :, None, :] - R[:, None, :, :], axis=-1)
        loop_can = multilaterate(dLR, Rcan[ch])                    # loop reconstructed from distances only
        T, M = loop_can.shape[0], Rcan[ch].shape[0]
        Rconst = np.broadcast_to(Rcan[ch], (T, M, 3))             # constant framework -> analyze does no realignment
        ml = decompose(loop_can, Rconst)
        kb = kab.get(cdr, {})
        dom_ml = max("cnh", key=lambda k: ml[f"f_{k}"])
        dom_kb = max("cnh", key=lambda k: kb.get(f"f_{k}", 0)) if kb else "?"
        agree = "yes" if dom_ml == dom_kb else "NO"
        print(f"{cdr:8}{ml['f_c']:.2f}/{kb.get('f_c',0):.2f}     {ml['f_h']:.2f}/{kb.get('f_h',0):.2f}       "
              f"{dom_ml}/{dom_kb:>3}        {agree:>5}")
        rows.append((cdr, ml, kb))
    # scatter: multilat vs kabsch for f_c and f_h
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    for j, (key, lab) in enumerate([("f_c", "hinge share f_c (ĉ)"), ("f_h", "twist share f_h (ĥ)")]):
        mlv = [r[1][key] for r in rows]; kbv = [r[2].get(key, np.nan) for r in rows]
        ax[j].plot([0, 1], [0, 1], "0.7", lw=1, ls="--")
        ax[j].scatter(kbv, mlv, s=60, color="#0072B2", edgecolor="white")
        for (cdr, _, _), xk, ym in zip(rows, kbv, mlv):
            ax[j].annotate(cdr, (xk, ym), fontsize=7, xytext=(4, 3), textcoords="offset points")
        r = np.corrcoef(kbv, mlv)[0, 1]
        ax[j].set_xlabel(f"Kabsch (align-based)  {lab}"); ax[j].set_ylabel(f"multilateration (align-free)  {lab}")
        ax[j].set_title(f"{lab}   r={r:+.2f}"); ax[j].set_xlim(0, 1); ax[j].set_ylim(0, 1); ax[j].set_aspect("equal")
    fig.suptitle(f"{sysid}: is hinge/twist typing alignment-free? multilateration (distances only) vs Kabsch", y=1.02, fontsize=12)
    out = f"{HERE}/figures/multilat_vs_kabsch_{sysid}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3SKN")
