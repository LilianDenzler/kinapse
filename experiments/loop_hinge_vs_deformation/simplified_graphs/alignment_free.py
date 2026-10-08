#!/usr/bin/env python
"""ALIGNMENT-FREE rigid-motion description from CA-CA distances, using the ultra-rigid framework as reference.
Distances are SE(3)-invariant, so no superposition is needed:
  d_LL (loop-loop)          -> pose-blind  -> DEFORMATION only          eps_deform = sqrt(mean Var d_LL)
  d_LR (loop -> ultra-rigid) -> encodes the loop's POSE vs the framework -> rigid motion  eps_pose = sqrt(mean Var d_LR)
Per-loop-residue Var[d_LR] PROFILE types the motion alignment-free:
  hinge (tip-elevation) -> fluctuation peaks at the TIP (feet on/near the axis stay put)
  twist (about up-loop)  -> fluctuation shifts toward the FREE foot (tip near axis moves least)
Spot-checks eps_pose against the coordinate/alignment-based rigid tip-motion (results_rotframe.json).
-> figures/alignment_free_<sys>.png"""
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
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}
COL = {"A_CDR1": "#4C72B0", "A_CDR2": "#4C72B0", "A_CDR3": "#2E5A88",
       "B_CDR1": "#C44E52", "B_CDR2": "#C44E52", "B_CDR3": "#8C2D2F"}


def analyze(loop, R):
    """loop (T,N,3), R (T,M,3) ultra-rigid. All from raw MD coords -> distances need NO alignment."""
    T, N, _ = loop.shape
    ii, jj = np.triu_indices(N, 1)
    dLL = np.linalg.norm(loop[:, ii] - loop[:, jj], axis=-1)              # (T, N*(N-1)/2)
    dLR = np.linalg.norm(loop[:, :, None, :] - R[:, None, :, :], axis=-1)  # (T, N, M)
    eps_deform = float(np.sqrt(np.var(dLL, axis=0).mean()))
    eps_pose = float(np.sqrt(np.var(dLR, axis=0).mean()))
    pose_prof = np.sqrt(np.var(dLR, axis=0).mean(axis=1))                 # per loop residue: pose fluct (A)
    def_prof = np.zeros(N)                                                # per loop residue: deform fluct (A)
    vLL = np.var(dLL, axis=0)
    for k, (a, b) in enumerate(zip(ii, jj)):
        def_prof[a] += vLL[k]; def_prof[b] += vLL[k]
    def_prof = np.sqrt(def_prof / (N - 1))
    return eps_deform, eps_pose, pose_prof, def_prof


def main(sysid="3SKN"):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    rot = json.load(open(f"{HERE}/results_rotframe.json")).get(sysid, {})
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    print(f"{sysid}  (alignment-free, distances only)")
    print(f"{'CDR':8}{'eps_deform':>11}{'eps_pose':>10}{'peak@':>8}{'  vs rotframe tip(A)':>20}")
    for idx, cdr in enumerate(CDRS):
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        R = xyz[:, np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])]
        ed, ep, pp, dp = analyze(loop, R)
        N = len(lk); pos = np.arange(N)
        peak = int(np.argmax(pp))
        clampend = "N(res0)" if CLAMP[cdr[2:]] == "N" else f"C(res{N-1})"
        tip = rot.get(cdr, {}).get("tip", float("nan"))
        print(f"{cdr:8}{ed:11.2f}{ep:10.2f}{peak:8d}{tip:20.2f}")
        a = ax.flat[idx]
        a.plot(pos, pp, "-o", color=COL[cdr], ms=4, lw=1.8, label="pose  Var[d_LR]  (rigid, align-free)")
        a.plot(pos, dp, "--s", color="0.5", ms=3, lw=1.3, label="deform  Var[d_LL]")
        cl = 0 if CLAMP[cdr[2:]] == "N" else N - 1
        a.axvline(cl, color="k", lw=1, ls=":"); a.text(cl, a.get_ylim()[1], " clamp", fontsize=7, va="top")
        a.set_title(f"{cdr}   ε_pose={ep:.2f}  ε_deform={ed:.2f} Å", fontsize=9)
        a.set_xlabel("loop residue (N-foot → tip → C-foot)"); a.set_ylabel("Å fluctuation")
        if idx == 0:
            a.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{sysid}: alignment-free rigid motion from CA–CA distances (ultra-rigid reference)\n"
                 "d_LR (rigid pose) low at the clamp, rises away — peak at a FREE FOOT = single-anchor lever (CDR1/2); "
                 "tent with both feet low = doubly-anchored tip-swing (CDR3).  d_LL (grey) = deformation", y=1.03, fontsize=11)
    out = f"{HERE}/figures/alignment_free_{sysid}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3SKN")
