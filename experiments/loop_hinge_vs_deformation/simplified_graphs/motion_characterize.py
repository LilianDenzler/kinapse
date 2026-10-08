#!/usr/bin/env python
"""Characterize the fitted rigid motion of each CDR as a VECTOR, not just a fraction.

Per frame, the geometric rigid fit gives Q_t (rotation), a_t (translation), in the framework frame.
Extract:  rotation vector omega_t = theta_t * u_t ;  screw-axis location (pivot) ;  centroid displacement.
Then per CDR report:
  theta_deg   : rotation amplitude (median)
  P1_axis     : how 1-D the rotation-axis cloud is  (lambda1/sum of <omega omega^T>); ->1 = single hinge axis
  pivot_frac  : screw-axis location along the N->C stem axis (0=N stem,1=C stem)
  D_int       : intrinsic deformation (median), for context
  -> classify: hinge (stable axis) / rocking (varying axis) / +deformation.
Everything is in the framework-local frame, so it is reproducible & comparable across TCRs.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch, CDRS


def characterize(loop, R, nstem, cstem):
    T, N, _ = loop.shape
    iu = np.triu_indices(N, 1)
    dLL = np.linalg.norm(loop[:, iu[0]] - loop[:, iu[1]], axis=-1)
    med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))
    Rref = R[med] - R[med].mean(0); Rc = R[med].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + Rc
    L0 = Lf[med]; L0c = L0 - L0.mean(0); L0m = L0.mean(0)
    D_int = np.sqrt(((dLL - dLL[med]) ** 2).mean(1))
    omega = np.empty((T, 3)); A = np.zeros((3, 3)); b = np.zeros(3)
    for t in range(T):
        Qt = kabsch(L0c, Lf[t] - Lf[t].mean(0))
        omega[t] = Rot.from_matrix(Qt).as_rotvec()
        ct = Lf[t].mean(0) - Qt @ L0m                                  # translation; fixed point (I-Q)p=ct
        M = np.eye(3) - Qt; A += M.T @ M; b += M.T @ ct
    theta = np.degrees(np.linalg.norm(omega, axis=1))
    P1 = np.linalg.eigvalsh((omega.T @ omega) / T)[::-1]; P1 = P1[0] / P1.sum()   # axis-cloud dimensionality
    p = np.linalg.lstsq(A, b, rcond=None)[0]                            # screw-axis location (pivot)
    Nc = L0[nstem].mean(0); Cc = L0[cstem].mean(0); ax = Cc - Nc
    frac = float(np.dot(p - Nc, ax) / np.dot(ax, ax))
    return dict(theta=float(np.median(theta)), theta_p95=float(np.percentile(theta, 95)),
                P1=float(P1), frac=frac, D_int=float(np.median(D_int)))


def main(sysid):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json")); RSET = {"A": rig["chain_A"]["ultra_rigid"], "B": rig["chain_B"]["ultra_rigid"]}
    CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}
    print(f"{sysid}   (framework-frame rigid-motion vector per CDR)")
    print(f"{'CDR':8}{'theta_med':>10}{'theta_p95':>10}{'P1_axis':>9}{'pivotFrac':>11}{'D_int':>7}  verdict")
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        ri = np.array([imap[ch][r] for r in RSET[ch] if r in imap[ch]])
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        nstem = [lk.index(k) for k in lk[:2]]; cstem = [lk.index(k) for k in lk[-2:]]
        c = characterize(loop, xyz[:, ri], nstem, cstem)
        single = "single-axis" if c["P1"] > 0.8 else "multi-axis/rocking"
        deform = " +deform" if c["D_int"] > 0.45 else ""
        side = CLAMP[cdr[2:]]; on = (c["frac"] < 0.5) == (side == "N")
        print(f"{cdr:8}{c['theta']:9.1f}°{c['theta_p95']:9.1f}°{c['P1']:9.2f}{c['frac']:11.2f}{c['D_int']:7.2f}  "
              f"{single}{deform}; pivot {'ON' if on else 'off'} {side}-clamp")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3QH3")
