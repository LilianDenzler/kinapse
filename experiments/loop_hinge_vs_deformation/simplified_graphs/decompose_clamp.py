#!/usr/bin/env python
"""Does the CDR rigid motion = rotation about the CLAMP AXIS (the stem-to-stem line)?
A rigid body pinned at 2 points can only rotate about the line through them. So test:
  clamp-hinge : motion explained by rotation about the FIXED stem-to-stem line (ideal double-tether swing)
  off-clamp   : rigid motion NOT about that line (a stem must move -> tether broken/asymmetric clamp)
  deform      : residual after full rigid fit
Also reports angle between the data-derived rotation axis u and the clamp axis c (small => clean tether hinge)."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch
from decompose3 import fit_angle, rot_about

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]


def decompose(loop, R):
    T, N, _ = loop.shape
    Rref = R[0] - R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + R[0].mean(0)
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))
    L0 = Lf[med]; L0m = L0.mean(0); L0c = L0 - L0m
    aN, aC = L0[0], L0[-1]                                 # the two CDR stem endpoints
    c = aC - aN; c /= np.linalg.norm(c)                    # CLAMP AXIS direction (stem-to-stem line)
    m = 0.5 * (aN + aC)                                    # a point on the clamp axis
    # data-derived rotation axis (for the angle comparison)
    om = np.empty((T, 3)); Qs = []
    for t in range(T):
        Q = kabsch(L0c, Lf[t] - Lf[t].mean(0)); Qs.append(Q); om[t] = Rot.from_matrix(Q).as_rotvec()
    u = np.linalg.eigh(om.T @ om)[1][:, -1]; u /= np.linalg.norm(u)
    ang = np.degrees(np.arccos(min(1, abs(u @ c))))       # angle between rotation axis and clamp axis
    Dtot2 = Dcl2 = Drig2 = Ddef2 = 0.0; thc = []
    for t in range(T):
        rigid = L0c @ Qs[t].T + Lf[t].mean(0)             # full 6-DOF rigid placement
        th = fit_angle(L0 - m, Lf[t] - m, c); thc.append(th)
        clamp = m + rot_about(L0 - m, th, c)              # rotation about the fixed clamp-axis LINE
        Dtot2 += ((Lf[t] - L0) ** 2).sum(1).mean()
        Dcl2 += ((clamp - L0) ** 2).sum(1).mean()
        Drig2 += ((rigid - L0) ** 2).sum(1).mean()
        Ddef2 += ((Lf[t] - rigid) ** 2).sum(1).mean()
    off2 = max(Drig2 - Dcl2, 0.0)
    return Dcl2 / Dtot2, off2 / Dtot2, Ddef2 / Dtot2, ang, float(np.degrees(np.ptp(thc))), np.sqrt(Dtot2 / T)


def main(sysid="3SKN"):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    print(f"{sysid} — is the rigid motion rotation about the CLAMP AXIS (stem-to-stem line)?")
    print(f"{'CDR':8}{'clamp-hinge':>12}{'off-clamp':>11}{'deform':>8}{'axis∠clamp':>12}{'θc range':>10}{'Dtot':>7}")
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        R = xyz[:, np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])]
        fcl, fo, fd, ang, thr, dt = decompose(loop, R)
        print(f"{cdr:8}{fcl:11.0%}{fo:11.0%}{fd:8.0%}{ang:10.0f}°{thr:9.0f}°{dt:7.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3SKN")
