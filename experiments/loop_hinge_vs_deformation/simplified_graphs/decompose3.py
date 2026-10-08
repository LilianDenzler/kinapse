#!/usr/bin/env python
"""Three-way split of CDR framework-relative motion (coordinate space, after framework alignment):
  hinge      = rigid rotation ALONG the fixed hinge axis (1 DOF about pivot p, axis u)
  off-axis   = rigid motion NOT along that axis (other rotation axes + translation) = D_rigid^2 - D_hinge^2
  deformation= residual after the full 6-DOF rigid fit
Reported as fractions of total motion (approximately additive: total^2 = hinge^2 + off^2 + deform^2)."""
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

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]


def fit_angle(a, b, u):
    ap = a - (a @ u)[:, None] * u; bp = b - (b @ u)[:, None] * u
    return np.arctan2(np.sum(np.cross(np.broadcast_to(u, ap.shape), ap) * bp), np.sum(ap * bp))


def rot_about(a, th, u):
    c, s = np.cos(th), np.sin(th)
    return c * a + s * np.cross(np.broadcast_to(u, a.shape), a) + (1 - c) * ((a @ u)[:, None] * u)


def decompose(loop, R):
    T, N, _ = loop.shape
    Rref = R[0] - R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + R[0].mean(0)
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))
    L0 = Lf[med]; L0m = L0.mean(0); L0c = L0 - L0m
    # hinge axis + pivot from the rotation cloud
    om = np.empty((T, 3)); A = np.zeros((3, 3)); bb = np.zeros(3); Qs = []
    for t in range(T):
        Q = kabsch(L0c, Lf[t] - Lf[t].mean(0)); Qs.append(Q); om[t] = Rot.from_matrix(Q).as_rotvec()
        c = Lf[t].mean(0) - Q @ L0m; M = np.eye(3) - Q; A += M.T @ M; bb += M.T @ c
    p = np.linalg.lstsq(A, bb, rcond=None)[0]
    u = np.linalg.eigh(om.T @ om)[1][:, -1]; u /= np.linalg.norm(u)
    Dtot2 = Dhin2 = Drig2 = Ddef2 = 0.0
    for t in range(T):
        rigid = (L0c) @ Qs[t].T + Lf[t].mean(0)                        # full 6-DOF rigid placement of L0
        th = fit_angle(L0 - p, Lf[t] - p, u)
        hinge = p + rot_about(L0 - p, th, u)                           # 1-DOF hinge about (u,p)
        Dtot2 += ((Lf[t] - L0) ** 2).sum(1).mean()
        Dhin2 += ((hinge - L0) ** 2).sum(1).mean()
        Drig2 += ((rigid - L0) ** 2).sum(1).mean()
        Ddef2 += ((Lf[t] - rigid) ** 2).sum(1).mean()
    off2 = max(Drig2 - Dhin2, 0.0)
    tot = Dtot2
    return Dhin2 / tot, off2 / tot, Ddef2 / tot, np.sqrt(Dtot2 / T)


def main(sysid="3SKN"):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    print(f"{sysid} — 3-way split of CDR motion (fraction of total framework-relative motion)")
    print(f"{'CDR':8}{'hinge(axis)':>12}{'off-axis rig':>14}{'deform':>9}{'D_total(A)':>11}")
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        R = xyz[:, np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])]
        fh, fo, fd, dt = decompose(loop, R)
        print(f"{cdr:8}{fh:11.0%}{fo:14.0%}{fd:9.0%}{dt:11.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3SKN")
