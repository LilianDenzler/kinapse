#!/usr/bin/env python
"""Validation of the hinge/deform math (plan.md 'Validation'). No MD needed."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np
import compute_hinge_deform as H

rng = np.random.default_rng(0)


def rand_rot(angle_deg, ax=None):
    ax = ax if ax is not None else rng.normal(size=3)
    ax = ax / np.linalg.norm(ax)
    th = np.radians(angle_deg)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


F, N, M = 200, 12, 40
loop0 = rng.normal(scale=5, size=(N, 3))
core0 = rng.normal(scale=12, size=(M, 3))                    # non-coplanar anchors

# ---- 1. SE(3)+reflection invariance of D_deform / D_framework -------------
Xl = np.repeat(loop0[None], F, 0) + rng.normal(scale=0.1, size=(F, N, 3))
Ac = np.repeat(core0[None], F, 0)
d0 = H.rms_delta(H.dist_mat(Xl))
f0 = H.rms_delta(H.cross_dist(Xl, Ac))
Xl2, Ac2 = Xl.copy(), Ac.copy()
for k in range(F):                                           # per-frame random rigid move + reflect
    R = rand_rot(rng.uniform(0, 360)); t = rng.normal(scale=50, size=3)
    if k % 2 == 0: R = R @ np.diag([1, 1, -1])               # throw in reflections
    Xl2[k] = Xl[k] @ R.T + t; Ac2[k] = Ac[k] @ R.T + t
d1 = H.rms_delta(H.dist_mat(Xl2)); f1 = H.rms_delta(H.cross_dist(Xl2, Ac2))
print(f"[1] invariance  max|dD_deform|={np.abs(d0-d1).max():.2e}  max|dD_frame|={np.abs(f0-f1).max():.2e}")
assert np.abs(d0 - d1).max() < 1e-6 and np.abs(f0 - f1).max() < 1e-6

# ---- 2. pure hinge: theta recovered, E_nonrigid ~ 0, D_deform ~ 0 ----------
angles = rng.uniform(0, 40, F); ax = rng.normal(size=3)
loop_scaf = np.stack([(loop0 - loop0.mean(0)) @ rand_rot(a, ax).T + loop0.mean(0) for a in angles])
h = H.hinge_fit(loop_scaf)
Xl_h = loop_scaf                                             # distances from a pure rotation
print(f"[2] pure hinge  E_nonrigid={h['E_nonrigid'].mean():.2e} A  D_deform={H.rms_delta(H.dist_mat(Xl_h)).mean():.2e} A  theta_mean={h['theta'].mean():.1f} (inj {angles.mean():.1f})")
assert h["E_nonrigid"].max() < 1e-4 and H.rms_delta(H.dist_mat(Xl_h)).max() < 1e-4

# recovery of the exact angle vs a known reference (single frame, ref = frame 0)
ang = 25.0
two = np.stack([loop0, (loop0 - loop0.mean(0)) @ rand_rot(ang, ax).T + loop0.mean(0)])
th = H.hinge_fit(two)["theta"].max()            # medoid is one frame (theta 0); the other = 25
print(f"[2b] exact recovery: injected {ang}  recovered {th:.3f}")
assert abs(th - ang) < 1e-2

# ---- 3. pure deformation: theta ~ 0, E_nonrigid > 0, D_deform > 0 ----------
loop_def = np.repeat(loop0[None], F, 0).copy()
loop_def[:, N // 2] += np.linspace(0, 3, F)[:, None] * np.array([1, 0, 0.3])   # one atom drifts out
h3 = H.hinge_fit(loop_def)
dd = H.rms_delta(H.dist_mat(loop_def))
print(f"[3] pure deform theta_mean={h3['theta'].mean():.2f} deg  E_nonrigid={h3['E_nonrigid'].mean():.3f} A  D_deform={dd.mean():.3f} A")
assert h3["E_nonrigid"].mean() > 0.1 and dd.mean() > 0.1

# ---- 4. anchor sufficiency: coplanar/too-few anchors miss rotation ---------
# (informational) 3 collinear anchors cannot localize a loop rotating on their axis
print("[4] anchor rule enforced in config (CORE_MIN_ATOMS>=6, rigid core ~non-coplanar)")

# ---- 5. rigid_core finds a planted rigid subset ---------------------------
rigid = rng.normal(scale=10, size=(15, 3))
Acore = np.repeat(rigid[None], F, 0) + rng.normal(scale=0.05, size=(F, 15, 3))   # rigid
Afloppy = rng.normal(scale=10, size=(F, 5, 3))                                    # independent noise
A = np.concatenate([Acore, Afloppy], axis=1)
idx, mx = H.rigid_core(A, std_thresh=0.8, min_atoms=6)
print(f"[5] rigid_core kept {len(idx)}/20 (planted 15 rigid), max_pair_std={mx:.3f}")
assert set(idx.tolist()) <= set(range(15)) and len(idx) >= 12

print("ALL TESTS PASSED")
