"""Alignment-free distance primitives + the pure-distance rigid-loop fit (validated core).

NO alignment in this module. Everything is a function of pairwise distances (SE(3)+reflection invariant).
`distance_rigid_fit` is the pure-distance alternative to the coordinate GPA in rotations.py -- kept as a
drop-in and as the oracle-checked fit. The Kabsch oracle for tests lives only in graph_tests.py.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


def dmat(X):
    """(...,N,3) -> (...,N,N) pairwise distances."""
    return np.linalg.norm(X[..., :, None, :] - X[..., None, :, :], axis=-1)


def cross(X, A):
    """loop (...,N,3), anchors (...,M,3) -> (...,N,M) loop-to-anchor distances."""
    return np.linalg.norm(X[..., :, None, :] - A[..., None, :, :], axis=-1)


def theta_of(R):
    """rotation angle (deg) of a rotation matrix (or stack)."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def framework_rigidity(F_traj):
    """F_traj (T,M,3): max std (over frames) of any framework-framework pairwise distance (A).
    Small => rigid anchor set. Alignment-free rigidity check for SETUP / anchor validation."""
    return float(dmat(F_traj).std(0).max())


def distance_rigid_fit(loop_ref, F_ref, dLF_obs, init=None):
    """Pure-distance rigid-loop fit: T*=argmin_{R,q} sum_ia [ ||R x_i^0 + q - a_a|| - d^LF_ia ]^2.
    loop_ref (N,3), F_ref (M,3) fixed reference-frame coords; dLF_obs (N,M) observed distances.
    Never touches frame-t coordinates. Returns (R, q, result)."""
    def resid(p):
        R = Rot.from_rotvec(p[:3]).as_matrix()
        return (cross(loop_ref @ R.T + p[3:], F_ref) - dLF_obs).ravel()
    x0 = np.zeros(6) if init is None else np.asarray(init, float)
    sol = least_squares(resid, x0, method="lm", max_nfev=2000)
    return Rot.from_rotvec(sol.x[:3]).as_matrix(), sol.x[3:], sol
