"""Rotation extraction for the CDR hinge (updated-plan Steps 1-2).

Coordinate superposition IS used here -- to recover the physical rigid rotation of the loop relative to the
framework, referenced to the ENSEMBLE MEAN (generalized-Procrustes template + Karcher/Frechet mean rotation),
never to an arbitrary frame 0. The downstream flexibility decomposition (graph_decomp) is alignment-free.

Pipeline:
  to_framework_frame : per-frame Kabsch onto the rigid framework -> removes global tumbling (framework frame).
  gpa                : iterative generalized Procrustes of the loop -> ensemble-mean template, per-frame loop
                       rotation R_t, and the Cartesian per-frame deformation residual (rigid-fit residual).
  karcher_mean       : Frechet mean rotation Rbar on SO(3).
  relative_rotvecs   : omega_t = log(Rbar^T R_t) = theta_t * u_hat_t  (excursion from the mean pose).
  consensus_axis     : principal hinge axis u* (top PC of the omega_t); signed angle = omega_t . u*.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def kabsch(P, Q):
    """rigid R,t with Q ~= P@R.T + t (rows). Proper rotation (reflection-safe)."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Q.mean(0) - P.mean(0) @ R.T


def to_framework_frame(loop, fw, fw_ref=None):
    """Align each frame's framework onto fw_ref (default frame 0); apply to the loop. (T,N,3),(T,M,3)->(T,N,3).
    Removes global tumbling. The choice of fw_ref does not affect downstream relative rotations (proved in tests)."""
    fw_ref = fw[0] if fw_ref is None else fw_ref
    out = np.empty_like(loop)
    for t in range(len(loop)):
        R, tt = kabsch(fw[t], fw_ref)
        out[t] = loop[t] @ R.T + tt
    return out


def gpa(X, iters=30, tol=1e-5):
    """Generalized Procrustes on X (T,N,3, already in a common frame), VECTORIZED (batched Kabsch). Returns:
      template (N,3)  = ensemble-mean rigid shape,
      R (T,3,3)       = per-frame loop orientation vs the template (X_t ~= tmpl @ R_t.T + q_t),
      resid (T,)      = per-frame Cartesian rigid-fit RMSD (A) = deformation the rigid model cannot explain."""
    tmpl = X[0] - X[0].mean(0)
    Xc = X - X.mean(1, keepdims=True)                              # (T,N,3) centered
    eye = np.eye(3)[None]
    R = np.repeat(eye, len(X), 0)
    for _ in range(iters):
        H = np.einsum("ni,tnj->tij", tmpl, Xc)                    # (T,3,3) = tmpl^T Xc per frame
        U, _, Vt = np.linalg.svd(H)
        VtT = np.transpose(Vt, (0, 2, 1)); UT = np.transpose(U, (0, 2, 1))
        d = np.sign(np.linalg.det(np.matmul(VtT, UT)))
        Dm = np.repeat(eye, len(X), 0); Dm[:, 2, 2] = d
        R = np.matmul(np.matmul(VtT, Dm), UT)                     # Xc_t ~= tmpl @ R_t.T
        aligned = np.einsum("tni,tij->tnj", Xc, R)               # bring frames back to template frame
        new = aligned.mean(0); new -= new.mean(0)
        shift = float(np.sqrt(((new - tmpl) ** 2).sum())); tmpl = new
        if shift < tol:
            break
    fit = np.einsum("ni,tji->tnj", tmpl, R)                       # tmpl @ R_t.T
    fit = fit + (X.mean(1, keepdims=True) - fit.mean(1, keepdims=True))
    resid = np.sqrt(((X - fit) ** 2).sum(-1).mean(-1))
    return tmpl, R, resid


def karcher_mean(R, iters=100, tol=1e-12):
    """Frechet/Karcher mean rotation on SO(3) (geodesic). Init from the middle frame."""
    Rm = Rot.from_matrix(R)
    Rbar = Rot.from_matrix(R[len(R) // 2])
    for _ in range(iters):
        w = (Rbar.inv() * Rm).as_rotvec().mean(0)
        Rbar = Rbar * Rot.from_rotvec(w)
        if np.linalg.norm(w) < tol:
            break
    return Rbar


def relative_rotvecs(R):
    """omega_t = log(Rbar^T R_t), the rotation excursion of each frame from the Karcher mean pose. (T,3)."""
    Rbar = karcher_mean(R)
    return (Rbar.inv() * Rot.from_matrix(R)).as_rotvec()


def consensus_axis(omega):
    """Principal hinge axis u* = top PC of the rotation vectors; signed so most excursions project positive.
    Returns (u_hat (3,), var_fracs (3,)). Signed hinge angle for frame t is then omega_t . u_hat (radians)."""
    _, S, Vt = np.linalg.svd(omega - omega.mean(0), full_matrices=False)
    u = Vt[0]
    if (omega @ u).sum() < 0:
        u = -u
    return u, (S ** 2) / (S ** 2).sum()


def fit_pivot(X_fwf, q, u, ref):
    """Consensus hinge-axis POSITION p* (framework frame): the point about which rotating the UNDISTORTED
    reference loop `ref` by q_t around u best reproduces the observed loop. Solve
    X_t[n] - R_t ref[n] = (I - R_t) p*  (least squares). `ref` (N,3) must be the undistorted mean-pose loop
    (Karcher-oriented GPA template), NOT the coordinate mean (which shrinks). Returns p* (3,)."""
    Rt = Rot.from_rotvec(q[:, None] * u[None, :]).as_matrix()      # (T,3,3)
    IR = np.eye(3)[None] - Rt                                       # (T,3,3)
    b = X_fwf - np.einsum("tij,nj->tni", Rt, ref)                  # (T,N,3) = IR @ p per (t,n)
    N = X_fwf.shape[1]
    AtA = N * np.einsum("tji,tjk->ik", IR, IR)
    Atb = np.einsum("tji,tnj->i", IR, b)
    if np.linalg.cond(AtA) > 1e10:                                 # near-zero rotation -> position ill-defined
        return ref.mean(0)
    return np.linalg.solve(AtA, Atb)
