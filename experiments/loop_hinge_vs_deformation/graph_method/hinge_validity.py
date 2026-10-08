"""STANDALONE single-hinge validity diagnostic -- run UPSTREAM of the joint d_LF = g(q)+Bz+eps model.

Question: can the rigid-like motion of this CDR be represented by rotation about ONE fixed axis?
Evaluated only on LOW-DEFORMATION frames (so deformation does not contaminate the rigid-rotation cloud).
Does NOT use the joint model -> the joint model is not used to validate its own hinge assumption.

Report per CDR:
  P_1D              lambda1/(sum lambda)  of the rotation-vector covariance (1 => rotations along one axis)
  r_perp            fraction of rotational variation off the PC1 axis  (~ 1 - P_1D)
  delta_rms_deg     exact off-axis rotational error: RMS geodesic angle between R_t and the single-axis fit
  hinge_excursion   RMS |q_t| (deg)
  median/p95 alpha  angle between per-frame instantaneous axis and the consensus axis u*
  E_free, E_hinge   loop->framework distance reconstruction error, free 3D rotation vs single fixed axis
  dE_axis           E_hinge - E_free  (~0 => one axis is enough)
  angle_to_anchor   angle between u* and the loop anchor-anchor line (optional structural sanity check)
  block_axis_agree  median pairwise angle between per-block consensus axes (temporal/replica reproducibility)
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import rotations as RO
import graph_fit as GF


def _dLL(loop):
    N = loop.shape[1]; iu = np.triu_indices(N, 1)
    return GF.dmat(loop)[:, iu[0], iu[1]]


def _pc_axis(omega):
    """PC1 of the rotation-vector cloud + P_1D. omega already Karcher-referenced (mean ~ 0)."""
    Cw = omega.T @ omega / len(omega)
    lam, vec = np.linalg.eigh(Cw)                          # ascending
    P1D = float(lam[-1] / lam.sum())
    u = vec[:, -1]
    if (omega @ u).sum() < 0:
        u = -u
    return u, P1D, lam[::-1]


def single_hinge_report(loop, fw, fw_ref=None, clean_frac=0.3, n_blocks=4, u_anchor=None):
    ref = fw[0] if fw_ref is None else fw_ref
    Lf = RO.to_framework_frame(loop, fw, ref)
    tmpl, R, gpa_resid = RO.gpa(Lf)
    D = np.sqrt(((_dLL(loop) - _dLL(loop).mean(0)) ** 2).mean(1))
    clean = D <= np.quantile(D, clean_frac)
    if clean.sum() < 10:
        clean = np.ones(len(D), bool)
    Rc = R[clean]; Lfc = Lf[clean]

    Rbar = RO.karcher_mean(Rc)
    R_rel = Rbar.inv() * Rot.from_matrix(Rc)                        # Karcher-relative rotations
    omega = R_rel.as_rotvec()                                       # (nc,3)
    u, P1D, _ = _pc_axis(omega)
    q = omega @ u
    r_perp = float(((omega - q[:, None] * u) ** 2).sum() / (omega ** 2).sum())

    # exact off-axis rotational error: geodesic angle between the single-axis fit and the ACTUAL relative rotation
    delta = (Rot.from_rotvec(q[:, None] * u).inv() * R_rel).magnitude()   # rad
    delta_rms = float(np.degrees(np.sqrt((delta ** 2).mean())))
    hinge_excursion = float(np.degrees(np.sqrt((q ** 2).mean())))

    mag = np.linalg.norm(omega, axis=1); big = mag > np.radians(2)
    alpha = np.degrees(np.arccos(np.clip(np.abs((omega[big] / mag[big, None]) @ u), 0, 1))) if big.any() else np.array([np.nan])
    med_alpha, p95_alpha = float(np.median(alpha)), float(np.percentile(alpha, 95))

    # reconstruction: free 3D rigid fit vs single fixed-axis hinge, on clean frames
    Ref = Rbar.apply(tmpl) + Lfc.mean(0).mean(0)
    p = RO.fit_pivot(Lfc, q, u, Ref)

    def dcross(X):
        return np.linalg.norm(X[:, :, None, :] - ref[None, None, :, :], axis=-1).reshape(len(X), -1)
    dobs = dcross(Lfc)
    recon_free = np.stack([(Ref @ RO.kabsch(Ref, Lfc[i])[0].T + RO.kabsch(Ref, Lfc[i])[1]) for i in range(len(Lfc))])
    E_free = float(np.sqrt(((dcross(recon_free) - dobs) ** 2).mean()))
    Rq = Rot.from_rotvec(q[:, None] * u).as_matrix()
    recon_h = p[None, None, :] + np.einsum("tij,nj->tni", Rq, Ref - p)
    E_hinge = float(np.sqrt(((dcross(recon_h) - dobs) ** 2).mean()))

    ang_anchor = (float(np.degrees(np.arccos(np.clip(abs(u @ (u_anchor / np.linalg.norm(u_anchor))), 0, 1))))
                  if u_anchor is not None else np.nan)

    # temporal reproducibility: per-block consensus axes (proxy for replica reproducibility)
    us = []
    for bl in np.array_split(np.arange(len(Lf)), n_blocks):
        Rb = R[bl]; ob = (RO.karcher_mean(Rb).inv() * Rot.from_matrix(Rb)).as_rotvec()
        us.append(_pc_axis(ob)[0])
    us = np.array(us)
    ang = [np.degrees(np.arccos(np.clip(abs(us[a] @ us[b]), 0, 1)))
           for a in range(len(us)) for b in range(a + 1, len(us))]
    block_axis_agree = float(np.median(ang)) if ang else np.nan

    return dict(P_1D=P1D, r_perp=r_perp, delta_rms_deg=delta_rms, hinge_excursion_deg=hinge_excursion,
                median_alpha_deg=med_alpha, p95_alpha_deg=p95_alpha, E_free=E_free, E_hinge=E_hinge,
                dE_axis=float(E_hinge - E_free), angle_to_anchor_deg=ang_anchor,
                block_axis_agree_deg=block_axis_agree, u=u, p=p, n_clean=int(clean.sum()))
