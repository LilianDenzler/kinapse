"""Alignment-free distance-space decomposition with a JOINT hinge+deformation model.

d_LF(t) = g(q_t) + B z_t + eps_t
  g(q_t)  : EXACT nonlinear loop->framework distances from rotating the rigid mean loop by finite angle q_t
            about the FROZEN consensus hinge axis L*=(p*,u*) (1 rigid DOF).
  z_t     : internal-deformation scores = PCA of centered loop->loop distances d_LL(t). A rigid hinge cannot
            change d_LL, so z_t is an INDEPENDENT observable of deformation -> B z_t is deformation's footprint
            on loop->framework distances.
  eps_t   : residual.
Fitting q_t and B JOINTLY (alternating least squares) stops a rotation-mimicking deformation from being
swallowed by the hinge (the GPA failure mode): its d_LL signal forces it into B z_t, not q_t.

L* is estimated from the CLEANEST frames (low D_deform, low GPA residual) and then frozen. Rotation discovery
uses coordinate GPA (rotations.py); the decomposition here is a function of distances only.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import rotations as RO


def dLF_vector(loop, fw):
    d = np.linalg.norm(loop[:, :, None, :] - fw[:, None, :, :], axis=-1)
    return d.reshape(len(loop), -1)


def dLL_pairs(loop):
    N = loop.shape[1]; iu = np.triu_indices(N, 1)
    d = np.linalg.norm(loop[:, :, None, :] - loop[:, None, :, :], axis=-1)
    return d[:, iu[0], iu[1]]


def deformation_scores(dLL, var=0.95, kmax=8, tiny=1e-6):
    """PCA of centered loop->loop distances -> deformation scores z (T,k). Empty if the loop is ~rigid."""
    dc = dLL - dLL.mean(0)
    if (dc ** 2).sum() < tiny:
        return np.zeros((len(dLL), 0)), np.zeros((0, dLL.shape[1]))
    U, S, Vt = np.linalg.svd(dc, full_matrices=False)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()
    k = min(int(np.searchsorted(cum, var)) + 1, kmax, len(S))
    return U[:, :k] * S[:k], Vt[:k]


def g_of_q(q, Xbar, F_ref, u, p):
    """EXACT rigid-hinge loop->framework distances: rotate Xbar by finite q about (p,u). q (G,) -> (G, N*M)."""
    Rt = Rot.from_rotvec(np.asarray(q)[:, None] * u[None, :]).as_matrix()
    loop = p[None, None, :] + np.einsum("gij,nj->gni", Rt, Xbar - p)
    d = np.linalg.norm(loop[:, :, None, :] - F_ref[None, None, :, :], axis=-1)
    return d.reshape(len(Rt), -1)


def joint_fit(dLF, z, Xbar, F_ref, u, p, q_init, iters=10, n_grid=401, pad=np.radians(25)):
    """Alternating LS for d_LF = g(q) + B z. Returns q (T,), B (k,ne), g (T,ne), Bz (T,ne)."""
    q = np.asarray(q_init, float).copy()
    span = max(np.abs(q).max() + pad, np.radians(20))
    grid = np.linspace(-span, span, n_grid)
    G = g_of_q(grid, Xbar, F_ref, u, p)                       # (n_grid, ne)
    GG = (G ** 2).sum(1)                                      # (n_grid,)
    B = np.zeros((z.shape[1], dLF.shape[1]))
    for _ in range(iters):
        g = g_of_q(q, Xbar, F_ref, u, p)
        if z.shape[1]:
            B = np.linalg.lstsq(z, dLF - g, rcond=None)[0]    # rho ~ z @ B
        target = dLF - (z @ B if z.shape[1] else 0.0)         # hinge-only target
        score = GG[None, :] - 2.0 * (target @ G.T)            # (T, n_grid) argmin == min ||target - G||^2
        q = grid[score.argmin(1)]
    g = g_of_q(q, Xbar, F_ref, u, p)
    Bz = z @ B if z.shape[1] else np.zeros_like(dLF)
    return q, B, g, Bz


def run_cdr(loop, fw, fw_ref=None, clean_frac=0.3, kmax=8):
    """Full per-CDR decomposition from lab-frame loop (T,N,3) and framework (T,M,3) coordinates.
    Returns the flexibility fingerprint and diagnostics. Distances-only decomposition; GPA for axis discovery."""
    ref = fw[0] if fw_ref is None else fw_ref
    Lf = RO.to_framework_frame(loop, fw, ref)                 # framework frame (removes global tumbling)
    tmpl, R, gpa_resid = RO.gpa(Lf)                           # undistorted mean shape + per-frame rotation + resid
    # any-axis rigid baseline: best rigid fit per frame (FULL R_t) -> its loop->framework distances
    recon_free = np.einsum("ni,tji->tnj", tmpl, R)
    recon_free = recon_free + (Lf.mean(1, keepdims=True) - recon_free.mean(1, keepdims=True))
    dLF_free = np.linalg.norm(recon_free[:, :, None, :] - ref[None, None, :, :], axis=-1).reshape(len(Lf), -1)
    Rbar = RO.karcher_mean(R)
    Ref = Rbar.apply(tmpl) + Lf.mean(0).mean(0)              # UNDISTORTED loop at the mean pose (no shrinkage)
    w = RO.relative_rotvecs(R)                                # Karcher-referenced rotation vectors

    dLL = dLL_pairs(loop); Dc = dLL - dLL.mean(0)
    D = np.sqrt((Dc ** 2).mean(1))                            # per-frame internal deformation
    clean = D <= np.quantile(D, clean_frac)                  # cleanest (least-deformed) frames for the axis
    if clean.sum() < 5:
        clean = np.ones(len(D), bool)
    u, axis_var = RO.consensus_axis(w[clean])                 # freeze axis from clean frames
    q_gpa = w @ u
    p = RO.fit_pivot(Lf[clean], q_gpa[clean], u, Ref)        # freeze pivot from clean frames (undistorted ref)

    z, _ = deformation_scores(dLL, kmax=kmax)
    dLF = dLF_vector(loop, fw)
    q, B, g, Bz = joint_fit(dLF, z, Ref, ref, u, p, q_gpa)

    ne = dLF.shape[1]; s = np.sqrt(ne)                        # per-edge normalization
    obs = (dLF - dLF.mean(0)) / s
    dg = (g - g.mean(0)) / s                                  # centered exact hinge
    dbz = (Bz - Bz.mean(0)) / s                               # centered deformation footprint
    eps = obs - dg - dbz                                      # residual
    tot = (obs ** 2).sum()
    f_hinge = float((dg ** 2).sum() / tot)
    f_deform_LF = float((dbz ** 2).sum() / tot)
    f_residual = float((eps ** 2).sum() / tot)
    f_coupling = float(1.0 - f_hinge - f_deform_LF - f_residual)   # cross-terms (mostly hinge<->deformation)
    dfree = (dLF_free - dLF_free.mean(0)) / s                      # any-axis rigid baseline
    f_rigid_free = float((dfree ** 2).sum() / tot)                 # fraction along ANY rigid rotation (multi-axis)
    # organized by graph space (LL vs LF); F_deform lives in LL space, the rest in LF space
    return dict(
        # --- Intrinsic loop deformation (LL graph space, from d_LL) ---
        F_deform=float(np.sqrt((D ** 2).mean())),
        # --- Framework-relative decomposition (LF graph space, Angstrom, per-edge) ---
        F_hinge=float(np.sqrt((dg ** 2).sum(1).mean())),
        F_deform_LF=float(np.sqrt((dbz ** 2).sum(1).mean())),
        F_residual=float(np.sqrt((eps ** 2).sum(1).mean())),
        # --- Variance attribution of the LF motion (sum to 1) ---
        f_hinge=f_hinge, f_deform_LF=f_deform_LF, f_coupling=f_coupling, f_residual=f_residual,
        f_rigid_free=f_rigid_free,                           # any-axis rigid baseline (gap to f_hinge = multi-axis)
        # --- Mechanics ---
        theta_deg=np.degrees(q), u=u, p=p, axis_var=axis_var,        # q_t gives P(q) -> F(q) = -kT ln P(q)
        gpa_resid=gpa_resid, n_deform_modes=int(z.shape[1]), clean_n=int(clean.sum()),
    )
