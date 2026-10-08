"""Alignment-free elasticity primitives for CDR loops — three formalisms.

Every observable here is a function of Cartesian CA coordinates but INVARIANT to
any rigid motion (rotation + translation) of any frame — no superposition is ever
performed. By construction, rigid motion is the ZERO-STRAIN NULL SPACE of every
method (see plan.md "rigid = the zero-strain null space"):

  Method 1  finite-strain field   rigid  <=>  E_i = 0, D2min = 0
  Method 2  elastic network       rigid  <=>  the zero modes of Gamma / H
  Method 3  data-driven stiffness rigid  <=>  zero variance of d_LL

Units: coordinates in Angstrom (mdtraj xyz is nm -> multiply by 10 upstream).
"""
from __future__ import annotations
import numpy as np

# ======================================================================= util
def dmat(X):
    """(...,N,3) -> (...,N,N) pairwise Euclidean distances."""
    diff = X[..., :, None, :] - X[..., None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def cross_dmat(X, A):
    """loop (...,N,3) x anchors (...,M,3) -> (...,N,M) cross distances."""
    diff = X[..., :, None, :] - A[..., None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def loop_pairs(N):
    """upper-triangle (i<j) index arrays for an N-node loop (the d_LL edges)."""
    i, j = np.triu_indices(N, k=1)
    return i, j


def medoid_reference(traj_X):
    """Invariant reference-frame choice: the frame whose intra-loop distance
    matrix is closest to the ensemble-mean distance matrix (the d_LL medoid).
    traj_X: (T,N,3) -> int frame index. Alignment-free (uses d_LL only)."""
    D = dmat(traj_X)                      # (T,N,N)
    Dbar = D.mean(0)                      # (N,N)
    dev = ((D - Dbar) ** 2).sum(axis=(1, 2))
    return int(np.argmin(dev))


# ============================================== Method 1: finite-strain field
def neighborhoods(Xref, rc, kmin):
    """For each node, indices of reference neighbors within rc (excluding self),
    expanding rc until at least kmin neighbors are present. Returns list length N."""
    D = dmat(Xref)
    N = Xref.shape[0]
    out = []
    for i in range(N):
        r = rc
        while True:
            js = np.where((D[i] < r) & (np.arange(N) != i))[0]
            if len(js) >= kmin or r > D[i].max() + 1e-6:
                break
            r *= 1.25
        out.append(js)
    return out


def strain_field(Xref, traj_X, neigh):
    """Per-node finite (Green-Lagrange) strain and non-affinity for every frame,
    referenced to Xref. Vectorized over frames.

    Xref: (N,3) reference; traj_X: (T,N,3); neigh: list of neighbor-index arrays.
    Returns dict of (T,N) arrays: vol=tr(E), shear=||dev E||_F, vm=von-Mises,
    pmax/pmin=extreme principal strains, d2min=non-affine residual.
    All entries are invariant to rigid motion (and reflection) of any frame.
    """
    T = traj_X.shape[0]
    N = Xref.shape[0]
    I3 = np.eye(3)
    vol = np.zeros((T, N)); shear = np.zeros((T, N)); vm = np.zeros((T, N))
    pmax = np.zeros((T, N)); pmin = np.zeros((T, N)); d2 = np.zeros((T, N))
    for i in range(N):
        js = neigh[i]
        D = Xref[js] - Xref[i]                      # (k,3) reference bonds
        Vinv = np.linalg.pinv(D.T @ D)              # (3,3)
        d = traj_X[:, js, :] - traj_X[:, i, :][:, None, :]   # (T,k,3) current bonds
        W = np.einsum("tka,kb->tab", d, D)          # (T,3,3)  sum_j d_j D_j^T
        F = W @ Vinv                                # (T,3,3)  F = W V^{-1}
        C = np.einsum("tba,tbc->tac", F, F)         # (T,3,3)  F^T F
        E = 0.5 * (C - I3)                           # (T,3,3)  Green-Lagrange strain
        tr = np.trace(E, axis1=1, axis2=2)          # (T,)
        dev = E - (tr / 3.0)[:, None, None] * I3
        vol[:, i] = tr
        shear[:, i] = np.sqrt((dev * dev).sum(axis=(1, 2)))
        vm[:, i] = np.sqrt((2.0 / 3.0) * (dev * dev).sum(axis=(1, 2)))
        w = np.linalg.eigvalsh(E)                    # (T,3) ascending
        pmin[:, i] = w[:, 0]; pmax[:, i] = w[:, -1]
        # non-affine residual: d_j - F D_j  (F applied to reference bond)
        res = d - np.einsum("kb,tab->tka", D, F)    # (T,k,3)
        d2[:, i] = np.mean(np.sum(res * res, axis=2), axis=1)
    return dict(vol=vol, shear=shear, vm=vm, pmax=pmax, pmin=pmin, d2min=d2)


# ================================================ Method 2: elastic network
def gnm_kirchhoff(X, rc):
    """GNM Kirchhoff (connectivity Laplacian) from the CA contact map. (N,3)->(N,N).
    Function of distances only -> rigid- and reflection-invariant."""
    D = dmat(X)
    A = ((D < rc) & (D > 1e-9)).astype(float)
    G = -A
    np.fill_diagonal(G, 0.0)
    np.fill_diagonal(G, -G.sum(axis=1))
    return G


def gnm_modes(G, tol=1e-9):
    """Eigendecomposition of Gamma. Returns (w, V) ascending; the single ~0 mode
    is the rigid (uniform-shift) mode."""
    w, V = np.linalg.eigh(G)
    return w, V


def gnm_msf(G, kT=1.0, tol=1e-9):
    """Per-residue mean-square fluctuation (Gamma^{-1})_ii and the full inverse,
    dropping the zero mode(s). <dR_i.dR_j> = kT (Gamma^+)_ij."""
    w, V = np.linalg.eigh(G)
    inv = np.zeros_like(G)
    for k in range(len(w)):
        if w[k] > tol:
            inv += np.outer(V[:, k], V[:, k]) / w[k]
    return kT * np.diag(inv), kT * inv


def anm_hessian(X, rc, gamma=1.0):
    """ANM Hessian (3N x 3N) from the contact map. Modes are Cartesian directions
    (compare only in a common/framework frame); eigenvalues/mobilities invariant.
    6 zero modes = rigid-body."""
    N = X.shape[0]
    H = np.zeros((3 * N, 3 * N))
    D = dmat(X)
    for i in range(N):
        for j in range(N):
            if i == j or D[i, j] >= rc or D[i, j] < 1e-9:
                continue
            dvec = (X[j] - X[i])[:, None]
            k = gamma * (dvec @ dvec.T) / (D[i, j] ** 2)     # 3x3 superelement
            H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = -k
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] += k
    return H


def mode_overlap(u, v):
    """|cos| overlap between two (normalized-agnostic) mode vectors."""
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return abs(float(u @ v))


# ============================================ Method 3: data-driven stiffness
def dll_series(traj_X):
    """Per-frame intra-loop CA-CA distances (the complete invariant deformation
    set). traj_X:(T,N,3) -> d:(T,P), pairs (i,j). Rigid motion => constant d."""
    i, j = loop_pairs(traj_X.shape[1])
    d = np.linalg.norm(traj_X[:, i, :] - traj_X[:, j, :], axis=-1)   # (T,P)
    return d, (i, j)


def compliance(d):
    """Covariance Sigma of the distance fluctuations (compliance-first, robust).
    d:(T,P) -> Sigma:(P,P), mean:(P,). Rigid motion => Sigma = 0."""
    return np.cov(d.T), d.mean(0)


def stiffness(Sigma, kT, ridge=0.0):
    """Effective stiffness K_eff = kT * Sigma^{-1} (regularized)."""
    S = Sigma + ridge * np.eye(len(Sigma))
    return kT * np.linalg.pinv(S)


def soft_modes(Sigma):
    """Eigendecomposition of the compliance; large eigenvalue = soft mode.
    Returns (w desc, V) with columns the mode footprints on the distance edges."""
    w, V = np.linalg.eigh(Sigma)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def quasiharmonic_entropy(Sigma_internal, kT):
    """Quasi-harmonic configurational entropy (nats) S ~ 1/2 ln det(2*pi*e*Sigma)
    from a MINIMAL (non-redundant) internal-coordinate covariance. Do NOT feed the
    redundant full d_LL set (P > 3N-6) — use the strain/internal coords."""
    sign, logdet = np.linalg.slogdet(2 * np.pi * np.e * Sigma_internal)
    return 0.5 * logdet if sign > 0 else float("nan")


def qh_entropy_from_cov(Sigma, ndof=None, tol=1e-9):
    """Redundancy-safe quasi-harmonic entropy (nats): sum 1/2 ln(2*pi*e*lambda_k)
    over the ndof largest eigenvalues > tol. The full d_LL covariance is rank-
    deficient (rank <= 3N-6); pass ndof=3N-6 to count only the real internal DOF."""
    w = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    w = w[w > tol]
    if ndof is not None:
        w = w[:ndof]
    return float(np.sum(0.5 * np.log(2 * np.pi * np.e * w)))


def edge_mode_to_residue(vec, pairs, N):
    """Map a compliance/soft-mode footprint over d_LL EDGES to a per-RESIDUE score
    (sum of squared edge weights touching each residue). Returns (N,) normalized."""
    i, j = pairs
    r = np.zeros(N)
    np.add.at(r, i, vec ** 2)
    np.add.at(r, j, vec ** 2)
    return r / (r.sum() + 1e-12)


def participation_ratio(w):
    """Participation ratio of a nonneg weight vector: ~1 => localized on one node,
    ~N => evenly spread. w need not be normalized."""
    p = w / (w.sum() + 1e-12)
    return float(1.0 / np.sum(p ** 2))


# ================================ optional: rigid hinge angle (Eckart frame)
def kabsch_R(P, Q):
    """Rotation mapping P onto Q (both (N,3), centered internally). Used ONLY to
    report an optional hinge ANGLE in the Eckart/best-fit frame — never in a
    deformation metric. Deformation is measured superposition-free above."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    Hh = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(Hh)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dd = np.diag([1.0, 1.0, d])
    return Vt.T @ Dd @ U.T


def hinge_angle(Xref, Xcur):
    """Best-fit (Eckart) rotation angle of the reference loop shape onto a frame,
    in degrees. This is the OPTIONAL 'how much did it also swing' number; pair it
    with the strain-field D2min / Sigma residual as the honest deformation part."""
    R = kabsch_R(Xref, Xcur)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
