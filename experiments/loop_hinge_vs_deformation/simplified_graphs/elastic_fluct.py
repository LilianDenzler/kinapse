#!/usr/bin/env python
"""Equilibrium-fluctuation elasticity of a CDR loop, from Cα geometry alone.

Elasticity theory near equilibrium:  U(q) ≈ ½ qᵀ K q, so with an approximately
equilibrium-sampled MD ensemble the effective stiffness is read straight off the
covariance of internal strain coordinates:

    K_eff = kT · Σ_q⁺          Σ_q = ⟨(q−⟨q⟩)(q−⟨q⟩)ᵀ⟩      (⁺ = pseudo-inverse)

Two complementary elastic descriptions, both alignment-free (every coordinate is a
SE(3)-invariant function of the raw Cα xyz — no superposition is ever performed;
rigid motion is the exact zero of every quantity here):

  GLOBAL  normalized-strain compliance/stiffness of the whole loop
            q_ij(t) = (d_ij(t) − ⟨d_ij⟩) / ⟨d_ij⟩            dimensionless strain
            Σ_q = Cov_t[q],   K_eff = kT · Σ_q⁺
            modes  Σ_q v_k = λ_k v_k  →  k_k = kT/λ_k        (soft: large λ)
            scalars  C_mean = tr(Σ_q)/m,  k_soft = kT/λ_max,  K_mean

  LOCAL   discrete-elastic-rod stiffness PROFILE along the backbone
            l_i (stretch), θ_i (bend), φ_i (torsion) internal coordinates
            k_s(i)=kT/Var(l_i)  k_b(i)=kT/Var(θ_i)  k_t(i)=kT/Var_circ(φ_i)

Harmonic-basin caveat (named honestly): kT/Var is an effective equilibrium
stiffness only inside ONE approximately harmonic basin. `basin_bimodality`
flags loops whose softest mode is multi-modal — there K_eff is a LOCAL/effective
stiffness, not a global one.

Units: Cα coords in Å (mdtraj nm × 10 upstream); kT in kcal/mol ⇒ global k_k in
kcal/mol, k_b/k_t in kcal/mol/rad², k_s in kcal/mol/Å².

Pure numpy, importable, self-tested — run `python elastic_fluct.py` for the gate.
"""
from __future__ import annotations
import numpy as np

KT_DEFAULT = 0.001987 * 300.0            # kcal/mol at 300 K (matches ../elasticity/config.py)


# ============================================================ global: strain q
def loop_pairs(N):
    """Upper-triangle (i<j) index arrays — the internal (rigid-invariant) d_LL edges."""
    return np.triu_indices(N, k=1)


def pair_distances(P):
    """P (T,N,3) → d (T,m) pairwise Cα distances, and the (i,j) pair indices."""
    i, j = loop_pairs(P.shape[1])
    d = np.linalg.norm(P[:, i, :] - P[:, j, :], axis=-1)
    return d, (i, j)


def normalized_strain(P):
    """Dimensionless internal strain  q_ij(t)=(d_ij−⟨d_ij⟩)/⟨d_ij⟩.  Rigid motion ⇒ q=0.
    Alignment-free, reference-frame agnostic (the ensemble mean is the equilibrium
    point, not a chosen MD frame). Returns q (T,m), dbar (m,), pairs (i,j)."""
    d, (i, j) = pair_distances(P)
    dbar = d.mean(0)
    return (d - dbar) / dbar, dbar, (i, j)


def compliance(q):
    """Compliance matrix Σ_q = Cov_t[q]  (m,m).  Rigid motion ⇒ Σ_q = 0."""
    return np.cov(q.T)


def spectrum(Sigma, kT=KT_DEFAULT, ndof=None, tol=1e-10):
    """Spectral decomposition of the compliance (the elastic modes).

    Σ_q is symmetric PSD; eigenpairs returned SOFT-first (largest variance λ first).
    A loop has only 3N−6 true internal DOF while q spans m=N(N-1)/2 redundant pairs,
    so Σ_q is rank-deficient — pass ndof=3N−6 to keep only the real modes for the
    rank-sensitive scalars (C_mean uses the full trace and needs no truncation).

    Returns dict:
      lam (desc, clipped≥0), V (columns = modes, matching order), keep (real-mode mask),
      k_mode = kT/λ over kept modes,
      C_mean = tr(Σ_q)/m           mean compliance per strain coordinate (↑ = softer),
      k_soft = kT/λ_max            stiffness of the EASIEST internal deformation,
      K_mean = mean(kT/λ) kept     (stiff-mode dominated — report with care),
      soft_frac1                   variance fraction in the softest mode,
      eff_modes                    participation ratio of the spectrum (1=one soft mode,
                                   →N_real = flexibility spread over many modes),
      n_real                       number of kept (nonzero) modes."""
    w, V = np.linalg.eigh(Sigma)
    order = np.argsort(w)[::-1]                       # soft (large var) first
    w = np.clip(w[order], 0.0, None); V = V[:, order]
    m = len(w)
    lam_max = w[0]
    keep = w > max(tol, tol * lam_max)
    if ndof is not None:
        idx = np.where(keep)[0][:ndof]
        keep = np.zeros(m, bool); keep[idx] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        k_mode = np.where(keep, kT / np.where(w > 0, w, np.nan), np.nan)
    wk = w[keep]; tot = wk.sum()
    return dict(
        lam=w, V=V, keep=keep, k_mode=k_mode,
        C_mean=float(np.trace(Sigma) / m),
        k_soft=float(kT / lam_max) if lam_max > 0 else float("nan"),
        K_mean=float(np.nanmean(k_mode[keep])) if keep.any() else float("nan"),
        soft_frac1=float(wk[0] / tot) if tot > 0 else float("nan"),
        eff_modes=float((wk.sum() ** 2) / (wk ** 2).sum()) if keep.any() else float("nan"),
        n_real=int(keep.sum()),
    )


def stiffness_matrix(Sigma, kT=KT_DEFAULT, tol=1e-10):
    """Effective elastic stiffness  K_eff = kT · Σ_q⁺  (Moore-Penrose; drops the null space)."""
    return kT * np.linalg.pinv(Sigma, rcond=tol)


def mode_residue_footprint(vec, pairs, N):
    """Map a compliance-mode footprint over the q EDGES to a per-RESIDUE weight
    (sum of squared edge weights touching each residue). Says WHERE a mode lives.
    Returns (N,) normalized to sum 1."""
    i, j = pairs
    r = np.zeros(N)
    np.add.at(r, i, vec ** 2)
    np.add.at(r, j, vec ** 2)
    return r / (r.sum() + 1e-12)


# ================================================== local: discrete elastic rod
def bond_lengths(P):
    """Virtual Cα–Cα bonds  l_i(t)=|r_{i+1}−r_i|.  P (T,N,3) → (T,N−1)."""
    return np.linalg.norm(P[:, 1:, :] - P[:, :-1, :], axis=-1)


def bend_angles(P):
    """Pseudo bond-angle at interior Cα r_i, i=1..N−2.  P (T,N,3) → (T,N−2) rad ∈(0,π)."""
    a = P[:, :-2, :] - P[:, 1:-1, :]                 # r_{i-1} − r_i
    b = P[:, 2:, :] - P[:, 1:-1, :]                  # r_{i+1} − r_i
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def pseudo_dihedrals(P):
    """Cα pseudo-dihedral φ_i of (r_{i-1},r_i,r_{i+1},r_{i+2}), i=1..N−3.  P (T,N,3)
    → (T,N−3) signed rad ∈(−π,π]. SE(3)-invariant; sign flips only under reflection."""
    b1 = P[:, 1:-2, :] - P[:, :-3, :]
    b2 = P[:, 2:-1, :] - P[:, 1:-2, :]
    b3 = P[:, 3:, :] - P[:, 2:-1, :]
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    b2n = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    m1 = np.cross(n1, b2n)
    return np.arctan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))


def circular_var(phi):
    """Variance of wrapped deviations about the circular mean (harmonic-well estimator
    for an angle). phi (T,K) rad → (K,). Reduces to ordinary variance for small spread."""
    mu = np.arctan2(np.sin(phi).mean(0), np.cos(phi).mean(0))
    d = np.arctan2(np.sin(phi - mu), np.cos(phi - mu))
    return (d ** 2).mean(0)


def local_stiffness(P, kT=KT_DEFAULT):
    """Per-position discrete-rod stiffness PROFILE from equilibrium fluctuations.
    Each profile is a 1-D array along the loop:
       stretch  k_s(i)=kT/Var(l_i)      length N−1  (virtual bonds)
       bend     k_b(i)=kT/Var(θ_i)      length N−2  (interior vertices)
       torsion  k_t(i)=kT/Var_circ(φ_i) length N−3  (4-atom windows)
    Returns the stiffness profiles and the underlying variances."""
    l = bond_lengths(P); th = bend_angles(P); ph = pseudo_dihedrals(P)
    vl, vth = l.var(0), th.var(0)
    vph = circular_var(ph) if ph.shape[1] else np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ks = kT / np.where(vl > 0, vl, np.nan)
        kb = kT / np.where(vth > 0, vth, np.nan)
        kt = kT / np.where(vph > 0, vph, np.nan)
    return dict(var_stretch=vl, var_bend=vth, var_torsion=vph,
                k_stretch=ks, k_bend=kb, k_torsion=kt)


# ============================================= harmonic-basin (caveat) checker
def basin_bimodality(q, V, k=1):
    """Sarle bimodality coefficient of q projected onto each of the k softest modes.
    BC = (skew²+1)/kurtosis (raw 4th standardized moment). Gaussian (harmonic) ⇒ 1/3;
    BC > 5/9 ≈ 0.555 suggests a bimodal / multi-basin projection, where kT/Var is only a
    LOCAL/effective stiffness. Returns (k,)."""
    out = []
    for c in range(min(k, V.shape[1])):
        a = q @ V[:, c]; a = a - a.mean()
        z = a / (a.std() + 1e-12)
        out.append(((z ** 3).mean() ** 2 + 1.0) / ((z ** 4).mean() + 1e-12))
    return np.array(out)


# ============================================================== one-loop driver
def analyse_loop(P, kT=KT_DEFAULT):
    """Full equilibrium-fluctuation elasticity fingerprint for one loop P (T,N,3).
    Returns (scalars dict, arrays dict)."""
    T, N, _ = P.shape
    ndof = max(3 * N - 6, 1)
    q, dbar, pairs = normalized_strain(P)
    Sigma = compliance(q)
    sp = spectrum(Sigma, kT=kT, ndof=ndof)
    foot = mode_residue_footprint(sp["V"][:, 0], pairs, N)          # softest-mode residue map
    bc = float(basin_bimodality(q, sp["V"], k=1)[0])
    loc = local_stiffness(P, kT=kT)
    scalars = dict(
        N=N, ndof=ndof,
        C_mean=sp["C_mean"], k_soft=sp["k_soft"], K_mean=sp["K_mean"],
        soft_frac1=sp["soft_frac1"], eff_modes=sp["eff_modes"], n_real=sp["n_real"],
        soft_mode_bc=bc, multibasin=bool(bc > 0.555),
        k_bend_min=float(np.nanmin(loc["k_bend"])) if loc["k_bend"].size else float("nan"),
        k_bend_med=float(np.nanmedian(loc["k_bend"])) if loc["k_bend"].size else float("nan"),
        k_tors_min=float(np.nanmin(loc["k_torsion"])) if loc["k_torsion"].size else float("nan"),
        k_tors_med=float(np.nanmedian(loc["k_torsion"])) if loc["k_torsion"].size else float("nan"),
        k_stretch_med=float(np.nanmedian(loc["k_stretch"])) if loc["k_stretch"].size else float("nan"),
    )
    arrays = dict(
        lam=sp["lam"][:min(len(sp["lam"]), 12)].astype(np.float32),
        k_mode=sp["k_mode"][sp["keep"]].astype(np.float32),
        soft_footprint=foot.astype(np.float32),
        k_stretch=loc["k_stretch"].astype(np.float32),
        k_bend=loc["k_bend"].astype(np.float32),
        k_torsion=loc["k_torsion"].astype(np.float32),
    )
    return scalars, arrays


# =============================================================== self-test gate
def _rot(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _selftest():
    rng = np.random.default_rng(0)
    T, N = 6000, 8
    base = np.zeros((N, 3)); base[:, 0] = np.arange(N) * 3.8      # straight rod, 3.8 Å spacing
    P = base[None] + rng.normal(scale=0.25, size=(T, N, 3))       # internal fluctuations
    q, _, _ = normalized_strain(P); S = compliance(q); sp = spectrum(S, ndof=3 * N - 6)
    loc = local_stiffness(P)
    ok = True

    # 1) invariance: random per-frame rotation+translation leaves every quantity fixed
    Pr = np.empty_like(P)
    for t in range(T):
        Pr[t] = P[t] @ _rot(t).T + rng.normal(scale=5.0, size=3)
    q2, _, _ = normalized_strain(Pr); loc2 = local_stiffness(Pr)
    e_q = np.abs(compliance(q2) - S).max()
    e_b = np.nanmax(np.abs(loc2["k_bend"] - loc["k_bend"]))
    e_t = np.nanmax(np.abs(loc2["k_torsion"] - loc["k_torsion"]))
    print(f"[1] rigid-invariance  Σ_q Δ={e_q:.2e}  k_bend Δ={e_b:.2e}  k_tors Δ={e_t:.2e}")
    ok &= e_q < 1e-9 and e_b < 1e-6 and e_t < 1e-6

    # 2) rigid-only motion of a FIXED shape ⇒ zero strain, zero compliance, infinite stiffness
    fixed = base + rng.normal(scale=0.4, size=(N, 3))
    Pf = np.stack([fixed @ _rot(1000 + t).T + rng.normal(scale=9.0, size=3) for t in range(T)])
    qf, _, _ = normalized_strain(Pf); Sf = compliance(qf)
    locf = local_stiffness(Pf)
    print(f"[2] rigid-null        max|Σ_q|={np.abs(Sf).max():.2e}  "
          f"max Var(θ)={locf['var_bend'].max():.2e}  max Var(l)={locf['var_stretch'].max():.2e}")
    ok &= np.abs(Sf).max() < 1e-16 and locf["var_bend"].max() < 1e-20

    # 3) reflection flips signed φ but preserves l, θ, |φ|
    Pref = P.copy(); Pref[..., 2] *= -1
    d_l = np.abs(bond_lengths(Pref) - bond_lengths(P)).max()
    d_th = np.abs(bend_angles(Pref) - bend_angles(P)).max()
    d_absphi = np.abs(np.abs(pseudo_dihedrals(Pref)) - np.abs(pseudo_dihedrals(P))).max()
    d_sign = np.abs(pseudo_dihedrals(Pref) + pseudo_dihedrals(P)).max()   # φ → −φ
    print(f"[3] reflection        Δl={d_l:.2e}  Δθ={d_th:.2e}  Δ|φ|={d_absphi:.2e}  "
          f"(signed φ flips, Δ(φ+φ')={d_sign:.2e})")
    ok &= d_l < 1e-10 and d_th < 1e-9 and d_absphi < 1e-9 and d_sign < 1e-9

    # 4) harmonic stretch recovery: impose a known bond-length variance ⇒ kT/Var recovers k_s
    Ph = np.repeat(base[None], T, 0).astype(float)
    sig = 0.15                                                    # Å std on one bond
    Ph[:, 3, 0] += rng.normal(scale=sig, size=T)                  # move atom 3 along x only
    ks = local_stiffness(Ph)["k_stretch"]
    exp = KT_DEFAULT / sig ** 2
    print(f"[4] stiffness recovery  k_s(bond2) est={ks[2]:.2f}  expected≈{exp:.2f}  "
          f"(bond3 est={ks[3]:.2f})")
    ok &= abs(ks[2] - exp) / exp < 0.08 and abs(ks[3] - exp) / exp < 0.08

    # 5) spectrum sanity: soft-first ordering and rank ≤ 3N−6
    print(f"[5] spectrum          λ desc={np.all(np.diff(sp['lam']) <= 1e-12)}  "
          f"n_real={sp['n_real']} (≤3N−6={3*N-6})  k_soft={sp['k_soft']:.3g} kcal/mol")
    ok &= np.all(np.diff(sp["lam"]) <= 1e-12) and sp["n_real"] <= 3 * N - 6

    print("\nRESULT:", "OK — all checks passed" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
