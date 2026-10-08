"""Validation gate for the elasticity primitives (all three formalisms).

Run first; every check must pass before trusting any MD numbers.
  mamba activate kinapse && python tests.py
"""
from __future__ import annotations
import numpy as np
import elast as E

rng = np.random.default_rng(0)
PASS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    PASS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def rand_rotation(g):
    A = g.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# a non-degenerate 3-D reference loop (Angstrom-ish) + a generic deformation gradient
N = 12
Xref = rng.normal(size=(N, 3)) * 3.0
ALL = [np.array([j for j in range(N) if j != i]) for i in range(N)]     # full neighborhood
F0 = np.array([[1.10, 0.05, 0.00],
               [0.00, 0.95, 0.03],
               [0.02, 0.00, 1.00]])
E0 = 0.5 * (F0.T @ F0 - np.eye(3))                                       # target strain

# ---- 1. invariance of strain invariants under SE(3) + reflection (current frame)
Xdef = Xref @ F0.T + rng.normal(size=(N, 3)) * 0.2                       # deformed + noise
neigh = E.neighborhoods(Xref, rc=9.0, kmin=4)
s0 = E.strain_field(Xref, Xdef[None], neigh)
R = rand_rotation(rng); t = rng.normal(size=3) * 5.0
Xrot = Xdef @ R.T + t
Xref_reflect = Xdef @ np.diag([1.0, 1.0, -1.0])                          # reflection
s1 = E.strain_field(Xref, Xrot[None], neigh)
s2 = E.strain_field(Xref, Xref_reflect[None], neigh)
inv_ok = all(
    np.allclose(s0[k], s1[k], atol=1e-8) and np.allclose(s0[k], s2[k], atol=1e-8)
    for k in ("vol", "shear", "vm", "pmax", "pmin", "d2min")
)
check("1 strain invariants: SE(3)+reflection invariant", inv_ok,
      f"max dvol={np.abs(s0['vol']-s1['vol']).max():.1e}")

# ---- 2. rigid motion => zero strain, zero d_LL variance, one GNM zero mode
Rr = rand_rotation(rng)
Xrigid = Xref @ Rr.T + rng.normal(size=3) * 4.0
sr = E.strain_field(Xref, Xrigid[None], ALL)
zero_strain = max(np.abs(sr[k]).max() for k in ("vol", "shear", "vm", "d2min"))
check("2a rigid => E_i=0, D2min=0", zero_strain < 1e-9, f"max={zero_strain:.1e}")

traj_rigid = np.stack([Xref @ rand_rotation(rng).T + rng.normal(size=3) * 3.0
                       for _ in range(50)])
d_rig, _ = E.dll_series(traj_rigid)
Sig_rig, _ = E.compliance(d_rig)
check("2b rigid => Sigma(d_LL)=0", np.abs(Sig_rig).max() < 1e-9,
      f"max|Sigma|={np.abs(Sig_rig).max():.1e}")

G = E.gnm_kirchhoff(Xref, rc=8.0)
w, _ = E.gnm_modes(G)
check("2c GNM has exactly one zero mode", w[0] < 1e-9 and w[1] > 1e-6,
      f"w0={w[0]:.1e}, w1={w[1]:.2e}")

# ---- 3. affine deformation recovered exactly, no non-affine part
Xaff = Xref @ F0.T + rng.normal(size=3) * 2.0
sa = E.strain_field(Xref, Xaff[None], ALL)
vol_ok = np.allclose(sa["vol"][0], np.trace(E0), atol=1e-8)
shear_ok = np.allclose(sa["shear"][0], np.linalg.norm(E0 - np.trace(E0) / 3 * np.eye(3)),
                       atol=1e-8)
check("3 affine F0 recovered (vol & shear) exactly", vol_ok and shear_ok,
      f"vol {sa['vol'][0].mean():.5f} vs {np.trace(E0):.5f}")
check("3 affine => D2min ~ 0", sa["d2min"].max() < 1e-9, f"max={sa['d2min'].max():.1e}")

# ---- 4. non-affine displacement detected
Xna = Xref @ F0.T + rng.normal(size=(N, 3)) * 0.4
sna = E.strain_field(Xref, Xna[None], ALL)
check("4 non-affine displacement => D2min>0", sna["d2min"].max() > 1e-3,
      f"max D2min={sna['d2min'].max():.3f}")

# ---- 5. GNM connected network: positive spectrum above the zero mode; MSF finite
msf, Ginv = E.gnm_msf(G, kT=1.0)
check("5 GNM MSF finite & positive", np.all(np.isfinite(msf)) and np.all(msf > 0),
      f"MSF range [{msf.min():.3f},{msf.max():.3f}]")

# ---- 6. stiffness recovery: kT * Sigma^{-1} recovers a known K; soft = low stiffness
P = 6
B = rng.normal(size=(P, P)); Ktrue = B @ B.T + P * np.eye(P)             # SPD stiffness
kT = 0.6
cov = kT * np.linalg.inv(Ktrue)                                          # sampling covariance
L = np.linalg.cholesky(cov)
S = (L @ rng.normal(size=(P, 200000))).T                                 # (T,P) samples
Sig = np.cov(S.T)
Krec = E.stiffness(Sig, kT=kT, ridge=0.0)
rel = np.linalg.norm(Krec - Ktrue) / np.linalg.norm(Ktrue)
check("6a K_eff = kT Sigma^-1 recovers known K", rel < 0.05, f"rel err={rel:.3f}")
wS, _ = E.soft_modes(Sig)                          # compliance eigenvalues, desc
wK = np.sort(np.linalg.eigvalsh(Ktrue))            # stiffness eigenvalues, asc
# softest compliance mode <-> smallest stiffness eigenvalue
corr = np.corrcoef(wS, 1.0 / wK)[0, 1]
check("6b soft compliance mode <-> low stiffness", corr > 0.99, f"corr={corr:.4f}")

# ---- 7. quasi-harmonic entropy scaling: S(cSigma) = S(Sigma) + P/2 ln c
Sig_int = Ktrue                                    # any SPD stands in for an internal-coord cov
s_a = E.quasiharmonic_entropy(Sig_int, kT=1.0)
s_b = E.quasiharmonic_entropy(4.0 * Sig_int, kT=1.0)
check("7 quasiharmonic entropy scales as P/2 ln c",
      np.isclose(s_b - s_a, P / 2 * np.log(4.0), atol=1e-6),
      f"d={s_b - s_a:.4f} vs {P/2*np.log(4.0):.4f}")

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
raise SystemExit(0 if all(PASS) else 1)
