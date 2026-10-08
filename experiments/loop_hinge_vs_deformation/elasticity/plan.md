# PLAN — elasticity theory of CDR loops (three formalisms, one principle)

**Fresh, separate approach** (does not build on `../` hinge/deform, `../graph_method/`, or `../elastic_rod/`).
Describe CDR-loop motion in MD with **classical elasticity theory**, three complementary formalisms, all
**alignment-free from CA–CA distances**:

1. **Continuum strain field** — per-residue finite (Green–Lagrange) strain tensor from local CA–CA distance changes;
   volumetric + shear + **non-affine** decomposition. *Where and how does the loop deform.*
2. **Elastic network model (GNM/ANM)** — spring network from the CA contact map; elastic normal modes; soft
   collective motions vs the observed MD fluctuations. *What modes the loop's shape affords.*
3. **Data-driven stiffness / linear response** — invert the CA–CA distance-fluctuation covariance to an effective
   stiffness network; compliance, soft modes, probe-force response, quasi-harmonic entropy. *How soft the loop is.*

## The one principle that unifies them: rigid = the zero-strain null space

Do **not** define "rigid" and subtract it. For a finite motion of a deformable body, the rotation attributed to a
shape change is gauge-dependent (falling-cat coupling), so fitting a rotation first lets a rotation-mimicking
deformation leak into the "rigid" bucket. Instead **define deformation** — it is convention-free — and let rigid be
its null space. Every formalism here is built from **SE(3)-invariants**, so a rigid motion contributes *exactly zero*:

| formalism | deformation observable | rigid motion is |
|---|---|---|
| strain field | `E_i = ½(FᵢᵀFᵢ − I)` (finite strain) | `E_i = 0` ⟺ `FᵢᵀFᵢ = I` — no frame choice |
| elastic network | non-zero-eigenvalue modes of Kirchhoff Γ / Hessian H | the **zero modes** (1 for GNM, 6 for ANM) |
| stiffness | covariance `Σ` of CA–CA distances | **zero variance** (distances are invariant) |

A pure hinge changes loop→framework distances but leaves all three deformation observables at zero; **simultaneous**
deformation is the only nonzero part → hinge and deformation are separated with no superposition and no gauge.

**Quantifying the rigid part as an angle (optional).** Only if a hinge *angle* is wanted: fit the **fixed reference
shape** to each frame in the **Eckart / best-fit frame** (alignment-free `distance_rigid_fit` from
`../graph_method/graph_fit.py`), fit it **jointly with the invariant deformation coordinate `d_LL`** (a rigid motion
cannot change `d_LL`, so a rotation-mimicking deformation is forced out of the angle), and report the honest error
bar `δθ ≈ E_nonrigid/ℓ` (→0 rigid, grows with deformation). The three formalisms **describe deformation without ever
needing this**; it is imported only for the headline "the loop also swings θ°" number.

## Scientific invariants (do not violate)
- **Alignment-free = no superposition, not "no coordinates."** All observables (strain invariants `tr E`, `‖dev E‖`,
  principal strains, `D²min`; Kirchhoff Γ; distance covariance `Σ`) are invariant to any rotation/translation (and,
  for the strain/network scalars, reflection) of any frame — computed from `tv.mdtraj.xyz` with **no Kabsch** in any
  core metric. Reflection cannot be resolved by these symmetric-tensor / distance observables (documented, not a bug).
- **Reference config = ensemble medoid frame** (the frame minimizing `Σ‖d_LL − ⟨d_LL⟩‖`) — an **invariant** choice,
  not "frame 0"; consistent with the parent's medoid. Also report vs crystal (frame 0) as secondary.
- **Transferable** — identical formulas / data-driven thresholds (cutoff `Rc`, neighborhood, ridge) for all systems &
  all six CDRs; no per-loop tuning ([[prefer-transferable-methods]]).
- **Per chain** (α-loops in the α frame, β in β), matching the dataset conventions; the strain field & `Σ` are
  intrinsic to the loop and need no anchor.
- Report **relative** patterns; magnitudes sampling-dependent (1KGC = ~10× breadth outlier — keep, flag).

---

## Method 1 — continuum finite-strain field
Reference loop `x⁰_i` (medoid). Neighborhood `𝒩(i)` = CA within `Rc` on the reference (≥ `kmin`, expand `Rc` if
short). Per node & frame, least-squares **deformation gradient** over reference→current bond vectors
`D_ij=x⁰_j−x⁰_i`, `d_ij=x_j−x_i`:
$$ F_i=\Big(\sum_{j\in𝒩}d_{ij}D_{ij}^\top\Big)\Big(\sum_{j\in𝒩}D_{ij}D_{ij}^\top\Big)^{-1},\qquad
   E_i=\tfrac12(F_i^\top F_i-I). $$
Frame-independent readouts: **volumetric** `θ_i=\mathrm{tr}E_i`, **shear** `γ_i=\|\mathrm{dev}\,E_i\|_F`, von-Mises
`ε^{vM}_i`, principal strains/directions `eig(E_i)`, and the **non-affine residual** (Falk–Langer)
$$ D^2_{\min,i}=\big\langle\,\|d_{ij}-F_iD_{ij}\|^2\,\big\rangle_{j\in𝒩}. $$
`E_i` invariants are functions of CA–CA distances only (via `C=F^\top F` and the dot-products the distances fix);
`F` is formed from coordinates purely for the fit — **no superposition**. `D²min` and the invariants are rigid-
(and reflection-) invariant. **Rigid ⇒ E_i=0, D²min=0** at every node (the unifying principle, method-1 form).
**Outputs:** per-residue strain-field profile (`γ_i`, `θ_i`) along the loop, shear-vs-volumetric character, and the
**non-affine hotspot** (max `D²min`) = the localized rearrangement / hinge point; PMF `F(γ_apex)`.

## Method 2 — elastic network model (GNM primary, ANM secondary)
From the CA contact map (distances < `Rc`): Kirchhoff `Γ` (GNM) / Hessian `H` (ANM).
$$ \Gamma_{ij}=-\mathbb 1[d_{ij}<R_c]\ (i\neq j),\quad \Gamma_{ii}=-\!\sum_{j\neq i}\Gamma_{ij};\qquad
   \langle\Delta R_i\!\cdot\!\Delta R_j\rangle = k_BT\,(\Gamma^{-1})_{ij}. $$
GNM predicts **per-residue mobility** `(Γ⁻¹)_ii` and cross-correlations from the shape alone (alignment-free — Γ is a
function of distances). Low-frequency modes = soft collective motions; **1 zero mode = rigid** (GNM), 6 for ANM. Build
Γ on the medoid **and** per frame (ensemble-averaged network). **Validate against the MD:** overlap of GNM/ANM soft
modes with the actual MD fluctuation modes (mode overlap / cumulative overlap); does textbook uniform-spring
elasticity already explain the observed loop motion, or is the loop anisotropically soft? ANM mode *directions* are
Cartesian (compared only in the framework frame); mobilities/eigenvalues are invariant.

## Method 3 — data-driven stiffness / linear response
Strain vector `s(t)` = intra-loop CA–CA distance fluctuations (`d_LL`, the complete invariant deformation set).
$$ \Sigma=\mathrm{Cov}_t[s],\qquad \mathbf K_{\rm eff}=k_BT\,\Sigma^{-1}\ (\text{regularized}). $$
- **Compliance-first (robust):** `Σ` directly — its spectrum gives the **soft elastic modes** (large eigenvalue =
  soft); the leading eigenvector's residue footprint says whether softness is **localized** (a bend-hinge) or
  **distributed** (the elasticity read of hinge-vs-deform). Rigid motion adds zero variance ⇒ no contamination.
- **Stiffness (harmonic model):** `K_eff` with ridge/shrinkage (report the regularizer); an MD-*derived* elastic
  network (spring constants inferred, not assumed — the data-driven counterpart of Method 2).
- **Linear response:** `Δs = K_eff⁻¹ f` — the loop's deformation under a unit probe force at the apex (Green's fn).
- **Quasi-harmonic entropy:** `S ≈ ½k_B ln det(2πe Σ)` on the **minimal 3N−6 internal coords** (not the redundant
  full distance set — note the redundancy/units caveat) → a superposition-free flexibility/entropy number; bound−unbound
  `ΔΣ`/`ΔS` on the Knapp sets tests whether binding **stiffens** the loop ([[lc13-interface-cycle-entropy]]).

---

## Validation gate (`tests.py` — run first, must pass; matches the project's test-gate culture)
1. **Invariance:** random rotation+translation (+ reflection) of a frame → strain invariants (`tr E`, `‖dev E‖`,
   `D²min`), Γ / mobilities, and `Σ` spectrum unchanged to ~1e-10.
2. **Rigid ⇒ zero strain (the principle):** apply a rigid motion between reference and frame → `E_i=0`, `D²min=0` ∀i;
   `d_LL` unchanged ⇒ `Σ_LL=0`; GNM/ANM zero modes carry it.
3. **Affine recovery:** impose a known homogeneous `F₀` (e.g. 10 % stretch + shear) → recover
   `E₀=½(F₀ᵀF₀−I)` exactly at every node, `D²min≈0` (affine ⇒ no non-affine part).
4. **Non-affine detection:** add a per-atom random displacement → `D²min>0`, localized.
5. **ENM sanity:** GNM `Γ` singular with exactly one zero eigenvalue; `(Γ⁻¹)_ii` matches an equipartition reference on
   a synthetic network.
6. **Stiffness recovery:** synthetic harmonic network with known `K`, Boltzmann-sampled → `k_BT\,Σ⁻¹` recovers `K`
   within sampling error; `Σ` eigen-softness ∝ `1/eig(K)`.

## Data (drop-in, same as the parent)
- **Primary:** `/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD/<ID>/<ID>.{pdb,xtc}` — 22 unbound TCRs, stride **25**.
- **Bound-vs-unbound:** Knapp `/mnt/larry/lilian/DATA/Unbound_Bound/MD_runs_unbound_bound/<SYS>/`
  (`SYS∈{LC13,A6,1G4,JM22}`), `*_TCR_run*` (unbound) / `*_TCRpMHC_run*` (bound), pool replicas.

## API (confirmed against source)
```python
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"; os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np, mdtraj as md
from kinapse.structures import load_tcr
md.load(xtc, top=pdb, stride=25).save_xtc(tmp)
tv = load_tcr(pdb, traj=tmp).pairs[0].traj
idx = np.asarray(tv.domain_idx(["A_CDR3"], atom_names={"CA"}))   # loop Cα (IMGT 105–117)
X = tv.mdtraj.xyz[:, idx] * 10.0                                  # (T,N,3) Å — the loop
```
`8YJ3` → `manual_chain_types={'A':'B','B':'A'}`; regions from `kinapse.regions`; nm→Å ×10; PMF via
`kinapse.dynamics_analysis.compute_pmf_kde`. Optional hinge angle imports `../graph_method/graph_fit.distance_rigid_fit`.

## Deliverables
- `config.py` (paths, systems, `STRIDE`, `CDRS`, `Rc`, `kmin`, ridge), `elast.py` (all three formalisms' primitives),
  `tests.py` (the gate). **This turn.**
- `compute_elasticity.py` (per system, per CDR: strain field + ENM + stiffness → per-frame/per-residue arrays,
  resumable, bg), `plot_elasticity.py` (aggregate + figures + `RESULTS.md`). **Next.**
- Figures: (f1) per-residue `γ_i`/`D²min` strain-field profile per CDR; (f2) shear-vs-volumetric map; (f3) non-affine
  hotspot vs the `graph_method` hinge location (payoff); (f4) GNM soft-mode shapes + mode-overlap with MD; (f5) `Σ`
  soft-mode localization (localized vs distributed); (f6) bound−unbound stiffness/entropy Δ (Knapp).
- Run: `mamba activate kinapse; python tests.py; python compute_elasticity.py [--systems 3QH3]; python plot_elasticity.py`.

## Open decisions (resolve at first run, keep transferable)
- Strain neighborhood: spatial `Rc` vs sequence window ±w — pick one, apply to all (default spatial `Rc≈8–10 Å`, `kmin≥4`).
- ENM cutoff `Rc` (GNM ~7–8 Å, ANM ~12–15 Å typical) — data-driven, fixed across systems.
- `Σ` currency for entropy: minimal 3N−6 internal coords (correct) vs full `d_LL` (redundant) — use the minimal set.
