# PLAN — CDR loop as a discrete elastic rod (alignment-free bend / twist / stretch decomposition)

**Question.** Recast CDR-loop motion in the language of **elastic-rod theory**: treat the Cα backbone as a discrete
rod centreline and split every frame's motion into **rigid reorientation** (zero internal strain) + **bending** +
**torsional deformation** + **stretch/compression** + a coupling **residual** — all from Cα geometry, **no
superposition**. This turns the parent experiment's single black-box `D_deform` into *mechanically labelled* strain
classes, and adds an **effective-stiffness** (compliance) model that says how soft each CDR is to each mode.

$$ \boxed{\ \text{CDR motion} \;=\; \underbrace{\text{rigid reorientation}}_{\Delta d_{LL}=0}\;+\;\underbrace{\text{bending}}_{\Delta\kappa}\;+\;\underbrace{\text{torsional deformation}}_{\Delta\tau}\;+\;\underbrace{\text{stretch}}_{\Delta\ell}\;+\;\underbrace{\text{coupling residual}}_{\text{cross-terms}}\ } $$

with a **single-axis hinge** treated as a *special case* of rigid reorientation, not assumed.

## Relation to prior work (what this reuses, what it adds — no duplication)

- **Parent** [`../`](..) (done, 22 TCRs): alignment-free `D_framework` (hinge displacement) vs `D_deform` (intra-loop
  distance change); ~80 % hinge, θ≈9–12°, CDR3 softest.
- **`../graph_method/`** (core validated): `motion = hinge g(q) + deformation Bz + residual ε`, GPA/Karcher rotation,
  standalone single-axis-hinge validity, joint ALS. Established **rigid-like motion ≠ one fixed-axis hinge**.

**Division of labour.** The *rigid / hinge half* (`d_LF`, framework-relative reorientation, the hinge axis and its
single-axis validity) is **already handled** by `graph_method` — this experiment **imports it** (`rotations.py`,
`hinge_validity.py`, `graph_fit.py`) and does **not** re-derive it. What is new here is the **deformation half**:
resolving the intrinsic shape change (`d_LL`) into **bend / twist / stretch** mechanical classes, plus an
effective-stiffness energy model. Net descriptor per CDR:

$$ \boxed{\ \mathcal F_{\rm CDR}=[\,F_{\rm rigid},\ F_{\rm bend},\ F_{\rm torsion},\ F_{\rm stretch},\ F_{\rm residual}\,]\ } $$

where `F_rigid` comes from `graph_method` (`d_LF` rigid part) and the last four are new (`d_LL` / internal coords).

## Scientific invariants (do not violate)
- **Alignment-free = no superposition, not "no coordinates."** Internal coordinates (virtual-bond lengths, pseudo
  bond-angles, pseudo-dihedrals) are **SE(3)-invariant functions of the raw Cα coordinates** — invariant to any
  rotation/translation of any frame — so computing them from `tv.mdtraj.xyz` is fully alignment-free. This matches
  the parent invariant ("no coordinate superposition in the core metrics") and how `graph_method` already uses GPA
  only for the *rigid* part. Only the effective-stiffness eigenmodes touch a covariance, never a superposition.
- **Transferable method** — no per-loop/per-system tuning; identical formulas and any thresholds data-driven
  (quantile/gap based), all six CDRs, all systems. See [[prefer-transferable-methods]].
- **α to α framework, β to β framework** (per-chain), inheriting the parent/graph_method anchoring; the rigid part
  reuses their rigid-core detection. The elastic-rod strains are intrinsic to the loop and need **no anchor at all**.
- Report **relative** patterns; magnitudes are sampling-dependent (1KGC is the ~10× breadth outlier — keep, flag).

---

## The strain variables — all from Cα, all alignment-free

Loop Cα `x_1..x_N` (a CDR, IMGT-ordered). Virtual-bond vectors `e_i = x_{i+1}-x_i`, lengths `ℓ_i=\|e_i\|=d_{i,i+1}`.

### Completeness (why there is no *shape* residual, only a coupling residual)
An N-point chain has **3N−6** internal DOF. The internal set
$$ \{\ell_i\}_{i=1}^{N-1}\ (\text{stretch}),\quad \{\phi_i\}_{i=2}^{N-1}\ (\text{bend}),\quad \{\tau_i\}_{i=2}^{N-2}\ (\text{torsion}) $$
has exactly `(N-1)+(N-2)+(N-3)=3N-6` members — a **complete, non-redundant** basis for the loop shape (the classic
Z-matrix / bond–angle–torsion count). So **bend+twist+stretch span *all* intrinsic deformation**; the parent's full
`d_LL` matrix (`N(N-1)/2` pairs) is *over-complete* and we replace it with this minimal mechanically-labelled basis.
Consequence: any "residual" in the *variance budget* is a **cross-class coupling / nonlinearity** term (the analogue
of `graph_method`'s `f_coupling`), **not** a missing degree of freedom.

### Bending — change in discrete curvature (3 points)
Turning angle from the three pairwise distances (reflection-safe, alignment-free):
$$ \cos\phi_i=\frac{d_{i-1,i}^2+d_{i,i+1}^2-d_{i-1,i+1}^2}{2\,d_{i-1,i}\,d_{i,i+1}},\qquad
   \kappa_i = \frac{2\tan(\phi_i/2)}{\tfrac12(\ell_{i-1}+\ell_i)}\ (\text{inv. length}). $$
Per-residue bending signal `Δκ_i(t)=κ_i(t)-\bar\kappa_i` (mean over frames), amplitude
`F_bend=\sqrt{\tfrac1{N_\kappa}\sum_i \mathrm{Var}_t[\kappa_i]}`. Keep both the dimensionless `φ_i` and the
length-normalised `κ_i`; `φ_i` is the most robust and the natural bending coordinate.

### Torsional deformation — change in Cα pseudo-dihedral (4 points)
Signed pseudo-dihedral `τ_i` of `x_{i-1},x_i,x_{i+1},x_{i+2}` (standard atan2 form from the three vectors).
**`τ_i` is SE(3)-invariant → already alignment-free, and keeps its sign** (it flips only under *reflection*, which
never occurs in an MD frame). We therefore use **signed `τ_i` from coordinates** as primary. The strict
distance-only route recovers `\cos\tau_i` (hence `|\tau_i|`) from the 4-point distance sub-matrix but **loses the
sign** — kept only as an invariance cross-check. `Δτ_i(t)=\mathrm{wrap}(\tau_i(t)-\bar\tau_i)`,
`F_torsion=\sqrt{\tfrac1{N_\tau}\sum_i \mathrm{Var}_t[\tau_i]}` (circular variance).
*Caveat (name it honestly):* this is **backbone torsional deformation** of the centreline, **not** the Cosservat
**material twist** — true material twist needs an oriented cross-section frame (unavailable from Cα alone). Optional
extension: add Cβ (or the peptide-plane O) to define a material frame and recover real twist; out of scope for the
pure-Cα default.

### Stretch/compression — axial strain (consecutive)
$$ \epsilon_i(t)=\frac{d_{i,i+1}(t)-\bar d_{i,i+1}}{\bar d_{i,i+1}},\qquad
   F_{\rm stretch}=\sqrt{\tfrac1{N_\ell}\sum_i\mathrm{Var}_t[\epsilon_i]}. $$
Expected **tiny** (`F_stretch ≪ F_bend,F_torsion`) — Cα–Cα spacing is covalently constrained; apparent "extension"
of a CDR is almost always bending/straightening, not backbone stretching. Reporting it *proves* that rather than
assuming it.

### Rigid reorientation — the zero-strain part (imported)
`Δd_LL≈0` **certifies** a motion is rigid (zero elastic strain energy) even while `d_LF` changes. The *magnitude* of
the rigid reorientation and its axis come from `graph_method` (`d_LF` rigid fit → `F_rigid`, hinge axis `L*`,
single-axis validity). We do **not** recompute it; we consume it and test consistency (`Δd_LL≈0` on the frames
`graph_method` calls rigid).

---

## The deformation variance budget (common Å currency, reconnects to the parent)

`F_bend`, `F_torsion`, `F_stretch` live in different units (rad², rad², dimensionless). To make them **comparable and
additive**, and to tie back to the parent's `D_deform` (Å), attribute the *observed intrinsic deformation variance*
— `Var_t[d_LL]`, the parent currency — to each mechanical class by **linear (small-fluctuation) regression** of the
mean-centred distance fluctuations `δd^{LL}(t)` onto the strain fluctuations `[δℓ, δφ, δτ](t)` (the Jacobian of the
minimal basis). Report variance fractions
$$ f_{\rm bend}+f_{\rm torsion}+f_{\rm stretch}+f_{\rm coupling}=1 $$
(of `d_LL` variance — **not** % of RMSD/Cartesian/entropy), exactly parallel to `graph_method`'s `f_*`. `f_coupling`
= the cross-class covariance that closes the budget. **Cross-check (a real cross-validation):**
`F_bend ⊕ F_torsion ⊕ F_stretch` should reconstruct the parent's `D_deform` for the same CDR — same deformation,
now mechanistically resolved. Example target readout (illustrative):

| class | typical fraction |
|---|---|
| rigid reorientation (`f_rigid`, from `d_LF`) | ~55 % |
| bending | ~32 % |
| torsional deformation | ~10 % |
| stretch | ~1 % |
| coupling residual | ~2 % |

---

## Effective-stiffness / elastic-energy model (the new mechanics, → entropy hook)

Discrete-rod energy in the minimal strains, harmonic (Gaussian-fluctuation) approximation:
$$ E_{\rm elastic}=\tfrac12\sum_i\Big[B_i(\Delta\kappa_i)^2+C_i(\Delta\tau_i)^2+K_i\,\epsilon_i^2\Big]
   \ \Longleftrightarrow\ E=\tfrac12\,\delta q^\top \mathbf K\,\delta q,\quad q=[\ell,\phi,\tau]. $$
Do **not** assume literal protein stiffness constants — **infer effective ones from the MD covariance**:
$$ \boxed{\ \mathbf K_{\rm eff}=k_BT\,\Sigma^{-1},\qquad \Sigma=\mathrm{Cov}_t[\delta q]\ } $$
- **Compliance-first (primary, robust):** report `Σ` directly — its diagonal blocks are the per-residue bending /
  torsional / stretch **softness** (large variance = soft). No inversion, no ill-conditioning. This alone answers
  "how mechanically soft is this CDR to bending vs twisting, and *where*?" — a per-residue **stiffness profile**
  whose minimum should coincide with the `graph_method` hinge location (the headline cross-method figure).
- **Stiffness `K_eff` (secondary, harmonic model):** needs `Σ^{-1}` → **regularise** (shrinkage / ridge, or restrict
  to the top modes; N is small so `Σ` is well-sampled but keep it honest). Per-residue effective moduli `B_i,C_i,K_i`
  read from the diagonal blocks (couplings kept).
- **Persistence length** from the bending block (worm-like-chain): `L_p = \bar B\,\langle\ell\rangle/k_BT`, and/or from
  the decay of `\langle \hat e_i\cdot \hat e_j\rangle` along the contour — a single interpretable stiffness number
  per CDR.
- **Elastic soft modes:** eigen-decompose `Σ` (or `K_eff`) → the softest internal mode's shape shows whether the
  compliance is **localised at one joint** (a bend-hinge) or **distributed** — the elastic-theory read of "hinge vs
  deformation," now *inside* the deformation.
- **Entropy hook (why this matters downstream):** the harmonic model gives the loop's **quasi-harmonic
  configurational entropy** `S ≈ \tfrac12 k_B\ln\det(2\pi e\,\Sigma)` in the *internal* (alignment-free) coordinates
  — a superposition-free flexibility/entropy estimator that connects directly to the MM/GBSA entropy-gap work
  ([[lc13-interface-cycle-entropy]], [[lc13-cross-system-generalisability]]). Bound-vs-unbound `ΔΣ` on the Knapp
  ensembles = does binding **stiffen** the CDR (lose configurational entropy)? — a clean, testable contribution.

---

## Validation gate (must pass before trusting numbers — `tests.py`, matching the project's test-gate culture)
1. **Invariance:** random rotation+translation of a frame → `κ_i, |\tau_i|, ℓ_i, F_*` unchanged to ~1e-12; a
   *reflection* leaves `κ,ℓ,|\tau|` unchanged but **flips signed `τ`** (documented behaviour, not a bug).
2. **Completeness / round-trip:** rebuild loop coordinates from `{ℓ,φ,τ}` → matches the original shape to ~1e-6
   (proves the basis is complete; the variance "residual" is coupling, not lost DOF).
3. **Synthetic pure modes:** impose (a) a pure rigid rotation → `Δκ=Δτ=Δε≈0`, `f_rigid≈1`, `graph_method` recovers the
   angle; (b) a pure single-joint bend → localised `Δκ` spike, `τ,ε≈0`, `f_bend≈1`; (c) a pure twist → `Δτ` spike,
   `κ,ε≈0`, `f_torsion≈1`; (d) a pure stretch → `Δε`, others ≈0.
4. **Budget closes:** `f_bend+f_torsion+f_stretch+f_coupling=1.000`.
5. **Stiffness recovery:** synthetic harmonic rod with known `B,C,K` and Boltzmann-sampled frames → `K_eff=k_BT\Sigma^{-1}`
   recovers `B,C,K` within sampling error; `Σ` diagonal ∝ `1/stiffness`.
6. **Cross-method consistency:** on real MD, `F_bend⊕F_torsion⊕F_stretch` reconstructs the parent `D_deform` (same
   CDR) within tolerance; `Δd_LL≈0` on the frames `graph_method` labels rigid.

---

## Data (unchanged from the parent — drop-in)
- **Primary:** `/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD/<ID>/<ID>.{pdb,xtc}` — 22 unbound TCRs
  (`1KGC 2BNU 2CDF 2CDG 3DX9 3QEU 3QH3 3SKN 3VXQ 3VXT 4DZB 4JFH 4UDT 4ZDH 6FRA 6OVN 6XQQ 7EA6 7N1D 7R7Z 7S8I 8YJ3`),
  stride **25** (~2.3k frames). Same set as parent/graph_method → direct comparison.
- **Bound-vs-unbound stiffness Δ:** Knapp replica ensembles
  `/mnt/larry/lilian/DATA/Unbound_Bound/MD_runs_unbound_bound/<SYS>/` (`SYS∈{LC13,A6,1G4,JM22}`),
  `*_TCR_run*/*.firstFrame.pdb` + `*.final.md.xtc` (unbound) and `*_TCRpMHC_run*` (bound); pool replicas before `Σ`.
  Reuse `../../LC13_calculations/config*.yaml` as the ready-made manifest.

## Implementation — reuse the kinapse API and the sibling primitives (confirmed against source)
```python
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":               # env's user-site h5py is ABI-broken
    os.environ["PYTHONNOUSERSITE"] = "1"; os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np, mdtraj as md
from kinapse.structures import load_tcr

md.load(xtc, top=pdb, stride=STRIDE).save_xtc(tmp)          # stride first (memory)
tv  = load_tcr(pdb, traj=tmp).pairs[0].traj                 # TrajectoryView
idx, names = tv.domain_idx(["A_CDR3"], atom_names={"CA"}, pass_names=True)   # loop Cα (+IMGT names)
X = tv.mdtraj.xyz[:, np.asarray(idx)] * 10.0                # (n_frames, N, 3) Å  — the rod centreline
# ℓ,φ (κ),τ,ε are pure numpy on X (SE(3)-invariant); Σ = np.cov of the stacked strain fluctuations.
```
- Reuse `../graph_method/graph_fit.py` (`dmat`, `cross`, `distance_rigid_fit`), `rotations.py`
  (`to_framework_frame`, `gpa`, `karcher_mean`, `relative_rotvecs`, `consensus_axis`), `hinge_validity.py`
  (single-axis test) for the **rigid/hinge half** — import, don't reimplement. IMGT `A_CDR3=105–117` etc. from
  `kinapse.regions`; stems for the axis check via the `names` list.
- Region prefixes `A_`/`B_` drive chain selection (γδ-safe). Vectorise strains across frames.
- Free-energy surfaces `F(φ_apex)`, `F(θ, κ_apex)`: reuse `kinapse.dynamics_analysis.compute_pmf_kde`.
- **Robustness (inherited — do not rediscover):** `8YJ3` needs `load_tcr(..., manual_chain_types={'A':'B','B':'A'})`;
  no periodic box → gate split-domain frames via the Vα–Vβ interface distance (matters only for the imported `d_LF`
  rigid part; the intrinsic strains are immune); nm→Å ×10; try/except per system, resumable.

## Deliverables / execution
- `config.py` — extends the parent's (same `DATA`, `SYSTEMS`, `STRIDE`, `CDRS`, `CHAIN_OVERRIDES`); adds
  covariance-regularisation choice and the circular-variance handling for `τ`.
- `elastic_rod.py` — strain primitives (`ell`, `phi`/`kappa`, `tau_signed`, `eps`, completeness round-trip),
  variance-budget regression, `Sigma`/`K_eff`/persistence-length/soft-modes. Pure numpy, importable, unit-tested.
- `tests.py` — the validation gate above (run first, must pass).
- `compute_elastic_rod.py` — per system: stride → load → per-CDR strains → `F_*`, `f_*` budget, `Σ`, `K_eff`, soft
  modes, persistence length; import `graph_method` for `F_rigid`/axis; save per-system `npz`+`csv`, resumable, bg.
- `plot_elastic_rod.py` — aggregate + figures + `RESULTS.md`. Figures: (f1) per-residue `Δκ`/`Δτ`/`ε` profiles along
  each CDR (localise the soft spot); (f2) `\mathcal F_{\rm CDR}` stacked variance budget per CDR
  (rigid/bend/torsion/stretch/coupling); (f3) **bending-stiffness profile `B_i(s)` with the `graph_method` hinge
  location overlaid** — the payoff: *does the rod bend where it hinges?*; (f4) persistence length per CDR and vs loop
  length; (f5) `F(θ, κ_apex)` PMF; (f6) bound−unbound `ΔΣ`/`Δentropy` (Knapp) — does binding stiffen the loop?
- Run: `mamba activate kinapse; python tests.py; python compute_elastic_rod.py [--systems 3QH3 8YJ3] [--force]; python plot_elastic_rod.py`.
  Outputs to git-ignored `results/`; PNGs to `figures/`.

## Open decisions (resolve at first run, keep transferable)
- Torsion: signed-from-coordinates (primary) — confirm the reflection cross-check documents the sign flip cleanly.
- `Σ` regularisation for `K_eff`: shrinkage vs top-mode truncation — pick one, apply to all systems/CDRs.
- Reference mean for strains: ensemble mean (primary) vs `graph_method`'s clean/low-deformation frames (secondary).
- Whether to add Cβ for true material twist — default **no** (keep pure-Cα, alignment-free); revisit only if
  backbone-torsion turns out to carry a large, mechanistically interesting fraction.
