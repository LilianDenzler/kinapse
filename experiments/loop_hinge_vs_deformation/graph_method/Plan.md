# PLAN — CDR-loop flexibility: hinge + deformation + residual (GPA rotation → distance-space decomposition)

Updated method (supersedes the earlier reference-based / pairwise drafts). Decompose each CDR loop's
framework-relative motion as

$$ \text{total framework-relative loop motion} = \text{hinge} + \text{internal deformation} + \text{residual}. $$

Built on the parent experiment [`../`](..) (validated the coordinate solver and the ~80 % hinge headline).

## Where alignment lives (read first)
- **Rotation extraction (Steps 1–2) uses coordinate superposition** — but referenced to the **ensemble mean**
  (generalized-Procrustes template + Karcher mean rotation), *never* an arbitrary frame 0. This is the physical
  hinge axis/angle. It also yields the **Cartesian deformation residual** for free.
- **Flexibility decomposition (Steps 3–8) is alignment-free** — pure loop→loop and loop→framework Cα distances.
- IMGT numbering gives node correspondence across TCRs; no TCR-to-TCR superposition anywhere.
- (If a *pure-distance* rotation is wanted instead of GPA, `graph_fit.distance_rigid_fit` is a validated drop-in.)

## Step 1 — framework coordinate system + rigidity
Same conserved IMGT framework Cα in every TCR, **per chain** (α-loops → α framework, β-loops → β framework);
rigid, distributed around the loop base, ≥4 non-coplanar, a rigid anchor on **both flanks** of the loop (the 3SKN
lesson from the parent). Validate rigidity from distances: `framework_rigidity` = max std of any framework–framework
Cα distance (small ⇒ rigid). Per-frame Kabsch onto this framework removes global tumbling → the framework frame.

## Step 2 — rotation extraction (ensemble-referenced), the CDR hinge axis
In the framework frame, run **generalized Procrustes (GPA)** on the loop → ensemble-mean **template** (central rigid
shape), per-frame loop rotation `R_t`, and per-frame **Cartesian rigid-fit residual** (deformation the rigid model
can't explain). Take the **Karcher/Fréchet mean** rotation `R̄`; define the excursion
$$ \omega_t=\log(\bar R^\top R_t)=\theta_t\,\hat u_t, \qquad \theta_t=\|\omega_t\|. $$
`θ_t` = rigid rotational excursion from the mean pose; `û_t` = instantaneous axis. Consensus hinge axis `û*` = top PC
of `{ω_t}`; **signed** angle `θ_t^s = ω_t·û*` (needed for the linear regression below). Report axis directional +
positional spread; if narrow, fit one consensus screw axis `L*=(p*,û*)`. **Cross-TCR:** compare `L*` in the shared
IMGT framework frame to test for a conserved hinge mechanism (later phase).

## Step 2b — single-hinge validity (STANDALONE, run before the joint model)
*"Can this CDR's rigid-like motion be represented by rotation about ONE fixed axis?"* — answered independently,
so the joint model is not used to validate its own hinge assumption. On **low-deformation frames only**
(so deformation can't contaminate the rigid cloud), from the Karcher-relative rotation vectors `ω_t`
([`hinge_validity.py`](hinge_validity.py)):
- **`P_1D = λ₁/Σλ`** of `⟨ω_tω_tᵀ⟩` (→1 ⇒ one axis); **`r_perp ≈ 1−P_1D`** off-axis fraction.
- **`δ_RMS`** = RMS geodesic angle between the exact single-axis fit `exp(q_t[u*]_×)` and the real `R_t` (interpretable
  in degrees, avoids tangent-space linearization).
- **`ΔE_axis = E_hinge − E_free`** — loop→framework distance reconstruction, single fixed axis vs free 3D rotation;
  `≈0` ⇒ one axis is enough. *(the strongest test for this application.)*
- **axis spread** `median/p95 ∠(û_t,u*)`, `∠(u*, anchor-anchor line)` structural check, and **replica/block axis
  reproducibility** `∠(u*_r,u*_s)`.
- Report `[P_1D, r_perp, δ_RMS, ΔE_axis, axis spread, replica agreement]`; **compare the distribution across TCRs**
  rather than imposing a universal cutoff. Only if it fails do you investigate changing-hinge regimes / a second
  rigid mode. (Validated: one-axis → P_1D=1.00, δ_RMS=0°, ΔE≈0; two-axis → P_1D=0.58, δ_RMS=10.7°, ΔE=0.52 Å.)

## Steps 3–7 — JOINT hinge + deformation model in distance space
Freeze the axis `L*=(p*,û*)` from the cleanest (least-deformed, low-GPA-residual) frames, then model the observed
loop→framework distance vector as
$$ \boxed{\;\mathbf d_{LF}(t)=\underbrace{\mathbf g(q_t)}_{\text{exact rigid hinge}}+\underbrace{B\,\mathbf z_t}_{\text{deformation footprint}}+\underbrace{\boldsymbol\epsilon_t}_{\text{residual}}\;} $$
- `g(q)` = **exact** nonlinear loop→framework distances of the **undistorted** mean loop (Karcher-oriented GPA
  template — never the coordinate mean, which shrinks) rotated by the finite angle `q` about the frozen `L*`.
- `z_t` = internal-deformation scores = **PCA of `d_LL(t)`**. A rigid hinge cannot change `d_LL`, so `z_t` is an
  **independent** deformation observable; `B z_t` is deformation's footprint on `d_LF`.
- `q_t` and `B` are fit **jointly** (alternating least squares), *not* taken from GPA — so a deformation that
  mimics a rotation is forced into `B z_t` (it has a `d_LL` signal), never swallowed by `q_t`.

Outputs (per-edge normalized so Å-comparable): `F_hinge=√⟨‖δg‖²⟩`, `F_deform_LF=√⟨‖Bδz‖²⟩`, `F_residual=√⟨‖ε‖²⟩`,
and the variance fractions `f_hinge, f_deform_LF, f_residual` (of the loop→framework distance variance — **not** %
of RMSD/Cartesian/entropy). Independent **internal** deformation from `d_LL` about its mean: `F_deform=√⟨D(t)²⟩`,
`D(t)=√(mean_{i<j} δd^{LL}_{ij}(t)²)` (=0 for a rigid loop). Also retain the linear mode `ĥ` as a compact descriptor
for PCA/modeling, and `P(θ)`, `F(θ)=−kT lnP`.

## Step 8 — residual analysis
If `F_residual` is consistently small, hinge + deformation suffices. Else **PCA `ε_t`** for a reproducible second
rigid mode (rocking/twisting); if found, add its `g₂(q₂)` and re-fit. Do not call `ε` "deformation" — that is
measured separately from `d_LL`.

## Both original caveats are now resolved (validated)
1. **Linear-mode curvature → gone.** `f_hinge` uses the **exact finite-rotation** `g(q)`, so a large pure hinge
   gives `f_hinge≈1` (validation 3: **1.001 at ±40°**), not the ~0.84 the linear projection gave.
2. **Deformation mimicking rotation → gone.** Because `z_t` comes only from `d_LL` (which a rigid hinge cannot
   move), a rotation-mimicking drift is forced into `B z_t` (validation 5: `f_hinge=0.003`, `f_deform_LF=0.998`),
   where GPA alone assigned it to the hinge. Still report `f_hinge, F_deform, GPA residual` together.

## Per-CDR fingerprint (organized by graph space — the two live in DIFFERENT spaces)
Mechanistic story: **`d_LL → intrinsic deformation`**; **`d_LF → hinge g(q) + deformation-footprint Bz + residual ε`**.

| group | quantities | space |
|---|---|---|
| **Intrinsic loop deformation** | `F_deform` | LL (loop↔loop) |
| **Framework-relative decomposition** | `F_hinge`, `F_deform_LF`, `F_residual` | LF (loop↔framework) |
| **Variance attribution** (of LF motion, sum→1) | `f_hinge`, `f_deform_LF`, `f_coupling`, `f_residual` | LF |
| **Mechanics** | `L*=(p*,û*)`, `P(q)` → hinge landscape `F(q)=−k_BT ln P(q)+C` | — |

`f_coupling` = the hinge↔deformation cross-term (nonzero when they co-vary; it is what closes the variance budget).
**Do NOT write `F_total = F_hinge + F_deform`** — `F_deform` is an LL observable, the others are LF observables.
Also: consensus-axis stability, GPA Cartesian residual, #deformation modes, loop length, framework rigidity; and
`RMSD_CDR(t)` as a **validation output only** (never `RMSD=H+D+R`; instead test whether the fingerprint explains
the observed CDR RMSD variation).

## Validation (all pass — [`graph_tests.py`](graph_tests.py))
1. SE(3)+reflection invariance (2e-14). 2. **Framework-reference invariance of `q_t`** (3e-14) — no arbitrary
reference. 3. Large (±40°) pure hinge → **`f_hinge=1.001`**, `F_deform≈0` (exact model, curvature caveat gone).
4. Breathing deformation → `f_hinge=0.31`, `F_deform=0.56`. 5. **Rotation-mimicking drift → `f_hinge=0.003`,
`f_deform_LF=0.998`** (deformation not swallowed by the hinge). 6. Pure-distance rigid fit == Kabsch oracle
(27.000°). 7. **Correlated simultaneous hinge+deformation (the hard regime) → hinge angle recovered at corr=0.998,
deformation still detected (`F_deform=0.69`), budget closes to 1.000** — the `d_LL` constraint separates them even
when correlated. Still to add on real MD: anchor-set, template/clean-frame, and replica sensitivity.

## Status & deliverables
- **[`graph_fit.py`](graph_fit.py)** — distance primitives + pure-distance `distance_rigid_fit`. **Done.**
- **[`rotations.py`](rotations.py)** — `to_framework_frame`, `gpa`, `karcher_mean`, `relative_rotvecs`,
  `consensus_axis`, `fit_pivot`. **Done.**
- **[`graph_decomp.py`](graph_decomp.py)** — `deformation_scores`, exact `g_of_q`, `joint_fit`, and **`run_cdr`**
  (the full joint decomposition → the fingerprint incl. `f_coupling`). **Done.**
- **[`hinge_validity.py`](hinge_validity.py)** — standalone Step-2b single-hinge diagnostic
  (`P_1D, r_perp, δ_RMS, ΔE_axis, axis spread, replica agreement`), upstream of the joint model. **Done.**
- **[`graph_tests.py`](graph_tests.py)** — the **9-check** gate (incl. correlated hinge+deformation and
  single-vs-two-axis validity). **All pass.**
- **anchor-stability** [`compute_anchor_stability.py`](compute_anchor_stability.py) +
  [`plot_anchor_stability.py`](plot_anchor_stability.py) — within-MD rigidity + across-TCR conservation of the
  framework anchors (alignment-free). **Done** (22 TCRs): within-MD std median **0.44 Å**; across-TCR distance CV
  median **2.8 %**, <5 % for **80 %** of anchor pairs → the anchors are rigid within every MD and geometrically
  conserved across TCRs.
- `compute_graph_decomp.py` — MD pipeline (per chain, per CDR: Step 1–8, both-flank `anchor_ok`, robust twins). **Next.**
- `axis_conservation.py` (Step 2 cross-TCR `L*`) · `residual_pca.py` (Step 8) · porcupine viz · `plot_*` (Fig 2
  `(F_hinge,F_deform)` scatter · Fig 3 `f_hinge`/`R²` per CDR · Fig 4 consensus-axis porcupines · Fig 5
  bound−unbound Δ) · `RESULTS.md`. **To do.**

## Data
22 unbound TCRs `/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD/<ID>/<ID>.{pdb,xtc}` (per chain, stride 25); Knapp
bound/unbound ensembles for the bound-vs-unbound Δ. API: `load_tcr`, `TrajectoryView.domain_idx` (prefix `A_`/`B_`),
`tv.mdtraj.xyz*10` (Å). `PYTHONNOUSERSITE=1`; `8YJ3` chain override; env `~/.local/share/mamba/envs/kinapse`.
