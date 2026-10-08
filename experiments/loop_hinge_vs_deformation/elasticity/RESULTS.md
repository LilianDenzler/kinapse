# Results — elasticity theory of CDR loops (three formalisms)

22 unbound TCRs (`CORY_ORIOL_MERGED_MD`), stride 25 (~2.3k frames/system). Every metric is **intrinsic to the loop**
(referenced to its own `d_LL` **medoid** frame) → invariant to rigid motion and immune to the merged-trajectory
split-domain issue; no framework/box gating. **Rigid motion is the zero-strain null space** of all three formalisms
(verified: `tests.py` 11/11 — rigid ⇒ `E_i=0` to 1e-14, `Σ=0` to 1e-30). **IMGT insertion-code guard passed on all 22
systems** (`assert_no_dropped_residues`): mdtraj CA count == Bio.PDB (icodes intact) == sequence, for every CDR — no
apex residue dropped (insertions kept as repeated resSeq, e.g. `1KGC A_CDR3 …111,112,112,113…`; short CDR3s show the
correct IMGT deletion gap, not a lost atom). Pooled over 21 TCRs (1KGC = ~10× breadth outlier, kept in `results/`).

## Headline

**Three independent elasticity formalisms converge on one story: CDR3 deformation is non-affine, anisotropically
soft, and entropy-rich — it cannot be captured by uniform-spring elasticity *or* a single hinge; CDR1/2 are affine,
localized, and well described by a plain elastic network.** The methods agree edge-to-edge: non-affine strain and
configurational entropy correlate at **r = 0.91** across 126 loops.

| CDR | N | shear ⟨‖devE‖⟩ | non-affine ⟨D²min⟩ | GNM corr | soft-mode PR | soft frac₁ | QH entropy (nats) |
|---|---|---|---|---|---|---|---|
| A_CDR1 | 5.9 | 0.33 | 0.19 | **0.70** | 3.8 | 0.56 | −5.2 |
| A_CDR2 | 6.3 | 0.22 | 0.15 | **0.51** | 4.1 | 0.50 | −4.5 |
| A_CDR3 | 10.7 | 0.18 | **0.39** | **0.06** | 6.6 | 0.38 | **+3.1** |
| B_CDR1 | 5.1 | 0.44 | 0.09 | **0.83** | 3.4 | 0.55 | −6.9 |
| B_CDR2 | 6.0 | 0.15 | 0.15 | **−0.07** | 3.9 | 0.49 | −5.4 |
| B_CDR3 | 12.0 | 0.19 | **0.45** | **0.08** | 7.5 | 0.40 | **+7.4** |

(PR = participation ratio of the softest compliance mode: low = localized/hinge-like, high = distributed/collective.
soft frac₁ = variance fraction in the softest mode. GNM corr = agreement of uniform-spring predicted mobility with
the observed alignment-free per-residue mobility.)

## Method 1 — finite-strain field: deformation is non-affine and lives at the apex (fig1, fig2)

- **CDR3 `D²min` profile is the clamped-elastic-beam signature** (fig1): ≈0 at both stems (rigidly anchored) rising
  to a broad **apex plateau** (`D²min ≈ 0.5–0.6`), while the *affine* shear `‖dev E‖` stays low and flat (~0.15–0.25).
  The loop's shape change at the apex is **non-affine** — a genuine local rearrangement no homogeneous strain
  (stretch/shear) reproduces — not smooth bending. CDR1/2 profiles are flat and low.
- **Character map (fig2):** CDR3 loops sit high on the non-affine axis at *low* shear; CDR1/2 cluster at low
  non-affinity. `⟨D²min⟩`: CDR3 **0.42** vs CDR1/2 **0.15**.
- Length drives non-affinity (`r(N, D²min) = +0.63`) but **not** affine shear (`r = −0.29`) — longer loops deform more,
  and the extra deformation is non-affine.

## Method 2 — elastic network: uniform-spring elasticity explains CDR1/2, fails for CDR3 (fig3)

- GNM per-residue mobility matches the observed mobility for the short loops (**B_CDR1 0.83, A_CDR1 0.70, A_CDR2 0.51**)
  but is **at chance for CDR3** (A_CDR3 0.06, B_CDR3 0.08); **28/42 CDR3 loops have corr < 0.2**. A single isotropic
  spring constant per contact captures the framework-proximal loops but not the apex — CDR3 is **anisotropically
  soft** in a way the contact topology alone does not predict. `r(N, GNM corr) = −0.50`: the longer the loop, the worse
  uniform elasticity does.
- **B_CDR2 anti-correlates** (mean −0.07; negative in 17/21 TCRs): its most-mobile residue is where GNM predicts the
  *least* motion — a short, one-sidedly-clamped loop whose mobility is set by something the loop-only contact graph
  gets backwards (candidate for adding a framework clamp; see caveats).

## Method 3 — data-driven stiffness: CDR3 softness is distributed; entropy tracks non-affinity (fig4)

- **Softest compliance mode is localized for CDR1/2 (PR ≈ 3.4–4.1) and distributed for CDR3 (PR ≈ 6.6–7.5)** — the
  short loops have a single soft spot; CDR3 is collectively soft along the apex. `r(N, PR) = +0.89`.
- **Quasi-harmonic entropy** (redundancy-safe, over the real 3N−6 internal DOF): strongly positive for CDR3
  (A_CDR3 +3.1, B_CDR3 +7.4) vs negative for CDR1/2 — CDR3 explores far more configurational volume. `r(N, S) = +0.78`.
- **Cross-method convergence:** `D²min` (strain, Method 1) vs `S` (compliance, Method 3) correlate at **r = 0.91** —
  two independent constructions of "how much does this loop deform" agree, validating the picture.

## How this answers "define the rigid part" (the design question)

Rigid reorientation is **never fit** — it is the zero of every deformation observable here (`E_i=0` / the GNM zero
mode / zero `d_LL` variance), so a hinge and a simultaneous deformation separate with no superposition and no gauge.
What remains after the rigid null space is exactly what these three methods characterize, and they agree it is, for
CDR3, **non-affine + distributed + entropy-rich** — i.e. *not* a single hinge coordinate. The optional Eckart hinge
*angle* (framework-referenced, imported from `../graph_method`) is a separate follow-up and is deliberately not
computed here, since the loop-intrinsic metrics need no framework.

## Caveats

- **Unbound** TCRs → intrinsic dynamics, not binding-induced (bound-vs-unbound Δ on the Knapp sets is the planned
  follow-up: does binding stiffen the apex / lower `S`?).
- **1KGC** excluded from pooled means (large transition inflates magnitudes ~10×); kept in `results/`.
- Magnitudes are ensemble-sampling-dependent; the **relative** pattern (CDR3 non-affine/soft/entropic, CDR1/2
  affine/localized) is the transferable finding. Entropy is a relative QH estimate, not an absolute free energy.
- GNM/ANM are built on the **loop only** (intrinsic, alignment-free); adding the flanking framework stems as fixed
  clamps would make the network more physical and likely fix B_CDR2's anti-correlation — a cheap next step.
- Reflection is unresolved by these symmetric-tensor / distance observables (documented, not a bug).

## Files
Per system `<ID>_elasticity.npz` (per-CDR `shear_by_res`, `vol_by_res`, `d2min_by_res`, `gnm_msf`, `obs_mobility`,
`soft_mode_residue`, `shear_frame`, `d2min_frame`, `compliance_eig`, `resnums`) + `<ID>_percdr.csv`. Aggregated
`master_percdr.csv` (132 rows), `summary_percdr.csv`. Figures: `fig1_strain_profiles` (D²min/shear along each CDR) ·
`fig2_character_map` (non-affine vs shear per loop) · `fig3_gnm_explains` (uniform elasticity corr per CDR) ·
`fig4_softmode_entropy` (localized-vs-distributed softest mode; QH entropy per CDR).
Scripts: `elast.py` (primitives), `compute_elasticity.py`, `plot_elasticity.py`, `preflight.py` (insertion guard),
`tests.py` (11-check gate). Run: `mamba activate kinapse; python tests.py; python compute_elasticity.py; python plot_elasticity.py`.
