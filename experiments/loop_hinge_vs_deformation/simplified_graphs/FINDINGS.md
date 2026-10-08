# TCR CDR-loop motion: rigid framework, clamps, and hinge-vs-deformation

Alignment-free analysis of 22 unbound TCR MD trajectories (kinapse IMGT numbering; α = chain A, β = chain B).
Everything below is derived from CA–CA distances and per-frame geometric rigid fits — no arbitrary superposition.

---

## 1. A fluctuation-derived RIGID framework (per chain, transferable)

From the **22-TCR-averaged** CA–CA distance fluctuation `D_std` we define, alignment-free:

- **kept** = largest residue set with *average* pairwise `D_std ≤ 0.5 Å` (mutually rigid on average).
- **RIGID** (use this) = ultra-rigid subset of *kept* whose **worst-case** (max over TCRs) pairwise `D_std ≤ 0.5 Å`.

  - **Chain A (13):** 40, 41, 89, 90, 91, 101–106, 124, 125
  - **Chain B (20):** 6, 10, 21, 38–43, 52, 53, 101–106, 121–123

Details: `RIGID_FRAMEWORK.md`, `rigid_framework.json`, figures `rigidpick_avg_chain{A,B}.png`.

- The **worst-case** cap matters: **Cys104** is bedrock-rigid (worst-case 0.43) and is in the set; **Cys23** is
  only *average*-rigid (mean 0.29 but worst-case 0.60–0.72) and is excluded — it sits on the peripheral B-strand
  and can swing around the disulfide while keeping Cys23–Cys104 fixed (that pair's `D_std` = 0.17 Å).
- Rigidity extends **from the F-strand through CDR3 residues 105–106** (the V-germline C-A-x stem) and **breaks
  sharply at 107** (worst-case jumps to 0.8–1.2 Å) — that 106→107 edge is the physical edge of the clamp.
- vs kinapse consensus: kinapse uses **cross-TCR alignment RMSD < ~2.0 Å** (structural *conservation*), a different
  metric; that ~80-residue set maps to roughly `τ ≈ 0.8 Å` on our within-MD fluctuation curve.

---

## 2. Each CDR loop is asymmetrically CLAMPED on one stem

Measured (fluctuation-to-rigid at each stem) and confirmed 6/6 across both chains:

| CDR | clamped stem | consistent α & β |
|---|---|---|
| CDR1 | **C-terminal** | yes |
| CDR2 | **N-terminal** | yes |
| CDR3 | **N-terminal** | yes |

**What sets the clamped side = the rigidity of the flanking β-strand, NOT proximity to a conserved residue.**
The stiffer flanking strand predicts the clamped stem 6/6. Mechanism (DSSP + burial):

- A strand clamps if it is a **buried, continuously H-bonded β-strand all the way to the loop stem.**
  CDR1's C-flank (C-strand, 39–41) is all-strand and **fully buried (SASA ≈ 0)** → clamps.
- CDR1's N-flank (B-strand end, 24–26) **frays to coil and is solvent-exposed (SASA 42–69)** → no clamp,
  even though Cys23 is disulfide-pinned just upstream. **Proximity to a conserved residue is neither necessary
  (CDR2 has none, still clamped) nor sufficient (Cys23 is adjacent, no clamp).**
- Conserved residues act by **stiffening whole strands** (the Cys23–Cys104 disulfide rigidifies CDR3's F-strand
  N-flank; Trp41, the buried core keystone, rigidifies CDR1's C-strand), not by clamping adjacent loops directly.

Conserved landmarks (annotated in `loops_flanks_fluctuation_annotated.png`): Cys23/Cys104 disulfide, Trp41,
hydrophobic-89, and the J-motif **F–G–X–G** (118–121, the mildly-flexible G-X-G turn just after CDR3).

---

## 3. The loops pivot ABOUT their clamped stem (direct test)

Per-frame 6-DOF rigid fit → pivot located along the N→C stem axis (0 = N-stem, 1 = C-stem):

| CDR | median pivot | clamp | frames on clamp side |
|---|---|---|---|
| A/B_CDR1 | 0.78 / 0.79 | C | 22/22, 22/22 |
| A/B_CDR2 | 0.39 / 0.18 | N | 19/22, 20/22 |
| A/B_CDR3 | 0.18 / 0.35 | N | 21/22, 20/22 |

Each loop's rotation axis passes through its clamped stem in 19–22 of 22 TCRs → the clamp *is* the hinge point.

---

## 4. Rigid (hinge-like) vs deformation — geometric decomposition

Per frame, fit the undeformed reference loop's 6-DOF rigid pose (closed-form Kabsch; **no PCA/regression/axis**),
then, in distance space: `D_deform` (internal shape change), `D_rigid` (rigid loop→framework motion),
`E_nonrigid` (rigid-fit error = how much to trust the rigid description). See `geomhinge_pooled.png`.

**Reliable per-TCR metric — rigid fraction of framework-relative motion = `D_rigid²/(D_rigid²+E²)`** (dimensionless):

| CDR | rigid fraction | IQR (22 TCRs) | med E_nonrigid | verdict |
|---|---|---|---|---|
| A/B_CDR1 | 0.81 / 0.78 | 0.73–0.84 | 0.21 / 0.16 | hinge-like, reliable |
| A/B_CDR2 | 0.79 / 0.83 | 0.75–0.87 | 0.19 / 0.16 | hinge-like, reliable |
| A_CDR3 | 0.61 | 0.47–0.68 | 0.26 | mixed, report E |
| B_CDR3 | 0.57 | 0.51–0.63 | 0.31 | deformation-heavy, report E |

Quadrant fractions (D_deform>0.4, D_rigid>0.4 Å; % of frames):

| CDR | stable | hinge | deform | both |
|---|---|---|---|---|
| A_CDR1 | 42 | 30 | 4 | 11 |
| A_CDR2 | 45 | 36 | 6 | 9 |
| A_CDR3 | 54 | 9 | 13 | 14 |
| B_CDR1 | 66 | 25 | 1 | 2 |
| B_CDR2 | 56 | 32 | 1 | 4 |
| B_CDR3 | 37 | 5 | 17 | 25 |

**Headline:** CDR1 & CDR2 move ~80% as **rigid, hinge-like bodies** relative to the framework (clean, tight across
TCRs). **CDR3 is deformation-heavy** (β_CDR3: ~42% of frames involve real internal deformation), and the rigid/hinge
reading there must always be reported with `E_nonrigid` because a rigid model fits those frames poorly.

**Reliability caveat:** absolute `D_deform`/`D_rigid` magnitudes scale with loop length (not comparable across CDRs);
the **fractions are dimensionless and comparable**. An earlier PCA-regression residual (`D_H`) was abandoned — it
correlated only 0.38–0.70 with the true geometric hinge, worst for CDR3.

---

## Scripts / data
`graph_build.py` (load + graph), `compute_fluct.py`/`aggregate_fluct.py` (rigid set), `loop_flank_analysis.py`
(clamps + annotated figure), `pivot_test.py` (hinge axis), `geom_hinge.py`/`compute_geom.py`/`plot_geom_pooled.py`
(rigid-fit decomposition). Results in `results_fluct/`, `results_pivot/`, `results_geom/`; figures in `figures/`.
