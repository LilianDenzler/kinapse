# PLAN — loop hinge vs. deformation (alignment-free, distance-matrix decomposition)

**Question.** For each MD ensemble, decompose every CDR-loop conformation into **rigid re-orientation
(hinge)** vs. **internal shape change (deformation)** — *without ever superposing coordinates*. Everything
is built from SE(3)-invariant **distance matrices** anchored to a rigid Vα-framework reference, so no Kabsch
alignment is required and the inter-domain (Vα–Vβ) hinge cannot pollute the reference frame.

## Why this experiment (relation to prior work)

The sibling study [`interface_analysis/MD_flexibility/`](../interface_analysis/MD_flexibility/) already asked
this and concluded **"CDR loops move as rigid hinges, not by deforming" (~92 % hinge)** — but via a
**Kabsch/RMSF heuristic**: superpose each frame on the α+β framework Cα, take `total` = framework-aligned loop
RMSF, `deform` = loop-self-aligned RMSF, `hinge = √(total²−deform²)`. Two weaknesses we fix here:

1. **Global α+β framework alignment mixes in the Vα–Vβ inter-domain hinge.** The unbound trajectories flex
   their inter-domain "ABangle" (`BA`) substantially — established in
   [`unbound_TCR_MD_angles/`](../unbound_TCR_MD_angles/). A loop's "hinge" measured in a frame that itself
   wobbles about the domain–domain joint is contaminated. → Anchor to a **single-chain (α) rigid core** instead.
2. **`√(total²−deform²)` is a heuristic split, not a real hinge.** It yields a magnitude, not an **angle**, and
   never tests whether a rigid model actually *explains* the motion.

This experiment reproduces the hinge-vs-deform verdict from an **independent, alignment-free** formulation and
upgrades it to a genuine **hinge angle θ** plus a **non-rigidity residual E_nonrigid** that quantifies how badly
a pure-hinge model fails. Agreement cross-validates the earlier result; disagreement localizes exactly where
the old global-framework alignment was polluted by inter-domain motion — that is the scientific payoff.

## Scientific invariants (do not violate)
- **No coordinate superposition in the core metrics.** `D_deform` and `D_framework` are computed directly from
  distances — invariant to arbitrary rotation/translation/reflection of any frame. Only the `θ` *solver*
  (Stage E) touches coordinates, and only as a fast, provably-equivalent stand-in for a distance-defined objective.
- **Transferable method** (per project policy — no per-loop / per-system tuning). Rigid-core detection,
  reference choice, and quadrant thresholds are **data-driven** (gap/quantile based), identical for all systems
  and all six CDRs. See [[prefer-transferable-methods]].
- **α-chain rigid core is the invariant anchor** (single chain → free of the Vα–Vβ inter-domain hinge). β-loops
  are additionally scored against the α core to *deliberately* capture inter-domain motion (see §7).
- Report **relative** patterns; magnitudes are sampling-dependent. Per-TCR normalisation as in MD_flexibility
  (1KGC is a ~10× breadth outlier — keep but flag).

---

## What separates cleanly — and what does not (read before trusting any "hinge" number)

The two halves of this decomposition are **not symmetric**, and that is a theorem, not a weakness of the method
(Eckart 1935; Guichardet 1984; Littlejohn & Reinsch, *Rev. Mod. Phys.* **69**, 213, 1997):

- **Deformation is exactly separable.** `D_deform` is built only from intra-loop distances — the complete set of
  SE(3) invariants of the loop — so it is a pure shape observable with **zero** pose contamination:
  `D_deform = 0 ⟺ identical shape` (up to rigid motion + reflection). A loop can hinge 50° and `D_deform` stays 0.
- **Hinge is not completely separable for finite motion.** "How much did a *deformable* body rotate" has no
  canonical answer — the rotation attributed to a shape change is path-dependent (holonomy, the falling-cat
  effect). Any hinge number must pick a convention, and conventions agree only **to first order in the deformation**.

Consequences that shape the stages below:
- **`D_framework` is a *reorientation proxy*, not a hinge angle.** It is a displacement in Å scaling as
  `≈ (hinge angle) × (loop lever arm) + (deformation footprint)` — apex-weighted, not scale-free, not comparable
  across loops of different size. Keep it as an axis; never call it "the hinge."
- **The best-fit rotation is the Eckart frame** — the unique convention that removes the first-order
  rotation–deformation coupling. So the fitted `θ` (Stage E) is the confound-free version of `D_framework`, and
  `E_nonrigid` measures the irreducible leftover, with an explicit error bar `δθ ≈ E_nonrigid / ℓ`.
- **Do not scalar-subtract `D_framework − D_deform`.** Deformation perturbs the two distance sets by different,
  geometry-dependent amounts (distance is nonlinear), so the remainder is not hinge and can even go negative. The
  geometrically-correct subtraction *is* the rigid fit in Stage E.
- Net honest decomposition: **exact deformation (`D_deform`) + Eckart-optimal hinge (`θ ± δθ`) + measured residual
  (`E_nonrigid`).** There is no exact `total² = hinge² + deform²` — that Pythagorean split (which the old
  `√(total²−deform²)` silently assumed) holds only in the small-deformation limit. CDR-loop deformation is sub-Å,
  so we sit in that limit and `E_nonrigid`/`δθ` prove it frame by frame.

---

## Method

Loop nodes `x_i`, i=1..N (a CDR's Cα atoms). Rigid scaffold anchors `a_j`, j=1..M (the α-framework rigid core).
Superscript `(k)` = frame k; `(0)` = reference conformation. **All coordinates in Å** (mdtraj `xyz` is nm → ×10).

### Stage A — rigid α-framework core, found *without alignment*
Candidate set = α-framework Cα (`A_FR1,A_FR2,A_FR3`; FR4/J-region excluded — it is mobile, matching the prior
FR1–3 choice). A set is rigid iff its **internal pairwise distances are frame-invariant**, so we never align:

1. For every candidate pair (p,q): `mean_k D_pq`, `std_k D_pq`, coefficient of variation `cv_pq = std/mean`.
2. Iteratively drop the atom with the largest median `cv` to other retained atoms until all retained pairwise
   `cv ≤ τ`, where `τ` is chosen from the **gap in the sorted cv distribution** (data-driven, transferable) with
   a floor (~2 %). Keep the largest surviving clique (≥ ~20 atoms) → the anchor set `a_j`.
3. **Validation (only to confirm rigidity — not used by the metrics):** Kabsch-superpose the core onto frame 0;
   require per-atom RMSF ≲ 0.5 Å ("aligns perfectly each frame"). Log core size, τ, RMSF, worst residual.

This core **doubles as the anchor set `a_m`** for Stages D–E (~20 atoms, inherently ≥4 non-coplanar — see the
anchor rule in Stage D). Per loop, optionally restrict to the core atoms nearest that loop's stem for locality.

### Stage B — reference conformation `(0)`
Two references, both reported:
- **Ensemble mean** (primary): `D^(0)_ij = mean_k D^(k)_ij`, `C^(0)_ij = mean_k C^(k)_ij` (mean *of distances* —
  stays invariant; not distances of mean coordinates). Makes `D_deform`,`D_framework` measure spread about the
  average shape, directly comparable to RMSF.
- **Crystal / frame 0** (secondary): "distance from the experimental structure."

### Stage C — internal deformation = the *exact* deformation term (pure distance)
For each frame, loop Euclidean distance matrix `D^(k)_ij = ‖x_i^(k) − x_j^(k)‖`, then

$$ D_{\rm deform}^{(k)} = \sqrt{ \tfrac{2}{N(N-1)} \sum_{i<j}\left(D_{ij}^{(k)}-D_{ij}^{(0)}\right)^2 }. $$

`D_deform = 0` ⇒ identical loop shape up to rigid motion/reflection (the loop may hinge freely and this stays 0).
Report both **all pairs** and a **sequence-separation-filtered** variant (`|i−j| ≥ 2`, drops trivially-stiff
bonded neighbours) to show the apex-vs-anchor deformation profile.

### Stage D — framework-relative displacement (reorientation *proxy*) + the 2-D map
Loop→anchor cross-distance matrix `C^(k)_im = ‖x_i^(k) − a_m^(k)‖` over the `M` framework anchors, then

$$ D_{\rm framework}^{(k)} = \sqrt{ \tfrac{1}{NM}\sum_{i,m}\left(C_{im}^{(k)}-C_{im}^{(0)}\right)^2 }. $$

This is the robust, **fit-free base descriptor**: how much the loop moved *relative to the framework*. Paired with
`D_deform` it already answers the question qualitatively via the 2-D map `(D_framework, D_deform)` (thresholds =
per-loop medians, or a small multiple of the deformation noise floor — data-driven):
- large `D_framework`, small `D_deform` → rigid hinge-like reorientation
- small `D_framework`, large `D_deform` → local deformation
- large both → hinge + deformation · small both → unchanged.

Optional scalar **hinge-dominance** `H = D_framework / (D_framework + D_deform)` (∈[0,1]) for *ranking only*.
Caveat: `D_framework` is a displacement, not an angle (`≈ angle × lever arm`), so `H` is biased toward "hinge" for
larger loops — use within-loop, **not** for cross-loop magnitude and **not** against MD_flexibility's
variance-based 92 %. For a scale-free hinge, go to Stage E. **Never** compute `D_framework − D_deform` (see §"What
separates cleanly").

**Anchor rule (critical).** Use `M ≥ 4` **non-coplanar** framework Cα spread around the loop base — not one, not
collinear. One anchor is blind to rotation on its sphere; 3 non-collinear anchors trilaterate each loop Cα up to a
mirror; 4 non-coplanar fix it and average noise. Default: the **entire Stage-A rigid core** (M ~ 20, inherently
non-coplanar); if restricting to a loop-local subset, enforce non-coplanarity (max-tetrahedron-volume pick).

### Stage E — the hinge, done right: fitted angle `θ` (the confound-free `D_framework`)
`D_framework` conflates angle × lever arm × deformation footprint. To recover a **scale-free hinge angle**, fit the
rigid-body motion of the loop that best reproduces the loop↔framework distances *starting from the reference loop
shape* — this is the geometrically-correct "subtract the deformation out of `D_framework`":

$$ (R^\*,t^\*) = \arg\min_{R\in SO(3),\,t}\; \sum_{i,m}\left[\,C_{im}^{(k)}-\big\|(R\,x_i^{(0)}+t)-a_m^{(k)}\big\|\,\right]^2. $$

By **Chasles**, `(R^*,t^*)` is a screw: report its **axis** (where the hinge physically lives), **angle**
`θ = cos⁻¹((tr R^* − 1)/2)` (independent of any hinge-point choice), and pitch. Residual + error bar:

$$ E_{\rm nonrigid} = \sqrt{ \tfrac{1}{NM}\sum_{i,m}\left[\,C_{im}^{(k)}-\big\|(R^\*x_i^{(0)}+t^\*)-a_m^{(k)}\big\|\,\right]^2 }, \qquad \delta\theta \approx \frac{E_{\rm nonrigid}}{\ell}, $$

`ℓ` = RMS perpendicular distance of the loop Cα from the screw axis (the lever arm). **`δθ` is the honest hinge
error bar: exact (→0) when the loop is rigid, growing with deformation** — the coupling made quantitative.

- **Why this is your subtraction, done exactly:** it uses the loop↔framework distances `C` *and* the fixed
  reference shape `x^{(0)}`, so it removes only the pose-explainable part and leaves `E_nonrigid` = the
  deformation's true footprint on `C` — direction-aware, never negative.
- **Reference implementation (alignment-free):** minimise the distance residual directly over SE(3) (6 params, LM
  on quaternion + translation) — no coordinates ever placed in a common frame.
- **Scale implementation (fast, provably equivalent):** in the Stage-A scaffold frame (rigid core → RMSD ≈ 0), a
  single Procrustes of `x^{(0)} → x^{(k)}` gives the same `R^*`, `θ`, `E_nonrigid`. **Assert agreement** with the
  distance-LM on a random subset (θ ≲ 1e-3°); use the fast route at scale. Alignment is only a *solver* here.
- **Hinge fraction** (comparable-in-spirit to MD_flexibility's 92 %):
  `1 − E_nonrigid² / Var_{im}(C^{(k)}_im − C^{(0)}_im)` — fraction of the framework-relative motion a rigid loop
  explains.

### Stage F — attachment-axis fingerprint (secondary, per-residue)
Axis from flanking stems `u = (x_C − x_N)/‖x_C − x_N‖` (N,C = 104,118 Cα). Per loop node, longitudinal
`z_i = (x_i − x_N)·u` and perpendicular `r_i = ‖(x_i − x_N) − z_i u‖`. Both rigid-invariant; give the apex
splay/protrusion profile and localise *where* deformation happens (complements MD_flexibility's apex analysis).
Note `r_i,z_i` discard rotation about `u`, so the framework-anchor representation (`D_framework`/`θ`) stays primary.

---

## Data

**Primary:** `/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD/` — **22 unbound TCRs** (chains A=α, B=β,
variable+constant, ~430–450 res), each `<ID>/<ID>.pdb` (topology) + `<ID>/<ID>.xtc` (~57k frames, ~1.4–1.5 GB).
Systems: `1KGC 2BNU 2CDF 2CDG 3DX9 3QEU 3QH3 3SKN 3VXQ 3VXT 4DZB 4JFH 4UDT 4ZDH 6FRA 6OVN 6XQQ 7EA6 7N1D 7R7Z
7S8I 8YJ3`. Stride **25** → ~2.3k frames/system (full optional). Same set the prior study used → direct compare.

**Other roots (drop-in, same code, different manifest):**
- **Knapp bound/unbound replica ensembles** `/mnt/larry/lilian/DATA/Unbound_Bound/MD_runs_unbound_bound/<SYS>/`
  — `SYS ∈ {LC13, A6, JM22, 1G4}`, per replica `<...>_TCR{,pMHC}_run<N>/` with `*.firstFrame.pdb` +
  `*.final.md.xtc` (801 frames = 100 ns; LC13 100+100 reps, others 10+10). Discover by glob, **pool replicas**
  (concatenate frames per state) before Stage B. Enables the **bound-vs-unbound** hinge/deform contrast — does a
  loop's hinge basin narrow on binding? — and reuses `LC13_calculations/config*.yaml` as the ready-made manifest.
- **New A6 GROMACS campaign** `/mnt/larry/lilian/DATA/A6_extended_MD/` (`prod.xtc`+`prod.tpr`, full-solvent) —
  needs a `gmx trjconv` protein strip → `prot.{pdb,xtc}` before `load_tcr` (still accumulating, ~9 reps).

**Dataset-agnostic loader.** A `config` system→files map (default = the 22) selects any of the above; the compute
path is identical once a `(topology.pdb, trajectory.xtc)` pair (or a replica glob to concatenate) is resolved.

---

## Implementation (reuse the kinapse API)

Loading + region/atom selection — confirmed against the source and the working
[`compute_md_flexibility.py`](../interface_analysis/MD_flexibility/compute_md_flexibility.py):

```python
from kinapse.structures import load_tcr
import mdtraj as md, numpy as np
md.load(xtc, top=pdb, stride=STRIDE).save_xtc(tmp)          # stride first (memory)
tv  = load_tcr(pdb, traj=tmp).pairs[0].traj                 # TrajectoryView (None if no traj)
core = np.asarray(tv.domain_idx(["A_FR1","A_FR2","A_FR3"], atom_names={"CA"}))   # anchor candidates
idx, names = tv.domain_idx(["A_CDR3"], atom_names={"CA"}, pass_names=True)        # loop nodes (+IMGT names)
X = tv.mdtraj.xyz[:, np.asarray(idx)] * 10.0                # (n_frames, N, 3) Å
A = tv.mdtraj.xyz[:, core]           * 10.0                # (n_frames, M, 3) Å  — anchors
# D_deform, D_framework are pure numpy on X,A; θ from one Procrustes in the scaffold frame. Stems 104/118 from `names`.
```

- **Region prefixes `A_`/`B_` drive chain selection** (not the literal chain letter) → works for γδ too.
- `domain_idx` returns **topology-order** indices; keep the `pass_names` list to map back to (chain, IMGT, atom).
- **Vectorise the distance matrices** across frames: `np.linalg.norm(X[:,:,None,:]-X[:,None,:,:], -1)` etc.
- Free-energy surfaces over `(θ, D_deform)` / `(D_framework, D_deform)`: reuse
  `kinapse.dynamics_analysis.compute_pmf_kde` + `identify_high_density_points`; per-CDR mean distance matrices via
  the same `_cadist.npz` idiom as the prior study.

### Robustness (inherited from the working MD scripts — do not rediscover)
- **`PYTHONNOUSERSITE=1` re-exec at top of script** (env's user-site has an ABI-broken h5py that crashes MDA).
- **No periodic box** in these merged trajectories → a contiguous block of frames has split/displaced domains.
  Detect via the Vα–Vβ interface Cα distance, mark `valid=False`, **exclude from statistics** (keep in output).
  Single-chain α anchoring is largely immune, but still gate for the β-vs-α cross-analysis (§7).
- **ANARCII mistyping:** `8YJ3` needs `load_tcr(..., manual_chain_types={'A':'B','B':'A'})` (α typed as δ); keep a
  per-system override map. Wrap each system in try/except, continue on failure (resumable).
- Units nm→Å (×10); `n_frames` preserved through the per-pair rebind (RuntimeError guard on atom-count mismatch).

---

## Validation / unit tests (must pass before trusting numbers)
1. **Invariance:** apply a random rotation+translation (and a reflection) to a frame → `D_deform`, `D_framework`
   unchanged to ~1e-6. (The whole premise.)
2. **Rigid core is rigid:** core internal-distance `cv ≲` few %, Kabsch RMSF ≲ 0.5 Å.
3. **Synthetic recovery:** inject a known pure rotation of the reference loop → recover `θ` (exactly),
   `E_nonrigid≈0`, `δθ≈0`, `D_deform≈0`; inject a pure local perturbation → `θ≈0`, `E_nonrigid>0`, `D_deform>0`;
   inject rotation+perturbation → `θ` recovered within `δθ`.
4. **Solver agreement:** distance-LM `(R^*,t^*)` vs scaffold-frame Procrustes agree (θ ≲1e-3°) on a random subset.
5. **Anchor sufficiency:** with `M<4` or coplanar anchors a synthetic pure rotation is under-recovered; assert
   `M≥4` non-coplanar recovers `θ` exactly (guards the Stage-D anchor rule).

## Outputs
`results/` (git-ignored): per-system `<ID>_perframe.parquet` (frame, valid, per-CDR `D_deform`,`D_deform_sep`,
`D_framework`,`H`,`θ`,`delta_theta`,`E_nonrigid`,`hinge_frac`, screw-axis), `<ID>_core.json` (anchor set, τ, RMSF),
`<ID>_axisprofile.npz` (`r_i`,`z_i`), `<ID>_cadist.npz` (mean D per CDR). Aggregated `master_perframe.parquet`,
`master_percdr.csv`.
`figures/`: (f1) `(D_framework, D_deform)` quadrant scatter per CDR; (f2) `θ ± δθ` and `E_nonrigid` distributions +
hinge-fraction per CDR (overlay MD_flexibility's 92 %); (f3) PMF over `(θ, D_deform)` with basins; (f4)
apex-centred `r_i` splay profile; (f5) α-anchored vs own-chain-anchored β-loop `θ` (isolates inter-domain hinge).

## Deliverables / execution
- `compute_hinge_deform.py` — per-system (stride → load → rigid core → D/C → `D_deform`/`D_framework`/`θ`/`δθ`/
  `E_nonrigid` → save), resumable, background.
- `plot_hinge_deform.py` — aggregate + figures + `RESULTS.md`.
- `config.py` (system→files map, overrides, τ floor, stride) · `README.md` (points here) · this `plan.md`.
- Run: `conda activate kinapse; python compute_hinge_deform.py [--systems 1KGC 8YJ3] [--full]`.

## Open decisions (resolve at first run, keep transferable)
- Reference = ensemble-mean (primary) vs crystal — report both; pick mean for headline.
- Quadrant thresholds: per-loop medians vs global deformation noise floor — pick and fix once, apply to all.
- Primary focus **α CDR3** (A_CDR3, IMGT 105–117); run all six CDRs for the transferable comparison.
