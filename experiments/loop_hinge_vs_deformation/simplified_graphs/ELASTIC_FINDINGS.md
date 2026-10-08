# Equilibrium-fluctuation elasticity of CDR loops

Effective stiffness inferred from the covariance of **internal strain coordinates** — the standard
near-equilibrium result $U(\mathbf q)\approx\tfrac12\mathbf q^\top K\mathbf q \Rightarrow \boxed{K_{\rm eff}=k_BT\,\Sigma_q^{+}}$.
22 unbound TCRs, `load_md` (kinapse IMGT, stride 25), pooled over 21 (1KGC = ~10× breadth outlier, kept in
`results_elastic/`, out of the pooled means). Alignment-free: every coordinate is an SE(3)-invariant function of the
raw Cα xyz — **rigid motion is the exact zero of every quantity here** (self-test gate: rigid ⇒ Σ_q = 0 to 1e-30,
`k_bend/k_tors` invariant to random rot+trans to 1e-6; `python elastic_fluct.py`). $k_BT=0.596$ kcal/mol (300 K).

Two elastic descriptions, both from the equilibrium covariance, no superposition ever performed.

---

## 1. GLOBAL — normalized-strain compliance/stiffness of the whole loop

Dimensionless internal strain $q_{ij}(t)=\dfrac{d_{ij}(t)-\langle d_{ij}\rangle}{\langle d_{ij}\rangle}$ over all Cα
pairs → compliance $\Sigma_q=\mathrm{Cov}_t[q]$ → stiffness $K_{\rm eff}=k_BT\,\Sigma_q^{+}$. The spectral
decomposition $\Sigma_q v_k=\lambda_k v_k$ gives deformation **modes** with $k_k=k_BT/\lambda_k$ (soft = large
$\lambda$). Normalizing by $\langle d_{ij}\rangle$ makes the compliance **dimensionless and size-comparable across
loops** — this is what lets a 5-residue CDR1 and a 14-residue CDR3 be put on the same axis (the earlier
[`../elasticity/`](../elasticity/) folder inverted the *raw*-distance covariance, Å², which is not cross-loop comparable).

Pooled per CDR (21 TCRs, 126 loops):

| CDR | N | C_mean = tr Σ_q/m | k_soft = kT/λ_max (kcal/mol) | eff_modes | soft_frac1 | multi-basin |
|---|---|---|---|---|---|---|
| A_CDR1 | 5.9 | 0.0020 | **61** | 3.3 | 0.49 | 38 % |
| A_CDR2 | 6.3 | 0.0028 | **48** | 3.7 | 0.46 | 19 % |
| A_CDR3 | 10.7 | 0.0044 | **15** | 5.3 | 0.36 | 52 % |
| B_CDR1 | 5.1 | 0.0013 | **157** | 3.2 | 0.47 | 5 % |
| B_CDR2 | 6.0 | 0.0028 | **68** | 3.8 | 0.46 | 24 % |
| B_CDR3 | 12.0 | 0.0044 | **5.3** | 4.9 | 0.39 | 62 % |

- **A stiffness ladder spanning ~30×:** softest-mode stiffness `k_soft` runs B_CDR1 (157) > A_CDR1 ≈ B_CDR2 (61–68)
  > A_CDR2 (48) ≫ A_CDR3 (15) > **B_CDR3 (5.3)**. β_CDR3's easiest internal deformation costs 30× less energy than
  β_CDR1's — the same rank order the geometric decomposition found (`../simplified_graphs` FINDINGS §4), now as an
  energy.
- **CDR3 is softer AND its flexibility is distributed** (fig `fig_elastic_charactermap`): CDR3 sits at high compliance
  *and* high `eff_modes` (participation ratio 4.9–5.3 vs 3.2–3.8 for CDR1/2) with lower `soft_frac1` (0.36–0.39 vs
  ~0.47) — its softness is spread over many deformation modes, not concentrated in one hinge. CDR1/2 have a single
  dominant soft mode. (Same localized-vs-distributed split as `../elasticity` participation-ratio; here from the
  normalized, dimensionless compliance.)
- **Deformation-mode spectrum** (`fig_elastic_spectrum`): the $k_k=k_BT/\lambda_k$ ladders are gap-separated for
  CDR1/2 (one soft mode, then a jump) and smooth for CDR3 (a soft continuum) — this is the *spectral decomposition of
  a statistically-defined operator*, not a predictive PCA fit (contrast the abandoned `D_H` regression).

---

## 2. LOCAL — discrete elastic-rod stiffness profile along the backbone

Treat the Cα chain as a discrete rod: virtual-bond lengths $l_i$, pseudo bond-angles $\theta_i$, pseudo-dihedrals
$\phi_i$ (all SE(3)-invariant, no alignment). Local harmonic stiffness per position:
$k_s(i)=k_BT/\mathrm{Var}(l_i)$, $k_b(i)=k_BT/\mathrm{Var}(\theta_i)$, $k_t(i)=k_BT/\mathrm{Var}_{\rm circ}(\phi_i)$.
Pooled onto a common contour (0 = N-stem → 1 = C-stem) in `fig_elastic_profiles`.

- **CDR3 is a clamped elastic beam with a soft apex** — a clean **U-shape** in both bending and torsion: stiff at
  both stems, softest at the apex (contour ≈ 0.5–0.6). Apex/stem stiffness ratio:

  | | bending apex/stem | torsion apex/stem |
  |---|---|---|
  | A_CDR3 | 0.63 | **0.26** |
  | B_CDR3 | 0.45 | **0.23** |
  | CDR1/2 | 0.5–1.7 (flat/asymmetric) | 0.76–0.93 (flat) |

  The CDR3 apex is **~4× more torsionally compliant than its stems** — the mechanical signature of a filament
  anchored at both ends with a floppy middle. CDR1/2 profiles are flat or monotonic (no deep central dip); their
  compliance sits toward a stem, not an apex.
- **This directly tests the "one side is clamped" idea** (FINDINGS §2–3, clamps + pivot). The *intrinsic* torsional
  softness of CDR3 collapses from ~13 kcal/mol/rad² at the Cys104 N-stem to ~2 at the apex and rises again toward the
  C-stem — high-stiffness stems bracketing a soft centre, exactly the profile predicted from the clamp geometry.
- **B_CDR2 is the exception that matches the known anomaly:** its bending stiffness is *higher* in the middle than at
  the stems (apex/stem 1.65) — the same one-sidedly-clamped oddity that made B_CDR2 anti-correlate with uniform-spring
  GNM in `../elasticity` (RESULTS §Method 2).

**Global ⇄ local cross-check.** The residue where the *global* softest compliance mode lives coincides with the
*local* soft spot: CDR3's softest mode peaks at mid-contour (0.50–0.55 = apex) while CDR1's peaks at a stem
(contour 0/1). Two independent constructions — collective mode vs per-residue rod stiffness — put CDR3's compliance
at the same place.

---

## 3. Length drives everything (Spearman across 126 loops)

`r(N, k_soft) = −0.79`, `r(N, C_mean) = +0.60`, `r(N, eff_modes) = +0.55`, `r(N, soft_mode_bc) = +0.48`
(all p ≤ 1e-8). Longer loops are softer, more distributed, and more anharmonic — CDR3's mechanics are, to first order,
a length effect (mirrors `../elasticity` `r(N, PR)=+0.89`, `r(N, S)=+0.78`). Magnitudes are sampling-dependent; the
**relative** pattern is the transferable result.

---

## 4. Caveat, made quantitative: where kT/Var is only a LOCAL stiffness

`kT/Var(q)` is an equilibrium stiffness **only inside one approximately harmonic basin**. Per loop we test the
softest-mode projection for multimodality (Sarle bimodality coefficient BC; Gaussian ⇒ 1/3, BC > 5/9 ≈ 0.555 ⇒
multi-basin), `fig_elastic_basins`:

- **B_CDR1 sits on the harmonic line** (BC ≈ 0.33, tight) — there `K_eff` is a legitimate *global* harmonic stiffness.
- **CDR3 is mostly multi-basin** (median BC 0.57 α / 0.60 β; **52 % / 62 % of loops over threshold**) — its soft mode
  hops between conformers, so its `K_eff` / `k_soft` must be read as a **local / effective (per-basin) stiffness**, not
  a single global spring. CDR2 and A_CDR1 are intermediate. This is not a failure of the method — it is the method
  reporting, honestly and per-loop, when the harmonic model is the right description and when it is a local one.

---

## Files
`elastic_fluct.py` (primitives + self-test gate) · `compute_elastic_fluct.py` (batch, imports `graph_build.load_md`
verbatim; insertion-safe loop selection) · `plot_elastic_fluct.py` (pooling + figures). Per-system
`results_elastic/<ID>_elastic.npz` (per-CDR `lam`, `k_mode`, `soft_footprint`, `k_stretch/k_bend/k_torsion`,
`resnums`) + `_percdr.csv`; `master_percdr.csv` (132 rows), `summary_percdr.csv`. Figures `figures/fig_elastic_*`.
Relation to [`../elasticity/`](../elasticity/): that folder inverted the raw-distance covariance (Method 3) + GNM/ANM;
this adds the **dimensionless normalized-strain** compliance (size-comparable), the **local discrete-rod stiffness
profile** (planned in [`../elastic_rod/`](../elastic_rod/), implemented here), and a **per-loop harmonic-basin flag**.
Run: `mamba activate kinapse; python elastic_fluct.py; python compute_elastic_fluct.py; python plot_elastic_fluct.py`.
