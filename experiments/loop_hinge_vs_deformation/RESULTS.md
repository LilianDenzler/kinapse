# Results — loop hinge vs. deformation (alignment-free, distance-matrix)

22 unbound TCRs (`CORY_ORIOL_MERGED_MD`), strided 25× (~2.3k frames/system). Per CDR loop, relative to the
**ensemble medoid** and its per-chain **rigid Vα/Vβ-framework core** (found alignment-free: maximal FR1–3 Cα
subset with pairwise-distance std ≤ 0.8 Å; typ. **69–71 Cα, RMSF ≈ 0.5 Å** = "aligns near-perfectly each frame").
Everything below is from **distance matrices only** — no Kabsch enters any metric; a single Procrustes enters
only the `θ` solver (validated equivalent to the distance-least-squares fit). All 5 math unit tests pass
([tests.py](tests.py)): SE(3)+reflection invariance to 1e-15, exact 25°→25.000° hinge recovery, pure-deformation
→ θ≈0.

## Headline

**CDR-loop motion is hinge-dominated: the loops swing rigidly about their framework anchor while their internal
shape barely changes.** Pooled over 21 TCRs (1KGC excluded as the 10× breadth outlier):

- **`D_deform` = 0.32 Å vs `D_framework` = 0.55 Å** — framework-relative displacement is ~1.7× the internal
  deformation; **every** per-TCR point sits below the `y=x` line (figs 1–2).
- **Hinge fraction `1 − E_nonrigid²/total²` = 0.81** (√ = 0.90 as a magnitude ratio → consistent with the prior
  Kabsch/RMSF estimate of **~92 %** in [`../interface_analysis/MD_flexibility/`](../interface_analysis/MD_flexibility/RESULTS.md),
  reproduced here from **independent, alignment-free** math).
- **Hinge angle θ ≈ 9–12° typical, p95 ≈ 18–25°** about the central conformation, at **~constant low deformation**
  — the PMF basin is elongated along θ at fixed `D_deform ≈ 0.2–0.4 Å` (fig 4).
- **Total flexibility splits (Å, additive: `total² = hinge² + deform²`) into 77–91 % hinge, 9–23 % deformation**
  per CDR (fig9); CDR3 loops carry the most deformation (A_CDR3 19 %, B_CDR3 23 %), CDR1/2 the least (9–19 %).

## Per-CDR (mean over 21 TCRs, 1KGC excluded)

| CDR | D_deform Å | D_framework Å | θ° (p95) | δθ° | hinge_frac | H |
|---|---|---|---|---|---|---|
| A_CDR1 | 0.29 | 0.54 | 11.5 (23.8) | 9.1 | 0.83 | 0.67 |
| A_CDR2 | 0.30 | 0.50 | 9.0 (17.8) | 7.5 | 0.78 | 0.64 |
| A_CDR3 | 0.40 | 0.60 | 11.0 (22.5) | 9.2 | 0.76 | 0.62 |
| B_CDR1 | 0.19 | 0.47 | 12.5 (24.9) | 7.8 | 0.87 | 0.72 |
| B_CDR2 | 0.24 | 0.53 | 12.3 (23.6) | 6.8 | 0.87 | 0.70 |
| B_CDR3 | 0.48 | 0.65 | 11.2 (22.4) | 9.9 | 0.74 | 0.59 |

- **CDR3 loops are the least hinge-like** (hinge_frac 0.74–0.76) — they carry the most internal deformation
  (`D_deform` 0.40–0.48 Å, the largest), so their hinge angle is the least well-defined (**δθ ≈ θ**: the honest
  error bar flags that "how much did it rotate" is intrinsically fuzzy for the floppy apex loops). CDR1/2 are
  crisp hinges (hinge_frac 0.78–0.87, δθ < θ).
- `D_deform` per CDR (0.19–0.48 Å) **reproduces MD_flexibility's 0.35–0.73 Å band** from the orthogonal
  distance-only route — a genuine cross-validation of the earlier Kabsch result.

## What this adds over the prior Kabsch/RMSF study

1. **Independent confirmation** of "loops hinge, don't deform" without any global framework alignment — so the
   result is not an artifact of the α+β superposition that mixes in the Vα–Vβ inter-domain wobble.
2. **A real hinge angle** `θ` (scale-free, comparable across loop sizes) + screw axis, where the old
   `√(total²−deform²)` gave only a magnitude.
3. **A per-frame honesty meter** `δθ ≈ E_nonrigid/ℓ` that says exactly when the hinge is well-defined (rigid
   loops) vs. not (CDR3 apex) — the coupling made quantitative rather than assumed away.

## Caveats
- **Unbound** TCRs (no pMHC) → this is intrinsic CDR dynamics, not binding-induced.
- **1KGC** excluded from pooled means (a large conformational transition inflates its magnitudes ~10×); it is the
  visible low-hinge_frac outlier in fig 2 and is kept in `results/`.
- Magnitudes are ensemble-sampling-dependent; the **relative** result (hinge ≫ deform, CDR3 the softest) is the
  transferable finding.
- `hinge_frac` here is a variance-explained fraction (R²-like); MD_flexibility's "92 %" is a magnitude ratio
  (√hinge_frac ≈ 0.90) — the two conventions agree once matched.
- Anchoring is **per chain to its own framework core** (pure intra-domain hinge). Re-anchoring β-loops to the α
  core to expose the Vα–Vβ inter-domain contribution (plan fig f5) is a cheap follow-up not yet run.

## Do TCRs share one profile, or differ? (fig5)

**Mixed — and the TCR-specific part dominates.** Two-way variance partition of `hinge_frac` (hinge-vs-deform
balance) across the 22 TCRs × 6 CDRs:

| explained by | hinge_frac | hinge angle θ | D_deform |
|---|---|---|---|
| **CDR identity** (universal loop pattern) | 34 % | 3 % | 29 % |
| **TCR identity** (some TCRs floppier overall) | 12 % | 30 % | 13 % |
| **TCR×CDR interaction + noise** | **54 %** | **66 %** | 59 % |

- There **is** a shared skeleton — CDR3 loops are the softest (least hinge-like) in nearly every TCR, CDR1/2 the
  crispest — that's the 34 % "CDR identity" term. But the **interaction term dominates (54 %)**: which loops hinge
  vs deform, and by how much, is **TCR-specific**, not one universal profile (fig5 B: the deviation heatmap is
  patchy, e.g. 3SKN's A_CDR3 and 4UDT's B_CDR3 are outliers in opposite directions).
- **CDR3 is where TCRs differ most** (hinge_frac SD ≈ 0.10, range 0.51–0.93); CDR1/2 are conserved (SD ≈ 0.04).
- The **hinge angle θ itself is highly TCR-specific** (TCR 30 %, CDR only 3 %) — the *amount* a loop swings is an
  individual-TCR property, not set by which CDR it is.

## Does length predict the hinge? Can we predict it from the TCR? (fig6)

**Loop length drives *deformation*, not *hinge angle*.** Correlations (n=126 loops, 1KGC excluded; within-CDR =
partialling out CDR identity):

| length vs | pooled r | within-CDR r | reading |
|---|---|---|---|
| hinge **angle θ** | **−0.00** | 0.09 | longer loops do **not** swing through bigger angles |
| hinge **displacement** `D_framework` | 0.40 | 0.26 | bigger Å at the *same* angle — pure lever-arm |
| **deformation** `D_deform` | **0.65** | 0.50 | longer loops deform much more |
| hinge fraction | −0.59 | −0.41 | longer ⇒ **less** pure hinge (because more deform) |

This **sharpens** MD_flexibility's "longer loops flex more (r=+0.62)": the extra flexibility of long loops is
**internal deformation, not extra hinging** — the hinge angle is length-invariant (flat fit, fig6 left).

**Predictability (OLS R²):**

| target | ~ length | ~ length + CDR |
|---|---|---|
| hinge **angle θ** | 0.00 | 0.05 |
| hinge displacement / total motion | 0.16–0.17 | 0.19–0.21 |
| **deformation** `D_deform` | 0.42 | 0.48 |
| hinge fraction (balance) | 0.35 | 0.46 |

So from static features you **can** predict how *deformable* a loop is and its hinge/deform *balance* (R² ≈ 0.4–0.5
from length alone — longer = floppier), but you **cannot** predict the *hinge angle / magnitude* (R² ≈ 0): that is
TCR-idiosyncratic (30 % of its variance is TCR identity) and ensemble-breadth-driven, needing richer features
(sequence composition, germline, anchor geometry) — and even then partly reflects sampling/dynamics, not structure.

**Per-CDR predictability (fig7).** Within each CDR (identity fixed → length is the only varying static feature),
`D_deform` is well predicted by length wherever length actually varies — **A_CDR3 R²=0.51, B_CDR3 R²=0.47**
(B_CDR1 R²=0.57 but on only two length values) — while the **hinge angle θ is unpredictable in every CDR**
(R² ≤ 0.17, mostly < 0.05). Pooled: deformation R²=0.48 (CDR identity 0.31 + length 0.42), hinge angle R²=0.05.
The signal that exists is entirely a *deformation* signal.

## How well can we predict an unseen TCR? (fig8) — leave-one-TCR-out

**Question (the one we actually care about):** using the other 21 TCRs, how accurately can we predict a **held-out
TCR's** total flexibility / deformation / hinge angle, for each of the 6 CDRs? The CDR **average is itself a valid
prediction**; loop **length is an optional feature**, used only where it lowers the held-out error.

**Design:** leave-one-TCR-out (the TCR is the independent replicate, *never* the frame). Two predictors per CDR:
M0 = mean of the training TCRs; M1 = linear fit in length. **Headline metric = held-out RMSE in physical units**
(Å or deg) and **relative error = RMSE / mean** (comparable across Å and °). No nested CV (plain OLS).

**Answer — all three are predictable to ~30–45 % relative error just from the other TCRs:**

| metric | typical held-out error | most predictable | least |
|---|---|---|---|
| total flexibility | ~30–40 % (0.3–0.6 Å) | B_CDR3 33 %, A_CDR1 29 % | B_CDR2 68 % |
| deformation | ~34–49 % (0.04–0.16 Å) | B_CDR1 23 %, CDR3 34 % | B_CDR2 58 % |
| hinge angle | ~35–44 % (3.4–5.4°) | B_CDR3 35 % | B_CDR2 81 % |

- **Length helps *deformation* the most** (fig8 A: the gap between the dashed average-only and solid with-length
  bars is large for deformation at A_CDR3, B_CDR1, B_CDR3) — the only quantity where knowing the length materially
  beats the CDR average. For **total and hinge, length adds ~nothing** (solid ≈ dashed); the population average is
  already the best simple predictor.
- **The hinge angle IS predictable (~35–44 %) — just not *from length*.** This reconciles "hinge doesn't
  generalize from length" (length ≯ the average) with "you can still predict it ~40 % from the population average."
- **B_CDR2's apparent unpredictability is two outlier TCRs, not intrinsic flexibility (fig11).** The *typical*
  B_CDR2 is among the **least** flexible loops (median total ≈ 0.85 Å, θ ≈ 7–9°), but **3SKN (θ≈45°, sustained
  93 % of frames) and 4JFH (θ≈35°)** undergo a genuine large B_CDR2 swing — a coherent, monotonic per-residue RMSF
  gradient (0.6→4.0 Å toward res 63–65), correct IMGT numbering, healthy β-core reference → **real motion, not a
  pipeline artifact**. Mechanism (anchoring is per-chain — β-CDR2 vs the β core, *not* the Vα–Vβ joint): in 3SKN
  the β-framework that should anchor CDR2's **C-terminal end (FR3, res 66+) is itself not rigid** (fails the
  rigid-core test; 3SKN's β core is only 36/78 CA, and has anchors at FR2 res 51–55 but **none at 66–72**, vs both
  sides in normal TCRs). So CDR2 is pinned on **one side only** and res 63–65 drift with the loose FR3 — a genuine
  but one-sidedly-anchored motion, and a method caveat for the rare loop whose own framework anchor is partly mobile.
  RMSE is dominated by these tails, so under the **robust median held-out error** B_CDR2 drops to ~0.33 Å (hinge) /
  0.34 Å (total) — normal. Under the robust metric the hardest-to-predict loops are the **CDR3s** (median ≈ 0.40 Å),
  as expected. Report the **median** for typical-loop predictability; RMSE only where the tails matter.
- **Where the prediction error lives, and whether length helps (fig10):** splitting total into its parts
  (`c_deform + c_hinge = total`, Å) and predicting each with/without length: the **deformation part is small *and*
  well-predicted** (held-out RMSE 0.06–0.24 Å) and **length lowers it further for CDR3** (A_CDR3 0.18→0.13,
  B_CDR3 0.24→0.20 Å); the **hinge part is large *and* poorly-predicted** (0.27–0.78 Å) and **length does not help
  it** (grey ≈ colored). Because total ≈ hinge, the hinge **dominates both the motion (fig9) and the
  unpredictability** — knowing the length only sharpens the minor deformation part.
- **Ranking power / AUC (fig12).** "AUC" here = concordance index (rank-AUC generalized to a continuous outcome;
  0.5 = random, rank-based ⇒ outlier-robust). Pooled over all 126 loops, ranking a loop's flexibility from **CDR
  identity → +length**: **deformation 0.64 → 0.71** (rankable), **hinge 0.49 → 0.56** (CDR identity is *chance* for
  the hinge), total 0.57 → 0.61. Within-CDR, length ranks only **CDR3 deformation** (A_CDR3 0.60, B_CDR3 0.71);
  the other loops' length barely varies so their AUC is at chance. Same verdict as the error analysis, rank-robust:
  deformation is discriminable from static features, the hinge is not.
- Caveat: generalization is tested across 21 structurally-related TCRs; a truly divergent TCR may fall outside.

**On the hinge metric (θ vs RMSD):** `θ` (angle) is the scale-free "how much did it rotate"; `hinge_rmsd` =
`√(total²−E_nonrigid²)`, `total_disp`, `D_framework` are the Å "how far did it move" versions (what "hinge by RMSD"
usually means). `θ` is the fairer test of "do longer loops hinge more" (the RMSD versions are inflated by
lever-arm), but the predictability conclusion holds under all of them. Per-CDR numbers +
Q²-vs-mean in `predictability_unseen_tcr.csv`.

## Files
`results/<ID>_perframe.npz` (per-CDR per-frame `D_deform`,`D_deform_sep`,`D_framework`,`H`,`θ`,`δθ`,`E_nonrigid`,
`total_disp`,`hinge_frac`, screw `axis`), `<ID>_core.json` (rigid-core atoms, RMSF), `<ID>_percdr.csv`;
aggregated `master_percdr.csv`, `summary_percdr.csv`, `length_correlations.csv`. Figures: `fig1_2Dmap` ·
`fig2_hinge_fraction` · `fig3_theta` · `fig4_pmf` · `fig5_tcr_profiles` (TCR-specificity + variance partition) ·
`fig6_length_vs_hinge` (length → deformation, not angle) · `fig7_predictability` (per-CDR: deformation predictable,
hinge angle not) · `fig8_cv_generalizability` (leave-one-TCR-out: held-out error per CDR for total/deform/hinge, native units) ·
`fig9_flexibility_decomposition` (total = hinge + deform, stacked in Å + normalized %) ·
`fig10_predicted_decomposition` (predicting each part of an unseen TCR's total; error lives in the hinge) ·
`fig11_robust_predictability` (RMSE vs median held-out error: B_CDR2's big RMSE is 2 tails) ·
`fig12_auc` (concordance/AUC of ranking flexibility from length).
Aggregates: `predictability_per_cdr.csv`, `length_correlations.csv`, `predictability_unseen_tcr.csv`,
`predicted_decomposition.csv`, `robust_predictability.csv`, `auc_concordance.csv`. Scripts: `plot_tcr_profiles.py`,
`plot_length_prediction.py`, `plot_predictability.py`, `cv_generalizability.py`, `plot_decomposition.py`,
`plot_predicted_decomposition.py`, `plot_robust_predictability.py`, `plot_auc.py`.
