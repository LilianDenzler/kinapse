# elasticity

A **fresh, separate** elasticity-theory treatment of CDR-loop motion (not built on `../`, `../graph_method/`, or
`../elastic_rod/`). Three complementary formalisms, all **alignment-free from CA–CA distances**:

1. **Continuum strain field** — per-residue finite (Green–Lagrange) strain from local CA–CA distance changes;
   volumetric + shear + non-affine (`D²min`). *Where/how the loop deforms; the non-affine hotspot = the hinge.*
2. **Elastic network (GNM/ANM)** — springs from the CA contact map; elastic normal modes vs the MD fluctuations.
3. **Data-driven stiffness** — invert the CA–CA distance covariance to an effective stiffness network; compliance,
   soft modes, probe-force response, quasi-harmonic entropy.

**Rigid motion is never fit** — it is the **zero-strain null space** of all three (`E_i=0` / the zero modes /
zero variance), so hinge and simultaneous deformation separate with no superposition and no gauge choice. An optional
hinge *angle* uses the Eckart best-fit frame only. Full method + the rigid-motion argument: **[`plan.md`](plan.md)**.

```bash
mamba activate kinapse
python tests.py                 # 11-check gate (all pass): invariance, rigid=>0 strain, affine/non-affine, ENM, stiffness
python preflight.py             # IMGT insertion-code guard: mdtraj CA == Bio.PDB == seq for every CDR (all 22 OK)
python compute_elasticity.py    # per system/CDR -> results/<ID>_elasticity.npz + _percdr.csv (resumable)
python plot_elasticity.py       # aggregate -> figures/ + summary_percdr.csv
```

- [`elast.py`](elast.py) — the three formalisms' primitives (strain field, GNM/ANM, covariance stiffness), all
  rigid-invariant. [`config.py`](config.py) — paths, systems, cutoffs, ridge. [`tests.py`](tests.py) — the gate.
  [`preflight.py`](preflight.py) — insertion-code integrity guard (wired into compute as `assert_no_dropped_residues`).

**Status: done (22 TCRs).** Headline — three independent elasticity formalisms converge: **CDR3 deformation is
non-affine, distributed, and entropy-rich (uniform-spring GNM corr ≈ 0), while CDR1/2 are affine, localized, and
GNM-explained**; non-affine strain and configurational entropy agree at r=0.91. Findings + figures: **[`RESULTS.md`](RESULTS.md)**.
Outputs land in git-ignored `results/`.
