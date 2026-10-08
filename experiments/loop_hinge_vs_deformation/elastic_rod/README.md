# elastic_rod

CDR loop as a **discrete elastic rod** (Cα backbone = centreline). Alignment-free decomposition of each frame into
**rigid reorientation** (zero internal strain) + **bending** (`Δκ`) + **torsional deformation** (`Δτ`) +
**stretch** (`Δε`) + coupling residual, all from Cα geometry — plus an effective-stiffness / compliance model
(`K_eff = k_BT Σ⁻¹`), persistence length, and quasi-harmonic entropy.

Method sub-study of the parent [`../`](..): it resolves the parent's single `D_deform` into mechanical strain
classes, and **imports** [`../graph_method/`](../graph_method) (`rotations.py`, `hinge_validity.py`, `graph_fit.py`)
for the rigid/hinge half rather than re-deriving it. Full method, physics, validation gate, and data:
**[`plan.md`](plan.md)**.

```bash
mamba activate kinapse
python tests.py                 # invariance + completeness + synthetic-mode + stiffness-recovery gate
python compute_elastic_rod.py [--systems 3QH3 8YJ3] [--force]
python plot_elastic_rod.py      # aggregate + figures/ + RESULTS.md
```

Status: **plan only** — primitives, tests, and MD pipeline to be built next. Outputs land in git-ignored `results/`.
