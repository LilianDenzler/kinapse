# loop_hinge_vs_deformation

**Question:** for each MD ensemble, how much of every CDR-loop conformation is **rigid re-orientation
(hinge)** vs. **internal shape change (deformation)** — measured *without any coordinate superposition*, from
SE(3)-invariant **distance matrices** anchored to a rigid Vα-framework core?

This is the alignment-free, distance-only reformulation (and hinge-*angle* upgrade) of the Kabsch/RMSF
prototype in [`../interface_analysis/MD_flexibility/`](../interface_analysis/MD_flexibility/), which found
~92 % hinge. Full method, data, API, and validation plan: **[`plan.md`](plan.md)**.

**Status: done (22 TCRs).** Headline — CDR loops are hinge-dominated (`D_framework` 0.55 Å ≫ `D_deform` 0.32 Å;
hinge fraction 0.81, √≈0.90 ↔ the prior 92 %), with a real hinge angle θ≈9–12° and a δθ error bar that flags the
floppy CDR3 apex. Findings + figures: **[`RESULTS.md`](RESULTS.md)**.

## Run
```bash
mamba activate kinapse        # env at ~/.local/share/mamba/envs/kinapse
python tests.py               # 5 math validation tests
python compute_hinge_deform.py [--systems 3QH3 8YJ3] [--force]   # per-system, resumable, ~40 s each
python plot_hinge_deform.py   # aggregate + figures/ + summary
```

- [`compute_hinge_deform.py`](compute_hinge_deform.py) — load MD → rigid α/β core → `D_deform`/`D_framework`/`θ`/
  `δθ`/`E_nonrigid` per CDR (all distance-based; one Procrustes only for θ). [`config.py`](config.py) holds paths,
  the system list, per-system chain overrides (8YJ3), and the transferable thresholds.
- Outputs land in `results/` (git-ignored); `figures/` holds the four PNGs.
