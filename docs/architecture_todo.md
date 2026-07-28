# Architecture

`kinapse` is five sub-packages layered on a shared config/data foundation. Lower
layers never import higher ones, so each is usable on its own.

```
                 ┌─────────────────────────────────────────────┐
   pipelines →   │  benchmark · best_method · analyse_md        │  end-to-end workflows
                 └─────────────────────────────────────────────┘
   ⑤ generation  DiG runner + postprocess  ─────────────┐
   ④ analysis    align · features · reduce · metrics · pmf · plots
   ③ geometry    α/β docking angles (per-structure & per-frame)
   ② embedding   fasta · MSA · Evoformer
   ① structures  TCR / TCR-pMHC loader, IMGT numbering, pairing, linkers   ← the basis
                 ────────────────────────────────────────────────
   foundation    config (paths)   ·   data (consensus refs, SO(3) tables)   ·   regions (IMGT)
```

## The five modules

### ① `structures` — the basis
Load a PDB, fix duplicate residue numbers, IMGT-renumber (ANARCI/ANARCII), pair
α/β (or γ/δ) chains by interface contacts (Hungarian assignment), slice to
CDR/framework regions, and optionally attach an MD trajectory.
- `tcr.py` — `TCR`, `TCRPairView`, `TrajectoryView`
- `numbering/` — ANARCI wrappers + `pair_tcrs_by_interface`
- `io.py`, `ops.py`, `select.py` — PDB IO, subset ops, region→atom predicates
- `linkers/` — scFv α–β linker building (MODELLER)
- `pmhc.py` — **scaffold** for pMHC / TCR-pMHC complexes (API-ready)

### ② `embedding` — sequence embedders
`fasta.py` (PDB→FASTA), `msa.py` (MMseqs2-GPU), `evoformer.py` (OpenFold).

### ③ `geometry` — α/β docking geometry
The ABangle-style 6 parameters (BA torsion, four bend angles, centroid distance)
describing how the two variable domains sit relative to each other — per
structure (`calc_geometry.py`) or per MD frame (`calc_geometry_MD.py`), plus the
inverse rebuild (`change_geometry.py`). Reference consensus data ships in
`geometry/data/`.

### ④ `analysis` — structure & MD analysis + metrics
The scientific core:
- `aligning.py` — Kabsch / TMalign / ProFit superposition (per region)
- `rmsd_tm.py` — per-frame RMSD & TM-score
- `embeddings/features.py` — Cα-distance / coordinate / dihedral features
- `embeddings/dim_reduction.py` — PCA, weighted PCA, kernel PCA, TICA, diffusion maps
- `embeddings/metrics.py` — trustworthiness, Mantel test
- `pmf_kde.py` — free-energy surfaces (PMF) + **Jensen–Shannon divergence** (headline metric)
- `embeddings/run_*.py` — per-region end-to-end runners
- `plotters.py`, `PCA_methods.py` (legacy projection), `embed_assesment.py` (legacy)

### ⑤ `generation` — conformer sampling
`dig_runner.py` (drive the external DiG diffusion sampler), `postprocess.py`
(fold generated PDBs into an `.xtc`, stripping the linker), `experiment.py`
(top-level driver).

## Data flow (benchmark)

```
GT PDB + MD .xtc ─┐
                  ├─► structures: renumber + pair + region-slice
model PDBs ─► generation.postprocess ─► .xtc ─┘
                  ▼
   analysis: align → featurize → reduce to 2-D → { trustworthiness/Mantel , PMF/JSD , RMSD/TM }
                  ▼
   pipelines.benchmark: shared global bins → JSD tables + figures
```

## Design choices in the reorg
- **Lazy imports** in every sub-package `__init__` (PEP 562) — `import kinapse`
  never fails because an optional dep (torch, pymol, deeptime, modeller) is absent.
- **Config over hardcoding** — `kinapse.config` resolves paths from env vars / YAML.
- **`src/` layout** + `pyproject.toml` with optional extras for clean, shareable installs.
- Science code was migrated **byte-for-byte**; only import wiring changed.
