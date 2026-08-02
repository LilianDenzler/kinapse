# Scorer benchmark (`kinapse.benchmarks`)

Benchmark every interface scorer on three sets of TCR-pMHC complexes and ask, per
scorer: (1) do **modelled** complexes score like their **ground truth**, and
(2) do modelled **real** complexes score differently from modelled **negatives**?
Plus a structure-only check: (3) **how close is each model to its GT** (Cα iRMSD over
the 6 CDRs → HQ/MQ/AQ/LQ tiers), independent of any scorer.

## Inputs
Three directories of complex PDBs:
- **`gt_dir`** — ground-truth (real) complexes (e.g. crystal structures, minimised).
- **`model_dir`** — modelled versions; **filenames match `gt_dir`** (stem = complex id).
- **`neg_dir`** — negative (artificial, non-existent) modelled complexes.

Each complex is loaded to derive correct chains — receptor = TCR α/β, ligand = pMHC
(everything else) — via `kinapse.scoring.infer_tcr_pmhc_chains` (ANARCII). Results are
cached (`chains_cache.json`); a structure that can't be numbered is skipped. Numbering is
the bottleneck, so it runs in a process pool (`-j` / `n_jobs`) — for the full TCR3d set
this turns a ~90-min serial pass into a few minutes.

### Loader backend (`--loader`)
The loader that does chain identification **and** CDR annotation is pluggable:
- `native` (default) — kinapse's own loader (ANARCII/ANARCI, no conda needed).
- `stcrpy` — OPIG **STCRpy** run as a fully external model (chains via `get_VA`/`get_VB`
  + antigen/MHC; CDRs via STCRpy's IMGT fragments). STCRpy is heavy (ANARCI models, PLIP,
  OpenBabel) and is **never imported into kinapse** — point kinapse at its environment with
  `export KINAPSE_STCRPY_PYTHON=/path/to/stcrpy-env/bin/python`. Both loaders emit the same
  region keys, so agreement/discrimination/tiers are computed identically either way.

## What it computes
Scoring is delegated to **ifscore** (`kinapse.scoring`). Reference-based scorers
(e.g. DockQ) run model-vs-GT; reference-free scorers (`geometry_scoring`, `prodigy`,
energy) drive discrimination. A metric's role is inferred from the data (reference-free
= has values on the negatives). Unprovisioned scorers yield NaN and are skipped.

A **leakage-aware split** groups by complex id (a complex's GT + model never straddle
train/test; stratified by label). Then, per metric:
- **Agreement** — Pearson/Spearman of GT-value vs model-value across complexes, + mean |Δ|.
- **Discrimination** — AUROC / AUPRC / Cohen's d separating modelled positives from
  negatives on the **test** split (threshold fit on train → test accuracy).

## Structural agreement (model vs GT)
Independent of the scorers, each modelled positive is compared to its GT with the
**same Kabsch region-superposition used in the ensemble analysis** (`kinapse.benchmarks.structural`):
the loader (native or `stcrpy`, see above) IMGT-numbers both structures and identifies the
CDR/FR regions, then per residue (matched by IMGT number) it computes Cα RMSDs —

- per-CDR RMSD after aligning on that chain's **framework** (α-fwk → α-CDRs, β-fwk → β-CDRs),
- per-CDR **local** RMSD (loop superposed on itself — conformation only),
- **framework RMSD**, and
- **Cα iRMSD over the 6 CDR loops** (aligned on the whole framework).

`assign_tier` maps (Cα iRMSD, DockQ) → **HQ/MQ/AQ/LQ** (HQ ≤2 Å & DockQ ≥0.8; MQ ≤5 Å &
≥0.49; AQ <5 Å & ≥0.23; else LQ; DockQ optional — omitted when the `dockq` scorer isn't run).
This adds `struct__*` columns to the modelled-positive rows and a `tier` label; the HQ set
is the subset of high-confidence model/GT pairs. Skip with `structural=False` / `--no-structural`.

Outputs to `out_dir`: `scores.csv`, `agreement.csv`, `discrimination.csv`, `structural.csv`,
`tiers.csv`, `chains_cache.json`.

## Run
```bash
python -m kinapse.benchmarks.scorer_benchmark \
    --gt    /path/TCR_complexes_openmm_minimised \
    --model /path/TCR_complexes_tfold_openmm_minimised \
    --neg   /path/negative_TCR_complexes_tfold/openmm_minimised \
    --out   scorer_benchmark_out --scorers all -j 16
# quick check: add  --limit 12 --scorers geometry_scoring   (geometry needs no provisioning)
# use STCRpy for chains + CDRs instead of the native loader:
export KINAPSE_STCRPY_PYTHON=/path/to/stcrpy-env/bin/python
python -m kinapse.benchmarks.scorer_benchmark ... --loader stcrpy
```
`-j` parallelises both chain resolution and scoring. Needs
`pip install "kinapse[bench,structures]"` + `ifscore` (and `ifscore install all` for the
non-geometry scorers). Or from Python:
```python
from kinapse.benchmarks import run_scorer_benchmark
res = run_scorer_benchmark(gt_dir, model_dir, neg_dir, out_dir="out", scorers="all",
                           n_jobs=16, loader="native")   # or loader="stcrpy"
res["agreement"]; res["discrimination"]; res["tiers"]   # tiers omitted if structural=False
```
Structural agreement alone, for one pair:
```python
from kinapse.benchmarks import structural_agreement, assign_tier
s = structural_agreement(model_pdb, gt_pdb, loader="native")   # or loader="stcrpy"
assign_tier(s["struct__cdr_irmsd"])                            # "HQ" / "MQ" / "AQ" / "LQ"
```
