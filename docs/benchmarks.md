# Scorer benchmark (`kinapse.benchmarks`)

Benchmark every interface scorer on three sets of TCR-pMHC complexes and ask, per
scorer: (1) do **modelled** complexes score like their **ground truth**, and
(2) do modelled **real** complexes score differently from modelled **negatives**?

## Inputs
Three directories of complex PDBs:
- **`gt_dir`** — ground-truth (real) complexes (e.g. crystal structures, minimised).
- **`model_dir`** — modelled versions; **filenames match `gt_dir`** (stem = complex id).
- **`neg_dir`** — negative (artificial, non-existent) modelled complexes.

Each complex is loaded to derive correct chains — receptor = TCR α/β, ligand = pMHC
(everything else) — via `kinapse.scoring.infer_tcr_pmhc_chains` (ANARCII). Results are
cached (`chains_cache.json`); a structure that can't be numbered is skipped.

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

Outputs to `out_dir`: `scores.csv`, `agreement.csv`, `discrimination.csv`, `chains_cache.json`.

## Run
```bash
python -m kinapse.benchmarks.scorer_benchmark \
    --gt    /path/TCR_complexes_openmm_minimised \
    --model /path/TCR_complexes_tfold_openmm_minimised \
    --neg   /path/negative_TCR_complexes_tfold/openmm_minimised \
    --out   scorer_benchmark_out --scorers all -j 16
# quick check: add  --limit 12 --scorers geometry_scoring   (geometry needs no provisioning)
```
Needs `pip install "kinapse[bench,structures]"` + `ifscore` (and `ifscore install all`
for the non-geometry scorers). Or from Python:
```python
from kinapse.benchmarks import run_scorer_benchmark
res = run_scorer_benchmark(gt_dir, model_dir, neg_dir, out_dir="out", scorers="all")
res["agreement"]; res["discrimination"]
```
