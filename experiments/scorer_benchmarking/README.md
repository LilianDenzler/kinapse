# Scorer benchmarking

Benchmark every interface scorer on the TCR3d tfold / OpenMM-minimised dataset and,
per scorer, measure:

1. **Agreement** — do the *modelled* complexes score like their *ground truth*?
   (Pearson/Spearman of GT-value vs model-value across complexes.)
2. **Discrimination** — do modelled *real* complexes score differently from modelled
   *negatives*? (AUROC / AUPRC / Cohen's d on a leakage-aware test split.)

Chains are loaded per complex (receptor = TCR α/β, ligand = pMHC). Reference-based
scorers (DockQ) run model-vs-GT; reference-free scorers drive the discrimination.
See [`../../docs/benchmarks.md`](../../docs/benchmarks.md) for the method.

## Datasets (defaults, edit in `run_benchmark.py` or pass `--gt/--model/--neg`)
- GT (real, minimised): `.../TCR3d_datasets/TCR_complexes_openmm_minimised`
- modelled positives (tfold, minimised): `.../TCR_complexes_tfold_openmm_minimised`
- negatives (tfold, minimised): `.../negative_TCR_complexes_tfold/openmm_minimised`

## Prerequisites
```bash
pip install -e ".[bench,structures]"          # in the kinapse env
pip install "ifscore @ git+https://github.com/LilianDenzler/scoring_functions"
ifscore install dockq prodigy foldx rosetta   # geometry_scoring needs nothing
```

## Run
```bash
# quick check (no external scorers needed):
python run_benchmark.py --limit 12 --scorers geometry_scoring --plots

# full run (long: ANARCII + scoring over the whole set; resumable via chains cache):
ANARCI_CPU=1 python run_benchmark.py --scorers all -j 16 --plots
```
Outputs land in `results/`: `scores.csv`, `agreement.csv`, `discrimination.csv`,
`chains_cache.json`, and `plots/` (if `--plots`).
