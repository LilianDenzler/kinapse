# Scorer benchmarking

Benchmark every interface scorer on the TCR3d tfold / OpenMM-minimised dataset and,
per scorer, measure:

1. **Agreement** — do the *modelled* complexes score like their *ground truth*?
   (Pearson/Spearman of GT-value vs model-value across complexes.)
2. **Discrimination** — do modelled *real* complexes score differently from modelled
   *negatives*? (AUROC / AUPRC / Cohen's d on a leakage-aware test split.)
3. **Structural agreement** — how close is each model to its GT? Cα iRMSD over the 6
   CDR loops (kinapse loader identifies the CDRs) → **HQ/MQ/AQ/LQ** tiers + per-CDR RMSDs
   (`structural.csv`, `tiers.csv`). Scorer-independent. Disable with `--no-structural`.

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
ifscore install all   # provision every scorer (geometry_scoring needs nothing)
```

## Run
```bash
# quick check (no external scorers needed):
python run_benchmark.py --limit 12 --scorers geometry_scoring --plots

# full run (resumable via chains cache; -j parallelises chain numbering AND scoring):
ANARCI_CPU=1 python run_benchmark.py --scorers all -j 16 --plots
```
`-j` now parallelises chain resolution too (numbering is the bottleneck) — the full
TCR3d set goes from ~90 min serial to a few minutes.

### Use STCRpy instead of the native loader
Identify chains + annotate CDRs with OPIG **STCRpy** (external env) rather than kinapse's
loader — everything downstream (scoring, agreement, tiers) is identical:
```bash
export KINAPSE_STCRPY_PYTHON=/path/to/stcrpy-env/bin/python   # its own heavy env
python run_benchmark.py --scorers all -j 16 --loader stcrpy --plots
```

Outputs land in `results/`: `scores.csv`, `agreement.csv`, `discrimination.csv`,
`structural.csv`, `tiers.csv`, `chains_cache.json`, and `plots/` (if `--plots`).
