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

Outputs land in `--out` (default `results/`): `scores.csv`, `agreement.csv`,
`discrimination.csv`, `structural.csv`, `tiers.csv`, `chains_cache.json`,
`_ifscore_manifest.csv`, and `plots/` (if `--plots`).

**Resumable.** Chains, scoring, and structural all checkpoint to disk
(`chains_cache.json`, `scores_cache/part_*.csv`, `structural_cache.csv`). If a run dies
(reboot, kill, OOM), just re-run the **same command into the same `--out`** — it skips
everything already done and only finishes the remainder. Use `--score-chunk N` to control
how often scoring checkpoints (default `max(jobs,16)`; smaller = more frequent saves).

**Peek while it runs.** `plot_from_cache.py` analyses whatever is cached *so far* — read-only,
safe to run against a live job (don't re-run the benchmark into the same `--out`, that races):
```bash
python plot_from_cache.py --out results_native      # writes results_native/partial/{agreement,discrimination}.csv + plots
```
Note the manifest scores all positives before the negatives, so **discrimination stays empty
until scoring reaches the negatives** — agreement (GT vs model) shows up as soon as a few
complete pairs are scored.

### Native loader vs STCRpy — run both and compare
Identify chains + annotate CDRs with OPIG **STCRpy** (external env) instead of kinapse's
loader; everything downstream (scoring, agreement, tiers) is computed identically, so the
two runs are directly comparable.

**Use a separate `--out` per run.** Each dir has its own `chains_cache.json`, which is keyed
by file path only — if both runs shared a dir, the second would reuse the first run's chains
(wrong loader) and overwrite its CSVs.

```bash
# 1) native loader
ANARCI_CPU=1 python run_benchmark.py --scorers all -j 16 --plots --out results_native

# 2) STCRpy loader (its own heavy env)
export KINAPSE_STCRPY_PYTHON=/path/to/stcrpy-env/bin/python
ANARCI_CPU=1 python run_benchmark.py --scorers all -j 16 --plots --loader stcrpy --out results_stcrpy

# 3) compare them
python compare_loaders.py results_native results_stcrpy --labels native stcrpy --out comparison_out
```
`compare_loaders.py` diffs **coverage** (structures each loader could resolve, and which
only one handled), **agreement** & **discrimination** per metric (side by side + Δ),
**tiers**, and **per-complex Cα-iRMSD** (correlation, mean |Δ|, and a tier-agreement
crosstab). It prints a summary and writes `comparison_*.csv` into `--out`.
