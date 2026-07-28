# Pipelines

The migrated end-to-end workflows live under `kinapse.pipelines`. They preserve
the original `TCR_Metrics` logic exactly. List them any time with:

```bash
kinapse pipelines
```

## Available workflows

### `kinapse.pipelines.benchmark` — model-vs-MD benchmark
The two-phase driver:
1. **collect** — for each TCR × model × region × (feature, reducer) mode: align,
   embed, RMSD-calibrate/center, accumulate shared histogram bin ranges.
2. **PMF/JSD** — build free-energy surfaces on the shared bins, compute
   Jensen–Shannon divergence, write summary tables + heatmaps.

Key modules:
- `benchmarks_calc_scaled.py` — sequential driver (`python -m kinapse.pipelines.benchmark.benchmarks_calc_scaled`)
- `benchmarks_calc_scaled_parallel.py` — process-pool parallel driver
- `run_calc_scaled.py` — the collect/PMF/table library functions
- `compare_MD_adaptive_sampling.py`, `sample_MD_for_baselines.py` — MD baselines
- `Standardise_by_RMSD.py` — RMSD-calibrated (Å-scaled) embeddings
- `benchmark_table_maker.py` — JSD tables + heatmaps
- `pymol_alignment_videomaker.py`, `visualise_*_pymol.py` — PyMOL rendering
- `config_assess_modes*.yaml` — region → (feature, reducer) maps

### `kinapse.pipelines.best_method` — best-mode selection
Screen every reducer per region and select the best (feature × reducer) mode by
trustworthiness + Mantel r.
- `run_test.py`, `asses_model_output.py`, `analyse_metrics.py`
- `global_analysis_best_metric.py`, `global_method_selection.py`
- `best_methods_config.yml` — the selected mode per region-pair (generated output)

### `kinapse.pipelines.analyse_md` — ground-truth flexibility
`gt_flexibility_analysis.py` — per-region RMSD-to-average, representative-frame
selection, and cross-TCR ridgeline KDE overlays.

## Running them

```bash
python -m kinapse.pipelines.benchmark.benchmarks_calc_scaled
```

> **Heads-up:** the `__main__` blocks in these scripts still contain the original
> absolute example data paths (e.g. `/mnt/larry/...`, `/mnt/dave/...`). Two ways
> to point them at your data:
>
> 1. **Config / env vars (preferred):** set `KINAPSE_OUTPUT`, `KINAPSE_DATASETS`
>    or drop a `kinapse.yaml` (see `config.example.yaml`), and read them via
>    `kinapse.config.get_paths()`.
> 2. **Edit the `__main__` block** of the specific driver.
>
> The reusable functions (`run_all_TCRs_collect`, `run_all_TCRs_pmf`,
> `run_make_tables`, `assess_screening`, …) take explicit path arguments, so you
> can also import and call them directly from your own script/notebook.

## Config maps (region → method)

`config_assess_modes.yaml` and friends map each region (e.g. `A_CDR3`) and
alignment context to a `feature_reducer` string like `ca_pca`, `ca_kpca_cosine`,
`coords_tica`, `dihed_pca`. Pass the one you want to the benchmark driver.
