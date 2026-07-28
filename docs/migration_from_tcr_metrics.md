# Migration map: `TCR_Metrics` → `kinapse`

The original code was copied **byte-for-byte** into the new layout; only import
wiring changed (`TCR_TOOLS.* → kinapse.*`, broken bare imports fixed, hardcoded
`sys.path` hacks removed, `path_config → kinapse.config`). The original
`TCR_Metrics/` folder is left untouched.

## Module map

| Old (`TCR_Metrics/`) | New (`src/kinapse/`) |
|---|---|
| `TCR_TOOLS/__init__.py` (IMGT ranges) | `regions.py` |
| `TCR_TOOLS/classes/tcr.py` | `structures/tcr.py` |
| `TCR_TOOLS/core/{io,ops,select}.py` | `structures/{io,ops,select}.py` |
| `TCR_TOOLS/numbering/*` | `structures/numbering/*` |
| `TCR_TOOLS/linkers/*` | `structures/linkers/*` |
| *(new)* | `structures/pmhc.py` (scaffold) |
| `TCR_TOOLS/embed_sequence/MMseq2_GPU_MSA_MAKER.py` | `embedding/msa.py` |
| `TCR_TOOLS/embed_sequence/openfold_wrapper_for_evoformer.py` | `embedding/evoformer.py` |
| `TCR_TOOLS/embed_sequence/pdb2fasta.py` | `embedding/fasta.py` |
| `TCR_TOOLS/geometry/*` (+ `data/`) | `geometry/*` (+ `data/`) |
| `TCR_TOOLS/aligners/aligning.py` | `analysis/aligning.py` |
| `TCR_TOOLS/scoring/rmsd_tm.py` | `analysis/rmsd_tm.py` |
| `TCR_TOOLS/scoring/pmf_kde.py` | `analysis/pmf_kde.py` |
| `TCR_TOOLS/scoring/plotters.py` | `analysis/plotters.py` |
| `TCR_TOOLS/scoring/PCA_methods.py` (legacy) | `analysis/PCA_methods.py` |
| `TCR_TOOLS/scoring/embed_assesment.py` (legacy) | `analysis/embed_assesment.py` |
| `TCR_TOOLS/scoring/embeddings/*` | `analysis/embeddings/*` |
| `dig_runner/pdb_to_inference_dig_all_variants.py` | `generation/dig_runner.py` |
| `dig_runner/full_experiment_pipeline.py` | `generation/experiment.py` |
| `dig_output_handler/process_output.py` | `generation/postprocess.py` |
| `pipelines/analyse_MD/gt_flexibility_analysis.py` | `pipelines/analyse_md/gt_flexibility_analysis.py` |
| `pipelines/calc_metrics_model_outputs/*` | `pipelines/benchmark/*` |
| `pipelines/metrics_for_outputs/*` | `pipelines/best_method/*` |
| `.so3_*.npy` (repo root) | `data/so3/so3_*.npy` |
| `path_config.py` (parent project) | `config.py` (ported + `KINAPSE_*` env support) |

## What changed beyond moving files
- **Imports** rewritten to the `kinapse.*` namespace.
- **Broken imports fixed**: `from pdb2fasta import …`, `from MMseq2_GPU_MSA_MAKER import …`
  (bare) and `from tcrgeometry.numbering import process_pdb` (a package that never
  existed) now resolve correctly.
- **`sys.path.insert(project_root)` hacks removed** (the package is importable);
  the one that points at an external OpenFold checkout is kept.
- **Config**: `from path_config import get_paths` → `from kinapse.config import get_paths`.

## Robustness fixes made during the reorg (no behaviour change on the happy path)
- **Optional heavy imports are now lazy/guarded** so the whole package imports
  from a plain `pip install` (validated in a clean venv), instead of hard-failing
  at import: legacy `anarci` and `pymol2` in `structures/numbering`, and the
  unused `pyemma` in `analysis/embeddings/dim_reduction` (TICA runs on `deeptime`).
  Each now raises a clear, actionable error only if you actually invoke the path
  that needs it.
- **Fixed a latent `NameError`**: `geometry/calc_geometry.py` called
  `write_renumbered_fv(...)` without importing it (broken in the original too);
  it now imports the function from `geometry/get_anchor_coords.py`.

## Known pre-existing issues (preserved, not introduced)
- `pipelines/benchmark/sample_MD_for_baselines.py` imports a module `run_calc`
  that does not exist in the source tree (it was already broken; left as-is so
  behaviour is unchanged — fix by pointing it at `run_calc_scaled`).
- `pipelines/**` `__main__` blocks still contain the original absolute example
  data paths. Prefer `kinapse.config` / env vars; see `docs/pipelines.md`.
- Two legacy modules (`analysis/PCA_methods.py`, `analysis/embed_assesment.py`)
  are superseded by `analysis/embeddings/*` but kept because `pca_project_two`
  is still imported by the pipelines.
