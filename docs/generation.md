# Conformer generation — DiG (`kinapse.conformer_generation`)

Generate a TCR conformational ensemble with the DiG diffusion sampler. The pipeline, per TCR:

1. **prep** — load & IMGT-number the TCR, link α+β into one chain (GGGGS×3; needs **MODELLER**),
2. **embedding** — compute the OpenFold/Evoformer representation the DiG model conditions on
   (**now in-pipeline**; skipped if you pass a precomputed `.pkl`),
3. **inference** — run your DiG checkpoint (`init` / `init_cdr_mask` / `vanilla_no_init`).

kinapse stays lightweight: it never imports torch/OpenFold — it *orchestrates* the external
DiG inference + OpenFold embedding as subprocesses (the embedding runs `conda run -n <env> …`),
with **all paths configurable** (no hardcoded `/workspaces` or `/mnt/bob`).

## Run

```bash
# one TCR (embedding computed for you), using YOUR newest model:
python -m kinapse.conformer_generation.dig_runner \
    --pdb my_tcr.pdb --out out/ --mode init --n 200 \
    --checkpoint /path/to/your_newest_model.pth

# a folder (batch):  --pdb-dir tcrs/
# already have embeddings:  --pkl-dir embeddings/   (and optionally --no-compute-embeddings)
```
```python
from kinapse.conformer_generation import run_one
run_one("my_tcr.pdb", "out/", n_samples=200, dig_mode="init")
```

## Configure (once) — `kinapse.yaml`

kinapse reads `./kinapse.yaml` (launch dir), `$KINAPSE_CONFIG`, or `~/.kinapse/config.yaml`.
Point it at your DiG checkout, checkpoint, OpenFold env and AlphaFold DBs:

```yaml
generation:
  run_inference:          /path/Graphormer/distributional_graphormer/protein/run_inference.py
  run_inference_addnoise: /path/Graphormer/distributional_graphormer/protein/run_inference_addnoise.py
  get_init_state:         /path/Graphormer/distributional_graphormer/protein/full_pipeline/get_init_state.py
  main_model:             /path/your_newest_model.pth      # or KINAPSE_CHECKPOINT_MAIN_MODEL / --checkpoint

evoformer:                                                  # the embedding step
  # backend "evoformer2" = DB-FREE: ColabFold *remote* MSA + alphaflow AF2 representation
  # (no local AlphaFold databases; needs internet + a GPU + the AF2 weights .npz). Recommended
  # when you don't have the ~2 TB genetic DBs. backend "openfold" = full local-DB jackhmmer path.
  backend:       evoformer2
  conda_env:     openfold_env                               # env with torch + openfold + alphaflow + esm
  env_run:       micromamba run -n
  weights:       /path/alphafold/params_model_1.npz         # AF2 params (not multimer)
  alphaflow_dir: /path/Graphormer/alphaflow
  evoformer_dir: /path/.../full_pipeline/evoformer_representation   # predict_evoformer2.py / make_MSA.py
  openfold_dir:  /path/Graphormer/openfold                  # alphaflow imports openfold
  # --- openfold backend only (ignored when backend: evoformer2) ---
  run_script:    run_pretrained_openfold_shortened.py
  config_preset: model_1_ptm
  model_device:  cuda:0
  mmcif_dir:  /path/alphafold/pdb_mmcif/mmcif_files
  uniref90:   /path/alphafold/uniref90/uniref90.fasta
  mgnify:     /path/alphafold/mgnify/mgy_clusters_2022_05.fa
  pdb70:      /path/alphafold/pdb70/pdb70
  uniclust30: /path/alphafold/uniclust30/uniclust30_2018_08/uniclust30_2018_08
  bfd:        /path/alphafold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
```
```bash
export KINAPSE_CONFIG=/path/to/kinapse.yaml
```

## Prerequisites

| step | needs |
|---|---|
| linking | **MODELLER** (licensed): `conda install -c salilab modeller` + `KEY_MODELLER` |
| embedding | the OpenFold conda env (`evoformer.conda_env`), OpenFold checkout, **AlphaFold DBs**, a GPU |
| inference | the DiG scripts (`generation.*`), your checkpoint (`main_model`), a GPU |

Recreate the OpenFold env from the bundled spec
([`environments/openfold_env.yml`](../environments/openfold_env.yml)):
```bash
micromamba create -f environments/openfold_env.yml               # python 3.9 / torch 1.12 + MSA tools
micromamba run -n openfold_env pip install -e /path/to/Graphormer/openfold   # OpenFold isn't on PyPI
```

**OpenFold's fused CUDA kernel** (`attn_core_inplace_cuda`) is an *optional* speed/memory
optimisation. Building it is finicky (needs a CUDA toolkit matching torch + GCC ≤10). You
don't need it for embeddings — make its imports optional so OpenFold falls back to standard
attention (used unless `use_memory_efficient_kernel`/`inplace_safe` are set):
```python
# in openfold/utils/kernel/attention_core.py and openfold/model/structure_module.py:
try:
    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")
except ImportError:
    attn_core_inplace_cuda = None
```

The embedding writes `<name>_output_dict.pkl` under the run dir; kinapse normalises it to
`<pdb_name>.pkl`. Every mode uses `get_checkpoint('main_model')`, so `--checkpoint` /
`KINAPSE_CHECKPOINT_MAIN_MODEL` swaps in your newest model everywhere.
