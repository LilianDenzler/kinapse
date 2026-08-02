# kinapse

**Modular toolkit for T-cell receptor (TCR) / TCR-pMHC structural dynamics.**

🌐 **Website:** https://liliandenzler.github.io/kinapse/ · 📖 **Docs:** [`docs/`](docs/)

`kinapse` reorganises the former `TCR_Metrics` research code into a single,
installable, shareable Python package. It measures how well a generative model
(DiG, AlphaFlow, AF3, sampled baselines) reproduces the conformational dynamics
of a TCR by comparing its predicted ensemble against ground-truth molecular
dynamics — region by region (CDR loops) — via low-dimensional embeddings,
free-energy surfaces, and Jensen–Shannon divergence.

It is built as **modular sub-packages** (one concern → one module → one extra) on a
**lightweight core**, and it's trivial to plug in your own external model.

```
kinapse  (lightweight core: config · regions · runners)
│
├─ science ─ structures · geometry · dynamics_analysis · dynabind ★novel
├─ runners ─ sequence_embedding · structure_prediction · conformer_generation
│            binding_prediction · docking · scoring · structure_analysis (STCRpy)
└─ data ──── datasets · benchmarks                         + pipelines · cli
```

`pip install kinapse` is tiny; add only what you use — `kinapse[structures]`,
`kinapse[dynamics]`, `kinapse[binding]`, `kinapse[all]`. The full module map is in
[`docs/MODULES.md`](docs/MODULES.md), and adding your own model is a two-file job:
[`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md).

Every capability from the original `TCR_Metrics` is preserved — see
[`docs/migration_from_tcr_metrics.md`](docs/migration_from_tcr_metrics.md) for the
full old→new module map.

---

## Install

**From a fresh clone — one command** (needs conda / mamba / micromamba on PATH):

```bash
git clone <your-repo-url> kinapse && cd kinapse
./setup.sh          # creates the `kinapse` env + installs the package + runs smoke tests
conda activate kinapse       # or: micromamba activate kinapse  (setup.sh prints the exact command)
kinapse info
```

`setup.sh` builds the environment from [`environment.yml`](environment.yml), which
also editable-installs the package with its `numbering` + `reduce` extras. Prefer
to do it by hand?

```bash
conda env create -f environment.yml      # env + package (numbering, reduce)
conda activate kinapse                   # micromamba users: micromamba activate kinapse
```

Pure-pip, no conda (if you can get the MD stack from wheels — works on Linux):

```bash
pip install -e ".[numbering,reduce]"     # core + ANARCII numbering + deeptime/TICA
# or straight from GitHub, no clone:
pip install "kinapse[numbering,reduce] @ git+https://github.com/<you>/kinapse"
```

Continuous integration ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
runs exactly that pip install + the smoke tests on every push, so a green badge
means "clones and installs cleanly."

Optional/external tooling, installed only if you use that feature:

| Feature | Needs |
|---|---|
| Legacy ANARCI numbering (default loader path) | `bioconda::anarci` |
| PyMOL visualisation pipelines | `pymol-open-source` (conda) |
| α–β linker building (`structures.linkers`) | MODELLER (licensed) |
| Evoformer embeddings (`embedding`) | `torch`, an OpenFold checkout, MMseqs2-GPU binary |
| TMalign / ProFit alignment engines | those binaries on `PATH` |
| Generation (`generation`) | the external DiG diffusion model |

---

## Quick start

### Load & prep a structure (the basis)

```python
from kinapse.structures import TCR, load_tcr

tcr = load_tcr("1kgc.pdb", traj="1kgc.xtc")     # IMGT-renumber + pair α/β + attach MD
pair = tcr.pairs[0]
print(pair.cdr_fr_sequences)                    # {'A_CDR3': '...', 'B_CDR3': '...', ...}
var = pair.variable_structure                   # Biopython structure of the Fv
cdr3 = pair.traj                                # TrajectoryView, sliceable by IMGT region
```

pMHC / full complexes are scaffolded (API-ready, TCR-only logic for now):

```python
from kinapse.structures import TCRpMHC          # loads the TCR half + a provisional pMHC
```

### Geometry

```python
from kinapse.geometry import calc_tcr_geometry, calc_tcr_geometry_MD
calc_tcr_geometry("1kgc.pdb", "out/")           # 6-parameter α/β docking geometry
df = calc_tcr_geometry_MD("1kgc.xtc", "1kgc.pdb")   # per-frame geometry DataFrame
```

### Analysis & dynamics metrics

```python
from kinapse.analysis import run_ca_dist, oriol_analysis, rmsd_tm
# featurize → reduce → embedding-quality metrics → PMF/JSD, per region
```

### Benchmark scorers & structural quality

Benchmark every interface scorer over ground-truth vs modelled vs negative complexes
(agreement + leakage-aware discrimination), and grade each model against its ground truth
by **Cα iRMSD over the 6 CDRs → HQ/MQ/AQ/LQ tiers**:

```python
from kinapse.benchmarks import run_scorer_benchmark, structural_agreement, assign_tier

res = run_scorer_benchmark(gt_dir, model_dir, neg_dir, out_dir="out", scorers="all", n_jobs=16)
res["agreement"]; res["discrimination"]; res["tiers"]

s = structural_agreement(model_pdb, gt_pdb)      # per-CDR + Cα iRMSD; loader="stcrpy" optional
assign_tier(s["struct__cdr_irmsd"])              # "HQ" / "MQ" / "AQ" / "LQ"
```

Chain identification + CDR annotation is pluggable — kinapse's native loader or external
**STCRpy** (`--loader stcrpy`, via `KINAPSE_STCRPY_PYTHON`). See
[`docs/benchmarks.md`](docs/benchmarks.md), the ready-made drivers in
[`experiments/scorer_benchmarking/`](experiments/scorer_benchmarking/), and the loader
agreement check in [`experiments/loader_comparison/`](experiments/loader_comparison/).
Tutorial: [`examples/07_benchmarking.ipynb`](examples/07_benchmarking.ipynb).

### CLI

```bash
kinapse info                       # version, config & packaged-data paths
kinapse prep    1kgc.pdb --traj 1kgc.xtc --out prep/
kinapse geometry 1kgc.pdb --out geom/
kinapse pipelines                  # list the migrated end-to-end workflows
```

---

## Configuration

No more hardcoded absolute paths. Point `kinapse` at your data via a YAML file or
environment variables — see [`config.example.yaml`](config.example.yaml):

```bash
export KINAPSE_OUTPUT=/mnt/dave/lilian/DIG_VARIATION_OUTPUTS
# or drop a kinapse.yaml in your working directory
```

```python
from kinapse.config import get_paths
paths = get_paths()
out = paths.get_output_dir("benchmarks")
```

---

## Pipelines

The migrated end-to-end workflows live under `kinapse.pipelines` and preserve the
original two-phase benchmark design (collect embeddings → shared-bin PMF/JSD →
tables). List them with `kinapse pipelines`; details in
[`docs/pipelines.md`](docs/pipelines.md).

## Docs

- [`docs/MODULES.md`](docs/MODULES.md) — the living module map (what exists, its extra, status)
- [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md) — plug in your own external model
- [`docs/architecture.md`](docs/architecture.md) — the modules & data flow
- [`docs/quickstart.md`](docs/quickstart.md) — worked examples
- [`docs/pipelines.md`](docs/pipelines.md) — running & configuring pipelines
- [`docs/scoring.md`](docs/scoring.md) — interface scoring of TCR-pMHC complexes (`kinapse.scoring` → ifscore)
- [`docs/benchmarks.md`](docs/benchmarks.md) — scorer benchmark, structural HQ/MQ/AQ/LQ tiers, native vs STCRpy loader
- [`docs/migration_from_tcr_metrics.md`](docs/migration_from_tcr_metrics.md) — old→new map

## Tests

```bash
pytest -q          # smoke tests: package imports, lazy loading, packaged data
```

## License

MIT (see [`LICENSE`](LICENSE)) — a suggested default; change it to whatever your
group prefers before sharing publicly.
