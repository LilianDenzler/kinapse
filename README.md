# kinapse

**Modular toolkit for T-cell receptor (TCR) / TCR-pMHC structural dynamics.**

🌐 **Website:** https://liliandenzler.github.io/kinapse/ · 📖 **Docs:** [`docs/`](docs/)

`kinapse` reorganises the former `TCR_Metrics` research code into a single,
installable, shareable Python package. It measures how well a generative model
(DiG, AlphaFlow, AF3, sampled baselines) reproduces the conformational dynamics
of a TCR by comparing its predicted ensemble against ground-truth molecular
dynamics — region by region (CDR loops) — via low-dimensional embeddings,
free-energy surfaces, and Jensen–Shannon divergence.

It is built as **five independently usable sub-packages** that also compose into
full, configurable pipelines.

```
kinapse
├── structures   ①  load & prep TCR / TCR-pMHC structures        ← the basis
├── embedding    ②  sequence embedders (MSA, Evoformer)
├── geometry     ③  TCR α/β inter-domain docking geometry
├── analysis     ④  structure & MD analysis + dynamics metrics
└── generation   ⑤  generative conformer sampling (DiG)
     + pipelines      composable end-to-end workflows
     + config         env/YAML path resolution (no more hardcoded /mnt paths)
     + cli            the `kinapse` command
```

Every capability from the original `TCR_Metrics` is preserved — see
[`docs/migration_from_tcr_metrics.md`](docs/migration_from_tcr_metrics.md) for the
full old→new module map.

---

## Install

**From a fresh clone — one command** (needs conda / mamba / micromamba on PATH):

```bash
git clone <your-repo-url> kinapse && cd kinapse
./setup.sh          # creates the `kinapse` conda env + installs the package + runs smoke tests
conda activate kinapse
kinapse info
```

`setup.sh` builds the environment from [`environment.yml`](environment.yml), which
also editable-installs the package with its `numbering` + `reduce` extras. Prefer
to do it by hand?

```bash
conda env create -f environment.yml      # env + package (numbering, reduce)
conda activate kinapse
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

### ① Load & prep a structure (the basis)

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

### ③ Geometry

```python
from kinapse.geometry import calc_tcr_geometry, calc_tcr_geometry_MD
calc_tcr_geometry("1kgc.pdb", "out/")           # 6-parameter α/β docking geometry
df = calc_tcr_geometry_MD("1kgc.xtc", "1kgc.pdb")   # per-frame geometry DataFrame
```

### ④ Analysis & dynamics metrics

```python
from kinapse.analysis import run_ca_dist, oriol_analysis, rmsd_tm
# featurize → reduce → embedding-quality metrics → PMF/JSD, per region
```

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

- [`docs/architecture.md`](docs/architecture.md) — the five modules & data flow
- [`docs/quickstart.md`](docs/quickstart.md) — worked examples
- [`docs/pipelines.md`](docs/pipelines.md) — running & configuring pipelines
- [`docs/scoring.md`](docs/scoring.md) — interface scoring of TCR-pMHC complexes (`kinapse.scoring` → ifscore)
- [`docs/migration_from_tcr_metrics.md`](docs/migration_from_tcr_metrics.md) — old→new map

## Tests

```bash
pytest -q          # smoke tests: package imports, lazy loading, packaged data
```

## License

MIT (see [`LICENSE`](LICENSE)) — a suggested default; change it to whatever your
group prefers before sharing publicly.
