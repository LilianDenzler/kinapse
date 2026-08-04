# kinapse tutorials

**New to kinapse? Run it in the browser — no install:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LilianDenzler/kinapse/blob/main/examples/colab_quickstart.ipynb)
&nbsp;[`colab_quickstart.ipynb`](colab_quickstart.ipynb) — pip-only, CPU: load a TCR, read its CDRs,
the model registry, structural HQ/MQ/AQ/LQ tiers, and interface scoring. (Private repo → Colab
asks you to authorize GitHub; paste a token in the install cell.)

The notebooks below are the **local** tutorials (run them from this `examples/` directory with the
full conda/MD stack installed).

| notebook | topic | install |
|---|---|---|
| [01_getting_started.ipynb](01_getting_started.ipynb) | the map, config, load a TCR | `kinapse[structures]` |
| [02_structures.ipynb](02_structures.ipynb) | load/prep, numbering, pairing, regions, pMHC | `kinapse[structures]` |
| [03_geometry.ipynb](03_geometry.ipynb) | α/β docking-angle geometry (per frame) | `kinapse[structures,geometry]` |
| [04_dynamics_analysis.ipynb](04_dynamics_analysis.ipynb) | features → reduce → PMF + JSD | `kinapse[structures,dynamics]` |
| [05_scoring.ipynb](05_scoring.ipynb) | interface scoring (ifscore) | `kinapse[scoring]` + ifscore |
| [06_add_your_own_model.ipynb](06_add_your_own_model.ipynb) | the pluggable-runner engine | `kinapse` |
| [07_benchmarking.ipynb](07_benchmarking.ipynb) | scorer benchmark, structural HQ/MQ/AQ/LQ tiers, native vs STCRpy loader | `kinapse[bench,structures]` (+ ifscore) |

`data/` holds a small example TCR (`example_tcr.pdb`, an α/β Fv) and a **synthetic**
ensemble (`example_ensemble.xtc`, generated from it) so the analysis notebooks run without
external MD data. Numbering uses ANARCII (`legacy_anarci=False`); set `ANARCI_CPU=1` to
force CPU. The full module map is in [../docs/MODULES.md](../docs/MODULES.md).
