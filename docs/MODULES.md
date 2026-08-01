# kinapse module map

The single source of truth for what exists, where it lives, and how ready it is.
Keep this in sync with `docs/kinapse_architecture.drawio`. One concern → one module
→ one extra. Import any module without its extra installed — it stays lazy and only
fails when you actually use a feature that needs a missing dependency.

**Kinds:** *core* (always installed) · *science* (kinapse's own algorithms) ·
*runner* (pluggable external models via `kinapse.runners`) · *data/research*.

**Status:** `stable` (works) · `scaffold` (structure + interface, partial/no logic) ·
`planned` (declared only).

## Core — `pip install kinapse`
| module | purpose | status | key API |
|---|---|---|---|
| `kinapse.config` | env/YAML path resolution (no hardcoded paths) | stable | `get_paths()`, `so3_dir()`, `consensus_output_dir()` |
| `kinapse.regions` | IMGT region constants | stable | `CDR_FR_RANGES`, `VARIABLE_RANGE` |
| `kinapse.runners` | the pluggable-model engine (registry + isolated backends + entry-point discovery) | stable | `RunnerSpec`, `register`, `specs`, `run`, `get` |

## Science modules
| module | extra | purpose | status | key API |
|---|---|---|---|---|
| `kinapse.structures` | `[structures]` | load/prep TCR & pMHC (numbering, pairing, linkers, pmhc) + interface characterization | stable | `TCR`, `TCRPairView`, `TrajectoryView`, `load_tcr`, `analyze_interface`, `TCRpMHC.interface()` |
| `kinapse.geometry` | `[geometry]` | α/β inter-domain docking-angle geometry (the TCR's own α/β domains) | stable | `calc_tcr_geometry`, `calc_tcr_geometry_MD` |
| `kinapse.dynamics_analysis` | `[dynamics]` | ensemble/MD analysis + metrics + structure features + reducers + PMF/JSD (MSM/NMA planned) | stable | `rmsd_tm`, `run_ca_dist`, `oriol_analysis`, `align_*` |
| `kinapse.dynabind` | `[dynabind]` | ★ novel: ensemble → dynamics features → binding/cross-reactivity | scaffold | `featurize`, `predict` |

## Runner tiers (pluggable external models)
Each exposes `available()` (list specs) and `run(name, inputs)`; models are declared in the tier's `registry.py`. **Add your own:** see [`ADDING_A_MODEL.md`](ADDING_A_MODEL.md).
| module | extra | wraps | status | registered (all `planned`/`scaffold` until wired) |
|---|---|---|---|---|
| `kinapse.sequence_embedding` | `[sequence]` | sequence embedders | stable API | `mmseqs2_msa`, `openfold_evoformer`, `esm2`; native `pdb_to_fasta` |
| `kinapse.structure_prediction` | `[modelling]` | structure predictors | scaffold | `alphafold3`, `boltz2`, `tcrdock`, `tcrmodel2`, `immunebuilder` |
| `kinapse.conformer_generation` | `[generation]` | ensemble generators | stable API | `dig`, `alphaflow`, `bioemu`; native `postprocess` |
| `kinapse.binding_prediction` | `[binding]` | TCR-pMHC binding/specificity | scaffold | `nettcr`, `tulip`, `mixtcrpred`, `stag`, `tcren` |
| `kinapse.docking` | `[docking]` | docking engines | scaffold | `haddock`, `rosettadock`, `cluspro` |
| `kinapse.structure_analysis` | `[structure_analysis]` | external TCR structure annotators | stable API | `stcrpy` (external env via `KINAPSE_STCRPY_PYTHON`) |
| `kinapse.scoring` | `[scoring]` | interface scoring via **ifscore** | stable ✔ | `geometry_scoring`, `dockq`, `prodigy`, `foldx`, `rosetta`, `haddock`, `esmif`, `proteinmpnn`, `voromqa`, `voroif_gnn`, `zrank`; energy `mmgbsa`/`mmpbsa`/`rosetta_flexddg`/`fep` (planned) |

`kinapse.scoring` has a richer native API (`score`, `score_batch`, `score_tcr_pmhc`,
`available_scorers`) and delegates to the separate `ifscore` package — see
[`scoring.md`](scoring.md). Its interface-geometry scorer is surfaced as
`geometry_scoring` so it never clashes with the `geometry` module.

## Data / research
| module | extra | purpose | status | key API |
|---|---|---|---|---|
| `kinapse.datasets` | `[datasets]` | loaders (ATLAS, STCRDab, TCR3d, SKEMPI, VDJdb, IEDB, 10x, DMS) + leakage-aware splits | scaffold | `CATALOG`, `load`, `split` |
| `kinapse.benchmarks` | `[bench]` | scorer benchmark (GT vs modelled vs negatives) + ensemble-quality (scaffold) | scorer bench ✔ | `run_scorer_benchmark`, `agreement_analysis`, `discrimination_analysis` |

## Cross-cutting
- `kinapse.pipelines` — thin, named end-to-end workflows composing the modules (`benchmark`, `best_method`, `analyse_md`).
- `kinapse.cli` — the `kinapse` command (`info`, `prep`, `geometry`, `pipelines`, `score`).

## Back-compat aliases (old import paths still work)
`kinapse.embedding` → `sequence_embedding` · `kinapse.generation` → `conformer_generation` · `kinapse.analysis` → `dynamics_analysis`. Legacy `PCA_methods` / `embed_assesment` live under `dynamics_analysis/_legacy/`.
