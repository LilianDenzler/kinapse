# Scoring TCR-pMHC complexes (`kinapse.scoring` → ifscore)

kinapse scores protein-interface quality/energy by delegating to
[**ifscore**](https://github.com/LilianDenzler/scoring_functions) rather than
vendoring any scorer. This keeps kinapse light: ifscore's core has **no scorer
dependencies**, and each scorer (DockQ, PRODIGY, `geometry`, …) runs in its own
isolated environment behind a subprocess/JSON contract — so mutually
incompatible tools (DockQ pins `numpy<2`, PRODIGY needs `numpy>=2`, PyRosetta,
torch-based models) can be used together and can never break kinapse's env.

This backs **§5 "scoring structures"** of the architecture roadmap
(DockQ / RMSD / TM-score + energy scoring).

## Install (ifscore is an optional, separate package)

ifscore is not a hard dependency of kinapse — it is imported lazily, so kinapse
installs and runs fine without it. Add it when you want scoring:

```bash
pip install "ifscore @ git+https://github.com/LilianDenzler/scoring_functions"
# or, from a local checkout:
pip install -e /path/to/scoring_functions
```

Then provision the scorers you want (one-off; uses [uv](https://github.com/astral-sh/uv)):

```bash
ifscore doctor                 # what is ready
ifscore install all  # build every scorer's isolated env (geometry needs nothing)
# the `geometry` scorer needs nothing and works immediately
```

## Use — Python

```python
from kinapse import scoring

# 1) Auto-derive chains from the TCR: receptor = TCR α/β, ligand = pMHC (the rest)
scores = scoring.score_tcr_pmhc("complex.pdb", native="native.pdb", scorers="default")
# -> {'geometry__bsa': 1523.4, 'dockq__dockq': 0.62, 'prodigy__ba_val': -11.2, ...}

# 2) Or specify chains explicitly (any interface, not just TCR-pMHC)
scores = scoring.score("complex.pdb", native="native.pdb",
                       rec=["D", "E"], lig=["A", "B", "C"], scorers="fast")

# 3) Batch a directory or a manifest CSV -> pandas DataFrame
df = scoring.score_batch(models="models/", native="natives/",
                         rec="D,E", lig="A,B,C", scorers="all", n_jobs=16)

# What can I run?
scoring.available_scorers()   # {name: {metrics, backend, needs_native, description}}
```

`infer_tcr_pmhc_chains(pdb)` is the helper behind auto-derivation: it loads the
TCR (IMGT numbering + α/β pairing) to find the TCR chains, and treats every
other polymer chain as the ligand (pMHC). It returns the **original** PDB chain
ids, ready for ifscore.

Auto-derivation numbers the TCR, which by default uses legacy ANARCI (bioconda).
On a pip-only install, pass `legacy_anarci=False` (Python) or `--new-anarci`
(CLI) to number with the pip-installable ANARCII instead.

## Use — CLI

```bash
kinapse score complex.pdb --native native.pdb --auto-chains --scorers default
kinapse score complex.pdb --auto-chains --new-anarci        # pip-only (ANARCII) numbering
kinapse score complex.pdb --rec D,E --lig A,B,C --scorers fast --out scores.json
kinapse score --list-scorers
```

## Scorers (today)

| scorer | backend | needs native | gives |
|---|---|---|---|
| `geometry` (`geometry_scoring`) | in-process (no deps) | no | BSA, SASA, contact counts, interface-residue counts, min distance |
| `dockq` | isolated env | yes | DockQ docking-quality metrics |
| `prodigy` | isolated env | no | PRODIGY predicted binding affinity / interface energy |
| `foldx` | isolated env | no | FoldX empirical ΔΔG (academic licence, see Install) |
| `rosetta` | isolated env | no | Rosetta InterfaceAnalyzer: dG_separated, dSASA, shape complementarity, packstat |
| `haddock` | isolated env | no | HADDOCK empirical docking score |
| `esmif` | isolated env | no | ESM-IF1 inverse-folding log-likelihood (reference-free quality) |
| `proteinmpnn` | isolated env | no | ProteinMPNN inverse-folding log-likelihood (reference-free) |
| `voromqa` | isolated env | no | VoroMQA interface energy (Voronoi statistical potential) |
| `voroif_gnn` | isolated env | no | VoroIF-GNN interface quality (CASP15-top GNN) |
| `zrank` | isolated env | no | ZRANK / ZRANK2 docking-pose rescoring energy |

Presets: `fast` = (geometry, dockq), `default` = (+ prodigy), `all` = everything
registered. ifscore's registry is extensible — each scorer runs in its own env.

## Free-energy / physics-based scorers (energy ranking)

These rank binding **energy** from an ensemble of MD snapshots. They are declared in
`kinapse.scoring` (`status: planned`) and provisioned in isolated GPU envs via ifscore
(they need OpenMM/Amber). They give **relative rankings + per-residue hotspots**, not
absolute ΔG. The cost↔accuracy ladder for TCR-pMHC:

| scorer | what | rigor / cost | notes |
|---|---|---|---|
| `mmgbsa` | MM-GBSA end-point ΔG over short MD | medium (~min/complex + GPU MD) | **primary** — TCR-validated (Crean 2022): GB-Neck2, **ε_int≈6**, ~5–15 short replicas, entropy usually omitted; per-residue decomposition = interface hotspots |
| `mmpbsa` | MM-PBSA (Poisson–Boltzmann solvent) | slower (~60 min/complex) | cross-check; better on charged interfaces |
| `rosetta_flexddg` | backrub-ensemble ΔΔG on mutations | cheap | orthogonal check for interface point mutations |
| `fep` | alchemical relative ΔΔG (OpenFE/OpenMM) | highest (~GPU-days) | reserve for a few top candidates |

**How MM-GBSA works.** End-point method: `ΔG_bind = G_complex − (G_receptor + G_ligand)`,
with `G = E_MM (bonded + elec + vdW) + G_solv (polar GB/PB + nonpolar γ·SASA) − T·S`.
Snapshots come from short MD; explicit water is stripped for a GB/PB continuum. The
single-trajectory protocol re-scores receptor/ligand on the complex's frames so bonded
terms cancel (low noise). Entropy (−TS) is usually dropped (limits it to *ranking*).
**Ensemble-averaging is the key accuracy lever** (rescoring R² 0.36→0.69 vs a single
structure). Charged TCR interfaces need `ε_int≈4–8` to avoid electrostatic over-counting.

Not a scorer but relevant: a fast **ML affinity** model (e.g. Boltz-2, ~1000× cheaper) is
a good *pre-filter* — that lives in `binding_prediction` / `structure_prediction`, not here.

Refs: Crean et al. 2022 *JCIM* (TCR MMPB/GBSA ranking, PMC9097153) · Genheden & Ryde 2015
(*Expert Opin. Drug Discov.*) · Wang/Hou 2019 *Chem. Rev.* · gmx_MMPBSA (Valdés-Tresanco 2021) · OpenFE.

## Note on the `geometry` name clash

ifscore's `geometry` scorer computes **interface** geometry (BSA / SASA /
contacts). That is different from [`kinapse.geometry`](architecture.md), which
computes the **α/β variable-domain docking angles** of a TCR. They are
complementary — just don't confuse the two.
