# Loader comparison — kinapse (native) vs STCRpy

Do kinapse's own loader and OPIG **STCRpy** identify the **same TCR/pMHC chains** and the
**same CDR residues**? This experiment runs both loaders on the same structures and reports
where they agree and where they differ — at the level of *loading*, before any scoring.

(For a comparison of the whole scorer **benchmark** under each loader, see
[`../scorer_benchmarking/`](../scorer_benchmarking/) and its `compare_loaders.py`.)

## What it checks (`compare_loading.py`)
Per structure, both loaders are run and compared on:

- **Chains** — receptor = TCR α/β, ligand = pMHC (everything else), as *original PDB chain ids*.
  Native via `kinapse.scoring.infer_tcr_pmhc_chains`; STCRpy via `get_VA`/`get_VB` +
  `get_antigen`/`get_MHC`. Reports whether the **receptor set** and **ligand set** match.
- **CDRs** — the residues each loader assigns to the 6 CDR loops, keyed by
  `(region, chain, IMGT number)`. Native via kinapse's IMGT renumbering + region ranges;
  STCRpy via its ANARCI IMGT fragments (`get_CDRs`). Per CDR it reports:
  - `n_native`, `n_stcrpy`, `n_shared`, `n_only_*` — residue counts;
  - `jaccard` — |shared| / |union| of the residue sets (1.0 = identical);
  - `shared_ca_rmsd` — for residues **both** put in that CDR, the Cα distance between them.
    ≈0 means the two loaders numbered the *same physical residue* the same way; a large value
    means the IMGT numberings disagree even though both call it (say) CDR3.

## Prerequisite — where is STCRpy? (set `KINAPSE_STCRPY_PYTHON`)
STCRpy is **not** part of kinapse and is not installed by default — kinapse runs it as a
fully external model. Install it in its **own** environment (it pulls ANARCI models, PLIP,
OpenBabel), then point kinapse at that environment's interpreter:

```bash
# create a dedicated env and install STCRpy (one-time)
micromamba create -n stcrpy python=3.10 -y
micromamba activate stcrpy
pip install stcrpy && ANARCI --build_models && pip install plip

# find the interpreter path and tell kinapse to use it
which python                       # e.g. /home/you/.local/share/mamba/envs/stcrpy/bin/python
export KINAPSE_STCRPY_PYTHON="$(which python)"
```
`KINAPSE_STCRPY_PYTHON` is the **absolute path to the `python` of the env that has STCRpy**
— not kinapse's env. If it's unset (and `stcrpy` isn't importable in the active interpreter),
this script prints the install hint and exits.

## Run
```bash
export KINAPSE_STCRPY_PYTHON=/path/to/stcrpy-env/bin/python
export ANARCI_CPU=1

# a directory of TCR-pMHC PDBs (default: the TCR3d GT set)
python compare_loading.py --pdb-dir /path/to/TCR_complexes --limit 50 -j 8

# or explicit files
python compare_loading.py 1ao7.pdb 1bd2.pdb
```
Each structure is loaded by **both** backends, so it's not fast (native load + one STCRpy
subprocess per structure); use `--limit` for a quick look and `-j` to parallelise.

## Outputs (`results/`, or `--out`)
- `per_structure.csv` — one row per structure: `rec_match`, `lig_match`, `cdr_all_identical`,
  `cdr_mean_jaccard`, `cdr_shared_ca_rmsd`, the chain ids from each loader, and any
  `native_error`/`stcrpy_error`.
- `per_cdr.csv` — one row per structure×CDR with the counts, `jaccard`, and `shared_ca_rmsd`.

The script also prints a summary: % of structures with identical receptor / ligand chains,
% with all 6 CDRs residue-identical, mean per-CDR Jaccard, the shared-residue Cα agreement,
a per-CDR breakdown, and a list of any structures whose chain assignments disagree.
