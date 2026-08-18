# VDJdb binder finder

For each TCR structure in a dataset, find **every pMHC it is a confirmed binder
for** in [VDJdb](https://vdjdb.cdr3.net/). CDR3α/β loops are read straight from
the structures with the kinapse loader and matched against VDJdb's paired
records.

## What it does
1. Load each `<ID>/<ID>.pdb` with `kinapse.structures.TCR` (IMGT renumbering +
   interface-based α/β pairing), and read the CDR3 loops via
   `TCRPairView.cdr_fr_sequences()`.
2. Rebuild each CDR3 in VDJdb's format — kinapse's `A_CDR3`/`B_CDR3` are IMGT
   **105–117** (loop only), VDJdb's `cdr3` is the full **104–118**, so we take
   `FR3[-1] + CDR3 + FR4[0]` (adds the conserved Cys104 / Phe·Trp118).
3. Index VDJdb into **paired α/β complexes** (rows sharing a `complex.id > 0`)
   and, by default, call a record the same TCR only when **both** CDR3α and
   CDR3β match (`--match paired`). An individual α or β matches many unrelated
   epitopes on its own; `--match {beta,alpha,either}` relaxes this.
4. Write, per matched TCR, every confirmed-binder pMHC with its `vdjdb.score`.

## ⚠️ No confirmed non-binders
VDJdb is **positive-only** — it records binders, never non-binders — so there
are none to retrieve here; every row is `label=confirmed_binder`. Confirmed
*non-binders* need an assay source (10x negative controls, IEDB negative
assays, or a `tcr_pmhc_db` interaction with `binding=False`). See the `tenx` /
`iedb` entries in `kinapse.datasets.CATALOG`.

## Prerequisites
```bash
pip install -e ".[structures]"     # in the kinapse env (biopython, anarcii, mdtraj)
```
Needs a local `vdjdb.slim.txt` (download from the VDJdb release, or reuse
`/mnt/larry/lilian/DATA/vdjdb/vdjdb.slim.txt`).

## Run
```bash
python find_vdjdb_binders.py \
    --dataset /mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD \
    --vdjdb   /mnt/larry/lilian/DATA/vdjdb/vdjdb.slim.txt
# explicit files:      python find_vdjdb_binders.py a.pdb b.pdb
# looser matching:     python find_vdjdb_binders.py --match beta
```

## Outputs (`results/`, git-ignored)
- **`vdjdb_confirmed_binders.csv`** — one row per (TCR, matched pMHC):
  `pdb_id, pair, cdr3a, cdr3b, label, epitope, antigen_gene, antigen_species,
  mhc_a, mhc_b, mhc_class, tcr_species, vdjdb_score, n_complexes, complex_ids,
  references`.
- **`tcr_cdr3_summary.csv`** — one row per TCR: extracted CDR3α/β + binder count.

## Generating the complexes with tFold-TCR

`run_tfold_binders.py` takes the confirmed-binder rows and predicts each TCR-pMHC
complex with **tFold-TCR** ([TencentAI4S/tfold](https://github.com/TencentAI4S/tfold))
via kinapse's `tfold_tcr` runner (`kinapse.structure_prediction.tfold`, an external
env behind a JSON contract — like the STCRpy runner).

Each tFold job is assembled as five chains:

| id | chain | sequence source |
|----|-------|-----------------|
| `B` | TCR β variable | source structure `<pdb_id>/<pdb_id>.pdb` (kinapse loader) |
| `A` | TCR α variable | source structure (kinapse loader) |
| `M` | MHC heavy (mature ectodomain) | IPD-IMGT/HLA fetch, truncated per `MHC_TRUNCATION.csv` (HLA-A/B/C → res 25–298) |
| `N` | β2-microglobulin | constant (tFold's own example B2M) |
| `P` | peptide | the VDJdb epitope |

The TCR of each row is the structure it was extracted from, so α/β come straight
back from that PDB; only the pMHC varies across a TCR's rows.

### Setup (one-time)
```bash
git clone https://github.com/TencentAI4S/tfold.git
conda create -n tfold-tcr python=3.8 pip -y
conda run -n tfold-tcr pip install "torch>=2.0" deepspeed==0.12.3 termcolor==2.3.0 \
    biopython==1.79 ml-collections==0.1.1 dm-tree==0.1.8 numpy==1.21.2 modelcif==0.9 scipy
conda run -n tfold-tcr pip install --no-deps ./tfold        # the cloned repo
```
Model weights (ESM-PPI-650M-TCR + tFold-TCR trunk, a few GB) auto-download from
Zenodo on first run into `~/.cache/torch/hub/checkpoints`.

### Run (in the `kinapse` env; it shells out to the tFold env)
```bash
python run_tfold_binders.py --dry-run          # assemble inputs only (no GPU)
KINAPSE_TFOLD_PYTHON=/path/to/tfold-tcr/bin/python \
  python run_tfold_binders.py                  # full run (21 complexes)
python run_tfold_binders.py --limit 2          # smoke-test two
```
`--tfold-python` / `--tfold-repo` override the env + repo (defaults point at the
local `tfold-tcr` env and `~/TCR_interface/tfold-tencent`).

### Outputs (`results/tfold_complexes/`, git-ignored)
- one predicted PDB per complex: `<pdb_id>__<epitope>__<allele>.pdb`
  (tFold writes ipTM / pTM into the `REMARK 250` header)
- `tfold_jobs.json` — the exact inputs sent to tFold (reproducible)
- `mhc_sequences.csv` — resolved allele → mature-ectodomain cache
- `tfold_summary.csv` — name, pdb_id, epitope, allele, status, ipTM, pTM, path

## Negative (non-binder) dataset

`make_negatives.py` builds putative non-binders for each TCR. A negative pMHC is
an **existing** pMHC from a database (VDJdb) whose **MHC is one the TCR is a
confirmed binder for**, but whose **peptide is not any confirmed-binder peptide
of that TCR** — matched on MHC so only the peptide differs (hard negatives). The
peptide is a real epitope presented by that MHC, so the pMHC itself exists.

```bash
python make_negatives.py                     # all candidate negatives (default)
python make_negatives.py --length-matched    # only peptides of a positive's length
python make_negatives.py --max-per-tcr 5     # cap per TCR (length-matched/high-conf first)
python make_negatives.py --exclude-looser-binders   # also drop CDR3β-level known binders
```

Outputs (`results/`):
- `vdjdb_negatives.csv` — same columns as the positives file (so it's a drop-in
  input to `run_tfold_binders.py`) + `neg_source`, `pep_len`, `length_matched`,
  `neg_epitope_vdjdb_score`; `label=non_binder`.
- `vdjdb_labeled_dataset.csv` — positives + negatives in one labeled file.

On this dataset: **122 negatives** (63 length-matched) for the 21 positives — but
skewed by VDJdb's thin per-allele pool (HLA-A\*02:01 TCRs get 33–38 each; the
HLA-A\*24:02 TCR 3VXQ gets only 1). For more/harder negatives on rare alleles,
use a larger presented-peptide DB (IEDB / NetMHCpan-predicted binders).

**Caveat — inferred negatives.** "Not recorded to bind" ≠ "confirmed non-binder";
some may be cross-reactive. Negatives are excluded by the same *paired* α+β
definition used for the positives, so a peptide bound only under looser matching
can still appear (e.g. WT TAX `LLFGYPVYV` for the A6 TCR 3QH3);
`--exclude-looser-binders` removes those.

To fold the negatives too:
```bash
python run_tfold_binders.py --csv results/vdjdb_negatives.csv --out results/tfold_negatives
```

## Structural annotation (solved bound complexes)

`annotate_structures.py` tags each confirmed binder with its experimentally
**solved bound TCR-pMHC PDB** (curated from RCSB / TCR3d) → `results/
vdjdb_confirmed_binders_annotated.csv`, adding `bound_pdb`, `structure_status`
(`native_bound` | `modified_bound` | `sequence_only`), `ligand_note`,
`peptide_modified`, `modification`, `row_source`.

```bash
python annotate_structures.py --drop-modified                  # -> annotated CSV (clean)
# regenerate negatives from the annotated binder set, and keep the modified-ligand
# strings out of the negative pool too (their sequence ≠ the real ligand):
python make_negatives.py --csv results/vdjdb_confirmed_binders_annotated.csv \
    --exclude-looser-binders --exclude-peptides-csv results/excluded_modified_ligands.csv
```

Result: **20 confirmed binders, all `native_bound`**, every one with an
RCSB-verified solved complex. Two curation points beyond the raw VDJdb query:

1. **Added `1AO7`** — the canonical A6 / WT-Tax `LLFGYPVYV` / HLA-A\*02:01 complex
   the paired VDJdb query missed (`row_source=literature_added`). It had been
   leaking in as a 3QH3 *negative*; negatives are regenerated so it is now excluded.
2. **Removed 2 chemically-modified ligands** (`--drop-modified`) whose VDJdb
   peptide string does not represent the real ligand — moved to
   `results/excluded_modified_ligands.csv` for provenance, and kept out of the
   negative pool:
   - `LLFGKPVYV` → **2GJ6** = Tax Y5**K-IBA** (Lys5 + 3-indolebutyric-acid hapten).
   - `LLFGPVYV` → **3D39 / 3D3V** = Tax with a **fluorinated Phe** at position 5;
     the VDJdb 8-mer had dropped the modified residue entirely.

### Is this all the binders? Yes, within VDJdb.
These are highly-specific monoclonal TCRs, so the set is genuinely small. Relaxing
the match from paired α+β to **β-only** adds essentially nothing: only `LLFGYPVYV`
for 3QH3 (WT Tax = 1AO7, already added) and one score-0 public-β artefact
(`GLCTLVAML` for 1KGC/LC13, not a real binder); every other TCR is unchanged under
β-only / α-only / either-chain. More binders would need a different database
(IEDB, McPAS) or literature, not looser matching.

| TCR | binders (VDJdb, paired) | β-only | note |
|---|---|---|---|
| 1KGC (LC13) | 3 | 4 | +GLCTLVAML score-0 (artefact) |
| 2BNU (1G4) | 2 | 2 | — |
| 3DX9 (DM1) | 1 (×2 MHC rows) | 1 | — |
| 3QH3 (A6) | 7 (+1AO7 = full 9-ligand series) | 9 | +WT Tax (added) |
| 3SKN (RL42) | 1 | 1 | — |
| 3VXQ (H27-14) | 2 | 2 | — |
| 4JFH (α24β17) | 3 | 3 | — |

## Notes on the `CORY_ORIOL_MERGED_MD` dataset
- 21/22 complexes pair cleanly; **7 match** a paired VDJdb record (21 pMHC
  rows). Hits are the expected public TCRs — e.g. 3QH3 = A6 → HTLV-1 TAX + its
  altered-peptide-ligand panel (8 pMHCs), 2BNU = 1G4 → NY-ESO-1 SLLMWITQ(C/V),
  4JFH → MART-1 APLs, 1KGC/3SKN → EBV FLRGRAYGL, 3VXQ → HIV Nef.
- **8YJ3** does not form a pair under kinapse's interface pairing (its chains
  number atypically) so it is skipped; it has no VDJdb match either way.
