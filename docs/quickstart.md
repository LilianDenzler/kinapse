# Quickstart

Assumes you installed the package (see the README). All examples use a TCR PDB
`example.pdb` and, where relevant, an MD trajectory `example.xtc`.

## 1. Load & prep (module ①)

```python
from kinapse.structures import load_tcr

tcr = load_tcr("example.pdb", traj="example.xtc")
print(len(tcr.pairs), "TCR pair(s)")

pair = tcr.pairs[0]
print(pair.cdr_fr_sequences)           # per-region one-letter sequences
var = pair.variable_structure          # Biopython Fv structure
sub = pair.domain_subset(["A_CDR3"])   # just the α CDR3 loop
```

Trajectory sliced to a region:

```python
tv = pair.traj                         # TrajectoryView (mdtraj-backed)
```

## 2. Geometry (module ③)

```python
from kinapse.geometry import calc_tcr_geometry, calc_tcr_geometry_MD

calc_tcr_geometry("example.pdb", "geom_out/")          # writes angles + optional PyMOL vis
df = calc_tcr_geometry_MD("example.xtc", "example.pdb") # per-frame BA/BC/AC/dc DataFrame
```

## 3. Analysis & metrics (module ④)

```python
from kinapse.analysis import run_ca_dist
# run_ca_dist(tv_gt, tv_pred, outdir, regions, reducer="pca", ...) →
#   *_embedding.npz + *_metrics.json (trustworthiness, Mantel r)
```

Free-energy surface + Jensen–Shannon divergence (GT vs model):

```python
from kinapse.analysis import oriol_analysis   # PMF (hist + KDE), JSD, extreme-frame selection
```

RMSD / TM-score:

```python
from kinapse.analysis import rmsd_tm
```

## 4. Generation (module ⑤)

```python
from kinapse.generation import runall, process_output
# runall(...) drives the external DiG sampler; process_output(...) folds the
# generated per-frame PDBs into an .xtc (stripping the α–β linker).
```

## 5. Sequence embedding (module ②)

```python
from kinapse.embedding import pdb_to_fasta
pdb_to_fasta("example.pdb", "example.fasta")
# MSA + Evoformer paths: kinapse.embedding.msa / kinapse.embedding.evoformer
```

## 6. Config instead of hardcoded paths

```python
from kinapse.config import get_paths
paths = get_paths()                    # reads KINAPSE_* env vars / kinapse.yaml
paths.get_output_dir("benchmarks")
```
