# unbound_TCR_MD_angles

**Question:** how much does each TCR's Vα/Vβ inter-domain docking geometry — the
shape/orientation of its combined CDR binding platform — explore during *unbound*
MD, before it ever sees pMHC?

[`angle_exploration.py`](angle_exploration.py) computes the packaged kinapse
6-parameter α/β docking geometry (`BA`, `AC1`, `AC2`, `BC1`, `BC2`, `dc` — the TCR
analogue of antibody VH–VL "ABangle") for every frame of every system in
`/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD` and visualises the sampled space.

The geometry is identical to `kinapse.geometry.calc_geometry.process` (biotite
`superimpose_structural_homologs` onto the packaged consensus Vα/Vβ). The topology
is ANARCI/IMGT-renumbered once via `kinapse.structures.TCR`, then the two
per-frame superpositions run on in-memory `AtomArray`s — reproducing `process()`
to < 1e-2° at ~30 ms/frame, so full multi-µs trajectories are tractable.

## Run

```bash
conda activate kinapse            # micromamba activate kinapse
python angle_exploration.py                        # all systems, ~4000 frames each
python angle_exploration.py --systems 1KGC 8YJ3    # a subset
python angle_exploration.py --full                 # every frame (slow)
```

Needs `gemmi` in the env (a hard dependency of `anarcii`, used for numbering).
The script re-execs itself with `PYTHONNOUSERSITE=1` to avoid an ABI-broken
user-site `h5py` that otherwise crashes MDAnalysis.

## Outputs (`results/`, git-ignored)

* `per_system/<ID>_angles.csv` — per-frame parameters + `interdom_min` + `valid`.
* `per_system/<ID>_timeseries.png` — the six parameters vs. time.
* `per_system/<ID>_corner.png` — explored 6-D subspace, coloured by time.
* `summary/system_summary.csv` — per-system mean/std/range + pairing/QC metadata.
* `summary/param_distributions.png` — per-parameter ridgelines across systems.
* `summary/exploration_spread.png` — std per parameter per system (exploration breadth).
* `summary/pca_landscape.png` — shared 6-D→2-D PCA landscape + per-system occupancy.

## Data caveats handled by the script

* **No periodic box** is stored in these merged trajectories, and a contiguous
  block of frames (appended from a second source) has the domains split/displaced.
  Such frames are detected by the Vα–Vβ interface Cα distance and excluded from
  statistics (kept in the CSV as `valid=False`). Without this, `dc`/`BA` spread is
  inflated ~30× by non-physical frames.
* **ANARCI TRAV/TRDV mistyping** (an α chain typed δ) breaks automatic α/β pairing;
  the script retries with coerced chain types, disambiguating orientation by
  superposition fit quality.
