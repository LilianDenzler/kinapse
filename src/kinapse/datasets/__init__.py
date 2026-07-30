"""datasets — TCR-pMHC dataset loaders + leakage-aware splits (data tier).

purpose: fetch/normalise standard datasets and build honest train/test splits.
extra:   ``[datasets]``
status:  scaffold — the catalog is declared; loaders are not implemented yet.

Planned loaders (see ``docs/architecture_todo.md``):
  structures/affinity: ATLAS (ΔG+mutants), STCRDab, TCR3d, SKEMPI (ΔΔG, ΔH/ΔS)
  specificity:         VDJdb, McPAS, IEDB, 10x dextramer (with negatives)
  cross-reactivity:    DMS / yeast-display maps (Birnbaum, Riley)
  dynamics:            FTCRDab / ITsFlexible (CDR flexibility)
Also planned: leakage-aware splitting (by-epitope, by-MHC) + principled negatives.
"""
from __future__ import annotations

CATALOG = {
    "atlas": "TCR-pMHC affinities (ΔG) linked to structures; WT + mutants",
    "stcrdab": "Structural TCR database (OPIG)",
    "tcr3d": "TCR structural repertoire + docking benchmark",
    "skempi": "ΔΔG upon mutation for PPIs (incl. ΔH/ΔS); standard ddG set",
    "vdjdb": "Curated TCR→epitope specificity (sequence)",
    "iedb": "Immune Epitope Database (largest TCR-epitope source)",
    "tenx": "10x dextramer single-cell paired TCR + pMHC (has negatives; noisy)",
    "dms": "Deep-mutational-scanning / yeast-display cross-reactivity maps",
}


def load(name, **kwargs):
    """Load a dataset by catalog name. (Scaffold — not implemented yet.)"""
    if name not in CATALOG:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(sorted(CATALOG))}")
    raise NotImplementedError(f"dataset loader {name!r} is scaffolded but not implemented yet")


def split(dataset, by="epitope", **kwargs):
    """Build a leakage-aware split (by epitope / MHC …). (Scaffold.)"""
    raise NotImplementedError("leakage-aware splitting is scaffolded but not implemented yet")


__all__ = ["CATALOG", "load", "split"]
