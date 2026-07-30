"""kinapse — a modular toolkit for TCR / TCR-pMHC structure preparation,
sequence embedding, geometry, structural & MD dynamics analysis, and
conformational-ensemble generation.

The package is organised into five independently usable sub-packages:

* :mod:`kinapse.structures`  — load & prep TCR / TCR-pMHC structures (the basis)
* :mod:`kinapse.sequence_embedding`   — sequence embedders (MSA, Evoformer)
* :mod:`kinapse.geometry`    — TCR inter-domain docking geometry
* :mod:`kinapse.dynamics_analysis`    — structure & MD analysis + dynamics metrics
* :mod:`kinapse.conformer_generation`  — generative conformer sampling (DiG)

plus :mod:`kinapse.pipelines` (composable end-to-end workflows) and
:mod:`kinapse.config` (path/config resolution).

Heavy or optional dependencies (torch, PyMOL, MODELLER, deeptime, ...) are only
imported when the relevant feature is used, so ``import kinapse`` stays light.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Cheap, dependency-free constants are always available.
from .regions import CDR_FR_RANGES, VARIABLE_RANGE  # noqa: E402

__all__ = ["CDR_FR_RANGES", "VARIABLE_RANGE", "TCR", "load_tcr", "__version__"]


def __getattr__(name):  # PEP 562: lazy convenience re-exports
    if name in ("TCR", "load_tcr"):
        from . import structures
        return getattr(structures, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
