"""Energetic analyses of TCR–pMHC structures.

``solvation`` — GROMACS SASA-based *nonpolar* solvation-energy change on binding
(γ·SASA + b), from a static complex split into unbound TCR + unbound pMHC.
``minimize`` — OpenMM energy minimisation + single-trajectory MM/GBSA ΔG_bind
(amber14 + implicit GB polar + ACE nonpolar), run in an external OpenMM env.
"""
from kinapse.energetics.solvation import (
    BindingSolvation,
    StructureSolvation,
    GmxError,
    binding_solvation,
    split_tcr_pmhc,
    sasa_gmx,
    find_gmx,
    gmx_version,
)
from kinapse.energetics.minimize import minimize, minimize_many

__all__ = [
    "BindingSolvation",
    "StructureSolvation",
    "GmxError",
    "binding_solvation",
    "split_tcr_pmhc",
    "sasa_gmx",
    "find_gmx",
    "gmx_version",
    "minimize",
    "minimize_many",
]
