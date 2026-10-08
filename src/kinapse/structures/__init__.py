"""Structure preparation & loading — the basis for everything else.

Load a PDB, IMGT-renumber it (ANARCI/ANARCII), pair the alpha/beta (or
gamma/delta) chains by interface contacts, slice to CDR/framework regions, and
(optionally) attach an MD trajectory — via the :class:`TCR` object and its
:class:`TCRPairView` / :class:`TrajectoryView` companions.

pMHC / peptide / MHC handling is scaffolded in :mod:`kinapse.structures.pmhc`
(the current science is TCR-variable-domain only).

Symbols are lazily imported so ``import kinapse.structures`` never fails just
because an optional numbering/MD dependency is missing — the import only
happens when you actually touch the symbol.
"""
from __future__ import annotations

# name -> (submodule, attribute)
_LAZY = {
    "TCR": (".tcr", "TCR"),
    "TCRPairView": (".tcr", "TCRPairView"),
    "TrajectoryView": (".tcr", "TrajectoryView"),
    "rename_two_chains_exact_inplace": (".tcr", "rename_two_chains_exact_inplace"),
    "load_pdb": (".io", "load_pdb"),
    "write_pdb": (".io", "write_pdb"),
    "structure_to_atomarray": (".io", "structure_to_atomarray"),
    "atomarray_to_structure": (".io", "atomarray_to_structure"),
    "mdtraj_from_biopython_path": (".io", "mdtraj_from_biopython_path"),
    # pMHC scaffold
    "PMHC": (".pmhc", "PMHC"),
    "TCRpMHC": (".pmhc", "TCRpMHC"),
    "load_pmhc": (".pmhc", "load_pmhc"),
    # CD8 co-receptor grafting
    "add_cd8": (".cd8", "add_cd8"),
    "select_cd8_template": (".cd8", "select_cd8_template"),
    "load_templates": (".cd8", "load_templates"),
    "CD8Template": (".cd8", "CD8Template"),
    "CD8Result": (".cd8", "CD8Result"),
    # interface characterization
    "analyze_interface": (".interface", "analyze_interface"),
    "InterfaceResult": (".interface", "InterfaceResult"),
}

__all__ = list(_LAZY) + ["load_tcr"]


def load_tcr(pdb, traj=None, **kwargs):
    """Convenience constructor: build a :class:`TCR` from a PDB (+ optional traj).

    Args:
        pdb: path to a TCR PDB structure.
        traj: optional trajectory path (``.xtc``) to attach.
        **kwargs: forwarded to :class:`TCR` (``contact_cutoff``, ``min_contacts``,
            ``legacy_anarci``, ``manual_chain_types``).
    """
    from .tcr import TCR
    return TCR(input_pdb=pdb, traj_path=traj, **kwargs)


def __getattr__(name):  # PEP 562 lazy loading
    import importlib
    if name in _LAZY:
        submod, attr = _LAZY[name]
        module = importlib.import_module(submod, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
