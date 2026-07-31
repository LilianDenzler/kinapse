"""Interface characterization for a loaded complex — descriptive, not a score.

The lightweight **native** engine (Biopython + optional freesasa) computes, from a
single structure: interface residues, atom contacts, and buried surface area (BSA).
It needs no external binary or network — matching kinapse's lightweight core.

Richer backends are optional and heavier (declared, wired on demand):
  * ``pisa``   — CCP4/PDBePISA: ΔG_int, ΔG_dissociation, assembly stability (heavy binary / web API)
  * ``plip``   — PLIP 2025: typed interactions (H-bonds, salt bridges, π-stacking …); needs OpenBabel

For OPIG STCRpy's TCR-aware interaction profiling + geometry, use the separate
:mod:`kinapse.structure_analysis` runner (``stcrpy``) — it is a standalone external
model, not an ``interface()`` backend.

This is *interface characterization* (what the interface looks like), distinct from
``kinapse.scoring`` (ranking/energy) — see docs/scoring.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_BACKEND_HINTS = {
    "pisa": "PISA backend not wired yet — install CCP4 (command-line `pisa`) or query PDBePISA; "
            "gives ΔG_int / ΔG_dissociation / assembly stability.",
    "plip": "PLIP backend not wired yet — `pip install plip` (needs OpenBabel + PyMOL); "
            "gives typed interactions (H-bonds, salt bridges, π-stacking, …). For STCRpy's "
            "TCR-aware profiling use kinapse.structure_analysis instead.",
}


@dataclass
class InterfaceResult:
    """Interface characterization between a receptor and a ligand chain set."""
    rec_chains: List[str]
    lig_chains: List[str]
    cutoff: float
    method: str
    n_atom_contacts: int
    # chain id -> sorted [(residue number, residue name)]
    interface_residues: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)
    bsa: Optional[float] = None                 # total buried surface area (Å²), if computed
    annotations: Dict[str, str] = field(default_factory=dict)  # "chain/resnum" -> label (e.g. TCR region)

    @property
    def n_interface_residues(self) -> int:
        return sum(len(v) for v in self.interface_residues.values())

    def summary(self) -> str:
        per = ", ".join(f"{c}:{len(v)}" for c, v in sorted(self.interface_residues.items()))
        bsa = f", BSA≈{self.bsa} Å²" if self.bsa is not None else ""
        return (f"interface[{self.method}] rec={self.rec_chains} lig={self.lig_chains}: "
                f"{self.n_interface_residues} residues ({per}), "
                f"{self.n_atom_contacts} atom contacts @ {self.cutoff} Å{bsa}")


def analyze_interface(structure, rec_chains: Sequence[str], lig_chains: Sequence[str],
                      cutoff: float = 5.0, method: str = "native",
                      with_bsa: bool = True) -> InterfaceResult:
    """Characterize the interface between ``rec_chains`` and ``lig_chains``.

    Args:
        structure: a Biopython ``Model``/``Structure`` (e.g. ``TCR.original_structure``).
        rec_chains, lig_chains: chain ids on each side of the interface.
        cutoff: heavy-atom contact distance in Å.
        method: ``"native"`` (default; Biopython + optional freesasa) or an optional
            backend (``pisa`` / ``plip`` — see module docstring). For STCRpy, use
            the separate :mod:`kinapse.structure_analysis` runner.
        with_bsa: compute buried surface area if ``freesasa`` is installed.
    """
    if method != "native":
        raise NotImplementedError(_BACKEND_HINTS.get(method, f"unknown interface backend {method!r}"))

    from Bio.PDB import NeighborSearch

    rec_set, lig_set = set(rec_chains), set(lig_chains)
    rec_atoms, lig_atoms = [], []
    for chain in structure:
        which = rec_atoms if chain.id in rec_set else lig_atoms if chain.id in lig_set else None
        if which is None:
            continue
        for res in chain:
            if res.id[0] != " ":            # skip hetero/water
                continue
            which.extend(res.get_atoms())

    res = InterfaceResult(list(rec_chains), list(lig_chains), cutoff, method, 0)
    if not rec_atoms or not lig_atoms:
        return res

    ns = NeighborSearch(rec_atoms)
    iface: Dict[Tuple[str, int], str] = {}
    contacts = 0
    for a in lig_atoms:
        near = ns.search(a.coord, cutoff)
        if not near:
            continue
        contacts += len(near)
        lres = a.get_parent()
        iface[(lres.get_parent().id, lres.id[1])] = lres.resname
        for r in near:
            rres = r.get_parent()
            iface[(rres.get_parent().id, rres.id[1])] = rres.resname

    by_chain: Dict[str, List[Tuple[int, str]]] = {}
    for (cid, num), name in sorted(iface.items()):
        by_chain.setdefault(cid, []).append((num, name))
    res.n_atom_contacts = contacts
    res.interface_residues = by_chain
    if with_bsa:
        res.bsa = _buried_surface_area(structure, rec_chains, lig_chains)
    return res


def _buried_surface_area(structure, rec_chains, lig_chains) -> Optional[float]:
    """Total BSA = SASA(rec alone) + SASA(lig alone) − SASA(complex), via freesasa.
    Returns None if freesasa is not installed."""
    try:
        import os
        import tempfile
        import freesasa
        from Bio.PDB import PDBIO, Select
    except Exception:
        return None

    class _Sel(Select):
        def __init__(self, chains):
            self.chains = set(chains)

        def accept_chain(self, chain):
            return chain.id in self.chains

    def sasa(chains) -> float:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        try:
            io = PDBIO()
            io.set_structure(structure)
            io.save(tmp.name, _Sel(chains))
            freesasa.setVerbosity(freesasa.silent)
            return freesasa.calc(freesasa.Structure(tmp.name)).totalArea()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    try:
        both = list(rec_chains) + list(lig_chains)
        return round(sasa(rec_chains) + sasa(lig_chains) - sasa(both), 1)
    except Exception:
        return None


__all__ = ["InterfaceResult", "analyze_interface"]
