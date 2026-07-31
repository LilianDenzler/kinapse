"""Interface characterization for a loaded complex — descriptive, not a score.

Two engines, same :class:`InterfaceResult`:

* **native** (default) — Biopython + optional freesasa. Interface residues, atom
  contacts, buried surface area. No external tool / network / GPU; matches the
  lightweight core.
* **stcrpy** — runs OPIG **STCRpy** (2025) *externally* (isolated subprocess) for
  TCR-aware, PLIP-typed interactions (H-bonds, salt bridges, …) with correct
  CDR↔peptide/MHC partitioning, plus its docking-geometry angles. STCRpy pulls
  heavy deps (ANARCI models, PLIP, OpenBabel), so it is **never imported into
  kinapse's env** — you install it in its own env and point kinapse at that
  interpreter (``KINAPSE_STCRPY_PYTHON``), or install it into the active env.

Still declared as optional backends (not wired): ``pisa`` (ΔG_diss/assembly, CCP4
binary/PDBePISA API), ``plip`` (direct PLIP). This is interface *characterization*
(what the interface looks like), distinct from ``kinapse.scoring`` (ranking/energy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_BACKEND_HINTS = {
    "pisa": "PISA backend not wired yet — install CCP4 (command-line `pisa`) or query PDBePISA; "
            "gives ΔG_int / ΔG_dissociation / assembly stability.",
    "plip": "PLIP backend not wired yet — use method='stcrpy' (which runs PLIP with TCR-aware "
            "partitioning), or `pip install plip` and wire it directly.",
}

_STCRPY_HINT = (
    "STCRpy not found. It needs its own env (heavy deps: ANARCI models, PLIP, OpenBabel):\n"
    "  pip install stcrpy && ANARCI --build_models && pip install plip\n"
    "Then either install it in the active env, or point kinapse at that env:\n"
    "  export KINAPSE_STCRPY_PYTHON=/path/to/env/bin/python"
)


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
    bsa: Optional[float] = None                     # total buried surface area (Å²), if computed
    annotations: Dict[str, str] = field(default_factory=dict)  # "chain/resnum" -> label
    interactions: Optional[List[Dict[str, Any]]] = None        # typed interactions (stcrpy/PLIP)
    geometry: Optional[Dict[str, Any]] = None                  # docking-geometry angles (stcrpy)

    @property
    def n_interface_residues(self) -> int:
        return sum(len(v) for v in self.interface_residues.values())

    def summary(self) -> str:
        per = ", ".join(f"{c}:{len(v)}" for c, v in sorted(self.interface_residues.items()))
        bsa = f", BSA≈{self.bsa} Å²" if self.bsa is not None else ""
        extra = f", {len(self.interactions)} typed interactions" if self.interactions else ""
        return (f"interface[{self.method}] rec={self.rec_chains} lig={self.lig_chains}: "
                f"{self.n_interface_residues} residues ({per}), "
                f"{self.n_atom_contacts} atom contacts @ {self.cutoff} Å{bsa}{extra}")


def analyze_interface(structure, rec_chains: Sequence[str], lig_chains: Sequence[str],
                      cutoff: float = 5.0, method: str = "native",
                      with_bsa: bool = True, pdb_path: Optional[str] = None) -> InterfaceResult:
    """Characterize the interface between ``rec_chains`` and ``lig_chains``.

    Args:
        structure: a Biopython ``Model``/``Structure`` (or a PDB path).
        rec_chains, lig_chains: chain ids on each side of the interface.
        cutoff: heavy-atom contact distance in Å (native engine).
        method: ``"native"`` (default) or ``"stcrpy"`` (external), or the declared
            ``"pisa"`` / ``"plip"`` backends.
        with_bsa: compute buried surface area if ``freesasa`` is installed (native).
        pdb_path: the complex PDB path — required by the ``stcrpy`` backend (it
            re-annotates chains itself). Falls back to ``structure`` if that is a path.
    """
    if method == "native":
        return _native_interface(structure, rec_chains, lig_chains, cutoff, with_bsa)

    if method == "stcrpy":
        # native base (residues/contacts/BSA) when a structure is available ...
        if structure is not None:
            res = _native_interface(structure, rec_chains, lig_chains, cutoff, with_bsa)
        else:
            res = InterfaceResult(list(rec_chains), list(lig_chains), cutoff, "stcrpy", 0)
        res.method = "stcrpy"
        # ... then enrich with STCRpy's typed interactions + geometry (external run)
        path = pdb_path or (structure if isinstance(structure, str) else None)
        if path is None:
            raise ValueError("the stcrpy backend needs the complex PDB path (pass pdb_path=...)")
        out = _run_stcrpy(str(path))
        res.interactions = out.get("interactions")
        res.geometry = out.get("angles")
        if isinstance(out.get("annotations"), dict):
            res.annotations.update(out["annotations"])
        return res

    raise NotImplementedError(_BACKEND_HINTS.get(method, f"unknown interface backend {method!r}"))


# ---------------------------------------------------------------- native engine
def _native_interface(structure, rec_chains, lig_chains, cutoff, with_bsa) -> InterfaceResult:
    from Bio.PDB import NeighborSearch

    if isinstance(structure, str):
        from Bio.PDB import PDBParser
        structure = PDBParser(QUIET=True).get_structure("complex", structure)[0]

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

    result = InterfaceResult(list(rec_chains), list(lig_chains), cutoff, "native", 0)
    if not rec_atoms or not lig_atoms:
        return result

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
    result.n_atom_contacts = contacts
    result.interface_residues = by_chain
    if with_bsa:
        result.bsa = _buried_surface_area(structure, rec_chains, lig_chains)
    return result


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


# ---------------------------------------------------------------- stcrpy engine (external)
def _stcrpy_python() -> Optional[List[str]]:
    """Locate a Python that has STCRpy. Prefers ``KINAPSE_STCRPY_PYTHON`` (an
    external env), else the active interpreter if stcrpy is importable there."""
    import importlib.util
    import os
    import sys

    env_py = os.environ.get("KINAPSE_STCRPY_PYTHON")
    if env_py:
        return [env_py]
    if importlib.util.find_spec("stcrpy") is not None:
        return [sys.executable]
    return None


def _run_stcrpy(pdb_path: str, timeout: float = 600.0) -> Dict[str, Any]:
    """Run the bundled STCRpy runner script in its (external) interpreter."""
    import json
    import subprocess

    py = _stcrpy_python()
    if py is None:
        raise ImportError(_STCRPY_HINT)
    runner = str(Path(__file__).with_name("_stcrpy_runner.py"))
    try:
        proc = subprocess.run(py + [runner], input=json.dumps({"pdb": pdb_path}),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"stcrpy runner timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise RuntimeError(f"stcrpy runner failed:\n{(proc.stderr or '').strip()[-800:]}")
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"stcrpy runner did not emit JSON: {proc.stdout[:300]!r}") from e
    if out.get("status") == "error":
        raise RuntimeError(f"stcrpy runner error: {out.get('error')}")
    return out


__all__ = ["InterfaceResult", "analyze_interface"]
