"""pMHC and TCR-pMHC complex loading — SCAFFOLD.

The original TCR_Metrics code operated purely on the TCR variable domains; it had
no MHC / peptide / pMHC handling. This module defines the *API surface* for
loading peptide-MHC and full TCR-pMHC complexes so the rest of kinapse
(analysis, geometry, generation) can be extended to them without a redesign.

Status
------
* :class:`PMHC` loads a structure and classifies chains with a **provisional**
  length/composition heuristic (peptide = short chain, MHC = long chain). This
  is a starting point, not validated science.
* Real capabilities still to implement are marked ``TODO`` and raise
  :class:`NotImplementedError`:
    - MHC class I vs class II detection (beta-2-microglobulin / alpha+beta chains)
    - IMGT/G-domain numbering of MHC
    - peptide register / anchor-residue identification
    - TCR-pMHC docking geometry & interface (CDR-loop <-> peptide/MHC contacts)

Heavy dependencies are imported lazily inside methods so importing this module
never fails on a machine without Biopython/mdtraj.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Rough guardrails for the provisional chain classifier.
_PEPTIDE_MAX_LEN = 30      # class-I epitopes are ~8-11; class-II ~13-25
_MHC_MIN_LEN = 90          # an MHC alpha1/alpha2 platform domain is ~180 aa


@dataclass
class PMHC:
    """A peptide-MHC structure.

    Args:
        input_pdb: path to a PDB containing (at least) an MHC and a peptide.
        traj_path: optional trajectory to attach (mirrors :class:`kinapse.structures.tcr.TCR`).
        mhc_class: ``"I"``, ``"II"`` or ``None`` (auto — see TODO below).

    Attributes populated by :meth:`load`:
        peptide_chains: chain ids classified as peptide.
        mhc_chains: chain ids classified as MHC.
        sequences: ``{chain_id: one_letter_sequence}``.
    """

    input_pdb: str
    traj_path: Optional[str] = None
    mhc_class: Optional[str] = None

    peptide_chains: List[str] = field(default_factory=list)
    mhc_chains: List[str] = field(default_factory=list)
    sequences: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.load()

    # -- loading -----------------------------------------------------------
    def load(self) -> "PMHC":
        """Parse chains and apply the provisional peptide/MHC classifier."""
        from Bio.PDB import PDBParser
        from Bio.SeqUtils import seq1

        model = PDBParser(QUIET=True).get_structure("pmhc", self.input_pdb)[0]
        for chain in model:
            resnames = [r.resname for r in chain if r.id[0] == " "]
            if not resnames:
                continue
            seq = "".join(seq1(r, custom_map={}) if len(r) == 1 else seq1(r) for r in resnames)
            cid = chain.id
            self.sequences[cid] = seq
            n = len(resnames)
            if n <= _PEPTIDE_MAX_LEN:
                self.peptide_chains.append(cid)
            elif n >= _MHC_MIN_LEN:
                self.mhc_chains.append(cid)
            # else: ambiguous (e.g. beta-2-microglobulin ~99 aa) -> left for
            # the class-aware classifier below.
        if self.mhc_class is None:
            self.mhc_class = self._guess_class()
        return self

    def _guess_class(self) -> Optional[str]:
        # TODO: distinguish class I (heavy chain + b2m) from class II
        # (alpha + beta) properly. Heuristic placeholder for now.
        if len(self.mhc_chains) >= 2:
            return "II"
        if self.mhc_chains:
            return "I"
        return None

    # -- not-yet-implemented science --------------------------------------
    def number_mhc(self):
        """IMGT/G-domain-number the MHC chain(s). TODO."""
        raise NotImplementedError("MHC numbering is not implemented yet (scaffold).")

    def peptide_anchors(self):
        """Identify peptide anchor residues / register. TODO."""
        raise NotImplementedError("Peptide anchor detection is not implemented yet (scaffold).")


@dataclass
class TCRpMHC:
    """A full TCR-pMHC complex: a TCR plus its peptide-MHC.

    Loads the TCR half with the existing, validated
    :class:`kinapse.structures.tcr.TCR` machinery and the pMHC half with
    :class:`PMHC`, then exposes hooks for interface / docking analysis (TODO).
    """

    input_pdb: str
    traj_path: Optional[str] = None
    tcr_kwargs: Dict = field(default_factory=dict)

    tcr: object = field(init=False, default=None)
    pmhc: Optional[PMHC] = field(init=False, default=None)

    def __post_init__(self):
        from .tcr import TCR
        # The TCR loader ignores non-TCR chains, so it works on a complex PDB.
        self.tcr = TCR(input_pdb=self.input_pdb, traj_path=self.traj_path, **self.tcr_kwargs)
        try:
            self.pmhc = PMHC(input_pdb=self.input_pdb, traj_path=self.traj_path)
        except Exception:
            self.pmhc = None  # complex may be TCR-only

    def interface(self, method: str = "native", cutoff: float = 5.0, with_bsa: bool = True):
        """Characterize the TCR↔pMHC interface (computed on demand, then cached).

        Receptor = the TCR α/β chains; ligand = every other polymer chain (pMHC).
        Returns an :class:`kinapse.structures.interface.InterfaceResult` with the
        interface residues (per chain), atom-contact count, and buried surface area
        (if ``freesasa`` is installed). ``method`` selects the engine: ``"native"``
        (default; no external tools) or the optional ``"pisa"`` / ``"plip"`` /
        ``"stcrpy"`` backends.
        """
        key = (method, cutoff, with_bsa)
        if getattr(self, "_iface_cache", None) is not None and getattr(self, "_iface_key", None) == key:
            return self._iface_cache
        from Bio.PDB import is_aa
        from .interface import analyze_interface

        rec: list = []
        for pair in self.tcr.pairs:
            for cid in (getattr(pair, "alpha_chain_id", None), getattr(pair, "beta_chain_id", None)):
                if cid and cid not in rec:
                    rec.append(cid)
        lig = [ch.id for ch in self.tcr.original_structure
               if ch.id not in rec and any(is_aa(r, standard=False) for r in ch)]
        if not lig:
            raise ValueError(
                "no non-TCR (pMHC) chains found — is this a TCR-pMHC complex? "
                "(the TCR-only case has no interface to characterize)."
            )
        res = analyze_interface(self.tcr.original_structure, rec, lig,
                                cutoff=cutoff, method=method, with_bsa=with_bsa)
        self._iface_cache, self._iface_key = res, key
        return res

    def interface_contacts(self, cutoff: float = 5.0):
        """Deprecated alias for :meth:`interface`."""
        return self.interface(cutoff=cutoff)

    def docking_geometry(self):
        """TCR-over-pMHC docking angle / crossing angle. TODO.

        (Note: :mod:`kinapse.geometry` currently computes the *intra*-TCR
        alpha/beta domain geometry, not the TCR-over-pMHC docking geometry.)
        """
        raise NotImplementedError("TCR-pMHC docking geometry is not implemented yet (scaffold).")


def load_pmhc(pdb: str, traj: Optional[str] = None, mhc_class: Optional[str] = None) -> PMHC:
    """Convenience constructor for :class:`PMHC`."""
    return PMHC(input_pdb=pdb, traj_path=traj, mhc_class=mhc_class)
