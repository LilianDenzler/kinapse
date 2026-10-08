"""GROMACS solvation energy of TCR–pMHC binding from a *static* structure.

Given a single TCR–pMHC complex PDB, this "unbinds" it *in silico* — the TCR and
the pMHC are treated as two separate molecules — and reports how the solvation
free energy changes on binding::

    ΔG_solv(bind) = G_solv(complex) − G_solv(TCR) − G_solv(pMHC)

``G_solv`` per structure is the **nonpolar / hydrophobic** solvation term derived
from the solvent-accessible surface area (SASA) that GROMACS' ``gmx sasa``
measures::

    G_solv = γ · SASA + b          (Sitkoff/Eisenberg nonpolar model)

This is the pure-GROMACS, static-structure analogue of the MM-PBSA *nonpolar*
term. It captures the hydrophobic burial on binding — the interface area that
becomes solvent-excluded — but **not** the polar/electrostatic desolvation
penalty. For the full solvation free energy add a Poisson–Boltzmann backend
(APBS / ``gmx_MMPBSA``); ``binding_solvation`` accepts a ``polar_backend`` hook
for exactly that. The physical, model-free quantity — the **buried SASA**
(``ΔSASA = SASA_TCR + SASA_pMHC − SASA_complex``) — is always reported too.

The TCR/pMHC split uses kinapse's interface pairing: the two TCR chains are found
via :class:`kinapse.structures.tcr.TCR`; everything else in the file is pMHC
(MHC heavy chain + β2m + peptide).

No MD is involved — one structure in, one solvation budget out.

Example
-------
>>> from kinapse.energetics.solvation import binding_solvation
>>> res = binding_solvation("1mi5_complex.pdb")            # doctest: +SKIP
>>> res["dG_solv_bind_kcal"], res["buried_sasa_A2"]        # doctest: +SKIP
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Nonpolar solvation model  (G_np = γ·SASA + b)
# --------------------------------------------------------------------------- #
# Sitkoff, Sharp & Honig (1994) / Eisenberg & McLachlan (1986): the standard
# MM-PBSA nonpolar surface term. Units are kcal/mol and Å² (SASA is converted
# from the nm² that gmx reports). Both are overridable per call.
GAMMA_KCAL_PER_A2 = 0.00542   # surface tension            [kcal·mol⁻¹·Å⁻²]
B_KCAL = 0.92                 # constant offset per molecule [kcal·mol⁻¹]
KJ_PER_KCAL = 4.184
A2_PER_NM2 = 100.0            # gmx sasa reports nm²; 1 nm² = 100 Å²


class GmxError(RuntimeError):
    """Raised when a GROMACS invocation is missing or fails."""


# --------------------------------------------------------------------------- #
# Data returned per structure / per binding calculation
# --------------------------------------------------------------------------- #
@dataclass
class StructureSolvation:
    """SASA + nonpolar solvation for one structure (complex, TCR or pMHC)."""
    name: str
    chains: List[str]
    n_atoms: int
    sasa_A2: float                      # total SASA
    g_solv_kcal: float                  # γ·SASA + b
    pdb: str = ""

    @property
    def g_solv_kJ(self) -> float:
        return self.g_solv_kcal * KJ_PER_KCAL


@dataclass
class BindingSolvation:
    """The bound-vs-unbound solvation budget for one TCR–pMHC structure."""
    complex: StructureSolvation
    tcr: StructureSolvation
    pmhc: StructureSolvation
    gamma_kcal_per_A2: float
    b_kcal: float
    polar: Optional[Dict[str, float]] = None   # filled if a polar backend is used
    meta: Dict[str, str] = field(default_factory=dict)

    # -- nonpolar (SASA) terms -------------------------------------------------
    @property
    def buried_sasa_A2(self) -> float:
        """SASA lost on binding = the interface area (always ≥ 0 physically)."""
        return self.tcr.sasa_A2 + self.pmhc.sasa_A2 - self.complex.sasa_A2

    @property
    def dG_solv_np_kcal(self) -> float:
        """Nonpolar solvation change on binding: G(complex) − G(TCR) − G(pMHC)."""
        return self.complex.g_solv_kcal - self.tcr.g_solv_kcal - self.pmhc.g_solv_kcal

    # -- total (nonpolar + optional polar) ------------------------------------
    @property
    def dG_solv_bind_kcal(self) -> float:
        dg = self.dG_solv_np_kcal
        if self.polar is not None:
            dg += self.polar.get("dG_solv_polar_kcal", 0.0)
        return dg

    def to_row(self) -> Dict[str, object]:
        row = {
            "complex_pdb": self.complex.pdb,
            "tcr_chains": "".join(self.tcr.chains),
            "pmhc_chains": "".join(self.pmhc.chains),
            "sasa_complex_A2": self.complex.sasa_A2,
            "sasa_tcr_A2": self.tcr.sasa_A2,
            "sasa_pmhc_A2": self.pmhc.sasa_A2,
            "buried_sasa_A2": self.buried_sasa_A2,
            "g_solv_complex_kcal": self.complex.g_solv_kcal,
            "g_solv_tcr_kcal": self.tcr.g_solv_kcal,
            "g_solv_pmhc_kcal": self.pmhc.g_solv_kcal,
            "dG_solv_np_kcal": self.dG_solv_np_kcal,
            "dG_solv_bind_kcal": self.dG_solv_bind_kcal,
            "gamma_kcal_per_A2": self.gamma_kcal_per_A2,
            "b_kcal": self.b_kcal,
        }
        if self.polar is not None:
            row.update({f"polar_{k}": v for k, v in self.polar.items()})
        row.update(self.meta)
        return row


# --------------------------------------------------------------------------- #
# GROMACS plumbing
# --------------------------------------------------------------------------- #
def find_gmx(gmx_bin: str = "gmx") -> str:
    """Return the resolved gmx executable path, or raise a clear error."""
    exe = shutil.which(gmx_bin)
    if exe is None:
        raise GmxError(
            f"GROMACS executable {gmx_bin!r} not found on PATH. Install GROMACS "
            f"or pass gmx_bin=/path/to/gmx.")
    return exe


def gmx_version(gmx_bin: str = "gmx") -> str:
    exe = find_gmx(gmx_bin)
    out = subprocess.run([exe, "--version"], capture_output=True, text=True)
    for line in (out.stdout or "").splitlines():
        if "GROMACS version" in line:
            return line.split(":", 1)[-1].strip()
    return "unknown"


def _run(cmd: Sequence[str], cwd: Path, dry_run: bool = False) -> subprocess.CompletedProcess:
    if dry_run:
        print("＄", " ".join(str(c) for c in cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise GmxError(
            f"command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc


def _parse_xvg_area(path: Path, col: int = 1) -> float:
    """Return the area (nm²) from the given column of a gmx ``-o`` .xvg file.

    Static structure = a single data row; we take the last one to be safe.
    Column 0 is time; column 1 is the first (total) area series.
    """
    last = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#@":
                continue
            last = line.split()
    if last is None:
        raise GmxError(f"no data rows in {path}")
    return float(last[col])


def sasa_gmx(pdb: Path, work: Path, *, gmx_bin: str = "gmx",
             probe_nm: float = 0.14, surface: str = "Protein",
             ndots: int = 24, dry_run: bool = False) -> float:
    """Total SASA (Å²) of a structure via ``gmx sasa`` run directly on the PDB.

    Runs on the PDB as its own topology (VdW radii from GROMACS' ``vdwradii.dat``)
    — robust for modelled / non-force-field-ready complexes. ``surface`` is a gmx
    selection (default ``Protein``; use ``all`` to include HETATM kept in the file).
    """
    area = work / f"{pdb.stem}_area.xvg"
    cmd = [gmx_bin, "sasa", "-s", pdb, "-f", pdb,
           "-o", area, "-probe", f"{probe_nm}", "-ndots", str(ndots),
           "-surface", surface, "-output", surface]
    _run(cmd, work, dry_run=dry_run)
    if dry_run:
        return float("nan")
    return _parse_xvg_area(area) * A2_PER_NM2


# --------------------------------------------------------------------------- #
# TCR / pMHC split (kinapse interface pairing)
# --------------------------------------------------------------------------- #
def split_tcr_pmhc(pdb: str, out_dir: Path, *, legacy_anarci: bool = True,
                   keep_het: bool = False) -> Dict[str, object]:
    """Write complex / TCR-only / pMHC-only PDBs and return their chain sets.

    The TCR chains come from :class:`kinapse.structures.tcr.TCR` (ANARCI numbering
    + interface pairing); pMHC is every other polymer chain in the file.
    """
    from Bio.PDB import PDBParser, PDBIO, Select
    from kinapse.structures.tcr import TCR

    out_dir.mkdir(parents=True, exist_ok=True)
    tcr = TCR(input_pdb=pdb, legacy_anarci=legacy_anarci)
    if not getattr(tcr, "pairs", None):
        raise ValueError(f"no TCR pair found in {pdb}; cannot split TCR from pMHC")
    tcr_chains: List[str] = []
    for p in tcr.pairs:
        tcr_chains += [p.alpha_chain_id, p.beta_chain_id]
    tcr_chains = [c for c in dict.fromkeys(tcr_chains)]        # de-dup, keep order

    model = PDBParser(QUIET=True).get_structure("cplx", pdb)[0]
    all_chains = [c.id for c in model]
    pmhc_chains = [c for c in all_chains if c not in set(tcr_chains)]
    if not pmhc_chains:
        raise ValueError(
            f"{pdb}: every chain typed as TCR — no pMHC to unbind (chains={all_chains})")

    class _Sel(Select):
        def __init__(self, chains, keep_het):
            self.chains, self.keep_het = set(chains), keep_het

        def accept_chain(self, chain):
            return chain.id in self.chains

        def accept_residue(self, residue):
            return True if self.keep_het else residue.id[0] == " "

    io = PDBIO()
    io.set_structure(model)
    files = {}
    for tag, chains in (("complex", tcr_chains + pmhc_chains),
                        ("tcr", tcr_chains), ("pmhc", pmhc_chains)):
        path = out_dir / f"{tag}.pdb"
        io.save(str(path), _Sel(chains, keep_het))
        files[tag] = str(path)

    return {"tcr_chains": tcr_chains, "pmhc_chains": pmhc_chains,
            "all_chains": all_chains, "files": files}


def _count_atoms(pdb: str) -> int:
    n = 0
    with open(pdb) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                n += 1
    return n


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _solvation_of(name: str, pdb: str, chains: List[str], work: Path, *,
                  gmx_bin: str, gamma: float, b: float, surface: str,
                  dry_run: bool) -> StructureSolvation:
    sasa = sasa_gmx(Path(pdb), work, gmx_bin=gmx_bin, surface=surface, dry_run=dry_run)
    g = gamma * sasa + b if np.isfinite(sasa) else float("nan")
    return StructureSolvation(name=name, chains=chains, n_atoms=_count_atoms(pdb),
                              sasa_A2=sasa, g_solv_kcal=g, pdb=pdb)


def binding_solvation(pdb: str, *, gmx_bin: str = "gmx",
                      gamma: float = GAMMA_KCAL_PER_A2, b: float = B_KCAL,
                      surface: str = "Protein", keep_het: bool = False,
                      workdir: Optional[str] = None, keep_workdir: bool = False,
                      legacy_anarci: bool = True,
                      polar_backend: Optional[Callable[[Dict[str, str]], Dict[str, float]]] = None,
                      dry_run: bool = False) -> BindingSolvation:
    """Solvation change on TCR–pMHC binding for one static complex structure.

    Parameters
    ----------
    pdb : path to a TCR–pMHC complex.
    gamma, b : nonpolar model  G_np = γ·SASA + b  (kcal/mol, Å²).
    surface : gmx selection for SASA (``Protein`` by default).
    polar_backend : optional callable ``files -> {"dG_solv_polar_kcal": .., ...}``
        given the ``{"complex","tcr","pmhc"}`` PDB paths, to add a PB/GB polar
        term (e.g. an APBS or gmx_MMPBSA wrapper). ``files`` also carries the
        chain sets. Nonpolar-only if omitted.
    dry_run : print the gmx commands instead of running them.

    Returns
    -------
    BindingSolvation with per-structure SASA/G_solv and the binding differences.
    """
    if not dry_run:
        find_gmx(gmx_bin)                       # fail early with a clear message

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="kinapse_solv_"))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        split = split_tcr_pmhc(pdb, tmp, legacy_anarci=legacy_anarci, keep_het=keep_het)
        files = split["files"]
        comp = _solvation_of("complex", files["complex"],
                             split["tcr_chains"] + split["pmhc_chains"], tmp,
                             gmx_bin=gmx_bin, gamma=gamma, b=b, surface=surface, dry_run=dry_run)
        tcr = _solvation_of("tcr", files["tcr"], split["tcr_chains"], tmp,
                            gmx_bin=gmx_bin, gamma=gamma, b=b, surface=surface, dry_run=dry_run)
        pmhc = _solvation_of("pmhc", files["pmhc"], split["pmhc_chains"], tmp,
                             gmx_bin=gmx_bin, gamma=gamma, b=b, surface=surface, dry_run=dry_run)

        polar = None
        if polar_backend is not None and not dry_run:
            polar = polar_backend({**files,
                                   "tcr_chains": split["tcr_chains"],
                                   "pmhc_chains": split["pmhc_chains"]})

        meta = {"pdb": str(pdb), "gmx_version": ("dry-run" if dry_run else gmx_version(gmx_bin))}
        return BindingSolvation(complex=comp, tcr=tcr, pmhc=pmhc,
                                gamma_kcal_per_A2=gamma, b_kcal=b, polar=polar, meta=meta)
    finally:
        if not keep_workdir and workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser(
        description="GROMACS solvation-energy change on TCR–pMHC binding "
                    "(static structure; γ·SASA nonpolar model).")
    ap.add_argument("pdb", nargs="+", help="TCR–pMHC complex PDB(s)")
    ap.add_argument("--gmx", default="gmx", help="gmx executable (default: gmx)")
    ap.add_argument("--gamma", type=float, default=GAMMA_KCAL_PER_A2,
                    help="nonpolar surface tension [kcal/mol/Å²]")
    ap.add_argument("--b", type=float, default=B_KCAL, help="nonpolar offset [kcal/mol]")
    ap.add_argument("--surface", default="Protein", help="gmx SASA selection")
    ap.add_argument("--keep-het", action="store_true", help="keep HETATM in the split PDBs")
    ap.add_argument("--csv", default=None, help="write results to this CSV")
    ap.add_argument("--dry-run", action="store_true", help="print gmx commands only")
    args = ap.parse_args(argv)

    rows = []
    for pdb in args.pdb:
        try:
            res = binding_solvation(pdb, gmx_bin=args.gmx, gamma=args.gamma, b=args.b,
                                    surface=args.surface, keep_het=args.keep_het,
                                    dry_run=args.dry_run)
        except Exception as e:
            print(f"[ERROR] {pdb}: {e}")
            continue
        if args.dry_run:
            continue
        rows.append(res.to_row())
        print(f"\n{Path(pdb).name}: TCR={''.join(res.tcr.chains)} "
              f"pMHC={''.join(res.pmhc.chains)}")
        print(f"  SASA  complex/TCR/pMHC = {res.complex.sasa_A2:.0f} / "
              f"{res.tcr.sasa_A2:.0f} / {res.pmhc.sasa_A2:.0f} Å²   "
              f"buried = {res.buried_sasa_A2:.0f} Å²")
        print(f"  ΔG_solv(nonpolar) on binding = {res.dG_solv_np_kcal:+.2f} kcal/mol"
              + ("" if res.polar is None else
                 f"   (+polar {res.polar.get('dG_solv_polar_kcal', float('nan')):+.2f} "
                 f"→ total {res.dG_solv_bind_kcal:+.2f})"))

    if args.csv and rows:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\n[saved] {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
