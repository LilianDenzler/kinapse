"""OpenMM energy-minimisation + single-trajectory MM/GBSA, run in an external env.

OpenMM + PDBFixer are heavy and often conflict with kinapse's own stack, so — like
the STCRpy / tFold runners — kinapse never imports them; it shells out to an
interpreter that has them, behind a JSON contract (:mod:`._openmm_runner`).

Point kinapse at that env with ``KINAPSE_OPENMM_PYTHON`` (an interpreter with
``openmm`` + ``pdbfixer``); otherwise the active interpreter is used if OpenMM is
importable there. On this machine the ``gromacs_smd`` conda env has both
(openmm 8.2 + pdbfixer), and also ``gmx`` for :mod:`kinapse.energetics.solvation`.

MM/GBSA note: this is amber14 + implicit solvent (GB polar + ACE nonpolar), a
single-trajectory ΔG_bind = E(complex) − E(rec) − E(lig) with NO entropy term —
distinct from the SASA-only nonpolar model in :mod:`kinapse.energetics.solvation`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_HINT = (
    "OpenMM/PDBFixer not found. They need an environment with `openmm` + `pdbfixer`.\n"
    "Point kinapse at it:  export KINAPSE_OPENMM_PYTHON=/path/to/env/bin/python\n"
    "(on this host, /home/lilian/.conda/envs/gromacs_smd/bin/python has both)."
)
_RUNNER = str(Path(__file__).with_name("_openmm_runner.py"))


def _python() -> Optional[List[str]]:
    env_py = os.environ.get("KINAPSE_OPENMM_PYTHON")
    if env_py:
        return [env_py]
    if importlib.util.find_spec("openmm") is not None:
        return [sys.executable]
    return None


def available() -> bool:
    return _python() is not None


def minimize_many(jobs: List[Dict[str, Any]], *, max_iter: int = 0, solvent: str = "gbn2",
                  platform: str = "auto", add_hydrogens: bool = True,
                  remove_heterogens: bool = True, timeout: Optional[float] = None) -> Dict[str, Any]:
    """Minimise a batch of structures in one external process.

    Each job: ``{"pdb", "out_pdb", "rec_chains"?, "lig_chains"?, "mmgbsa"?}``.
    ``max_iter=0`` runs OpenMM's minimiser to convergence. Returns
    ``{"status", "results": [...]}`` (per-job dicts with minimised energies and,
    when ``mmgbsa`` + chains are given, ``dG_bind_mmgb_kcal``). Never raises on a
    per-job failure; raises :class:`ImportError` only if no OpenMM env is found.
    """
    py = _python()
    if py is None:
        raise ImportError(_HINT)
    if timeout is None:
        timeout = 300.0 + 300.0 * max(1, len(jobs))
    payload = {"jobs": jobs, "max_iter": max_iter, "solvent": solvent, "platform": platform,
               "add_hydrogens": add_hydrogens, "remove_heterogens": remove_heterogens}
    try:
        proc = subprocess.run(py + [_RUNNER], input=json.dumps(payload),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"openmm exceeded {timeout:.0f}s", "results": []}
    if proc.returncode != 0:
        return {"status": "error", "error": (proc.stderr or "").strip()[-800:], "results": []}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "error": f"openmm runner did not emit JSON: {proc.stdout[:300]!r}",
                "stderr": (proc.stderr or "").strip()[-500:], "results": []}


def minimize(pdb: str, out_pdb: str, *, rec_chains: Optional[Sequence[str]] = None,
             lig_chains: Optional[Sequence[str]] = None, mmgbsa: bool = False,
             **kwargs) -> Dict[str, Any]:
    """Minimise one structure (optionally MM/GBSA). Returns that job's result dict."""
    job = {"pdb": str(pdb), "out_pdb": str(out_pdb), "mmgbsa": mmgbsa}
    if rec_chains is not None:
        job["rec_chains"] = list(rec_chains)
    if lig_chains is not None:
        job["lig_chains"] = list(lig_chains)
    res = minimize_many([job], **kwargs)
    if res.get("results"):
        return res["results"][0]
    return {"status": res.get("status", "error"), "error": res.get("error"),
            "pdb": str(pdb), "out_pdb": str(out_pdb)}


__all__ = ["available", "minimize", "minimize_many"]
