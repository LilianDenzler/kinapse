"""Run OPIG **STCRpy** (2025) as a completely separate external model.

STCRpy annotates a TCR / TCR-pMHC structure (chains + CDRs), computes docking
geometry, profiles PLIP-typed interface interactions, and reports TCR-DockQ /
interface-RMSD. It pulls heavy deps (ANARCI models, PLIP, OpenBabel), so kinapse
**never imports it** — it runs in its own environment behind a JSON contract.

Point kinapse at that environment with ``KINAPSE_STCRPY_PYTHON`` (recommended);
otherwise the active interpreter is used if ``stcrpy`` is importable there; else a
clear install hint is raised. Registered as the ``stcrpy`` runner in the
``structure_analysis`` tier (:mod:`kinapse.structure_analysis`).

Besides the full ``run_stcrpy`` (geometry + interactions), this module exposes a
light annotation path used as an **alternative loader** for the scorer benchmark:
``stcrpy_chains`` (TCR α/β vs pMHC chain ids) and ``region_ca_map`` (per-CDR/FR Cα,
keyed exactly like kinapse's native loader) — so ``kinapse.benchmarks`` can identify
chains and annotate CDRs with STCRpy instead of the native kinapse loader.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HINT = (
    "STCRpy not found. It needs its own environment (heavy deps: ANARCI models, PLIP, OpenBabel):\n"
    "  pip install stcrpy && ANARCI --build_models && pip install plip\n"
    "Then point kinapse at that env:\n"
    "  export KINAPSE_STCRPY_PYTHON=/path/to/env/bin/python\n"
    "(or install stcrpy into the active interpreter)."
)
_RUNNER = str(Path(__file__).with_name("_stcrpy_runner.py"))


def _python() -> Optional[List[str]]:
    """The interpreter to run STCRpy with: env var -> active interpreter -> None."""
    env_py = os.environ.get("KINAPSE_STCRPY_PYTHON")
    if env_py:
        return [env_py]
    if importlib.util.find_spec("stcrpy") is not None:
        return [sys.executable]
    return None


def available() -> bool:
    """True if a STCRpy environment can be located."""
    return _python() is not None


def run_stcrpy(pdb: str, timeout: float = 600.0, geometry: bool = True,
               interactions: bool = True) -> Dict[str, Any]:
    """Run STCRpy on a PDB (TCR or TCR-pMHC) in its external env.

    Returns a dict with ``receptor_chains``/``ligand_chains`` and per-region ``regions``
    (always), plus docking-geometry ``angles`` (if ``geometry``) and PLIP-typed
    ``interactions`` (if ``interactions``), and a ``status`` key. Raises
    :class:`ImportError` with an install hint if no STCRpy environment is found.
    """
    py = _python()
    if py is None:
        raise ImportError(_HINT)
    payload = {"pdb": str(pdb), "geometry": geometry, "interactions": interactions}
    try:
        proc = subprocess.run(py + [_RUNNER], input=json.dumps(payload),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"STCRpy exceeded {timeout}s"}
    if proc.returncode != 0:
        return {"status": "error", "error": (proc.stderr or "").strip()[-800:]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "error": f"stcrpy runner did not emit JSON: {proc.stdout[:300]!r}"}


def annotate(pdb: str, timeout: float = 300.0) -> Dict[str, Any]:
    """Light STCRpy pass: chains + CDR/FR Cα only (no geometry, no PLIP).

    Used as the benchmark's alternative loader. Raises on failure with STCRpy's error.
    """
    res = run_stcrpy(pdb, timeout=timeout, geometry=False, interactions=False)
    if res.get("status") != "ok":
        raise RuntimeError(f"STCRpy annotation failed for {pdb}: {res.get('error') or res}")
    return res


def stcrpy_chains(pdb: str, timeout: float = 300.0) -> Tuple[List[str], List[str]]:
    """(receptor, ligand) original chain ids via STCRpy — TCR α/β vs peptide+MHC.

    Drop-in for :func:`kinapse.scoring.infer_tcr_pmhc_chains` (the ``stcrpy`` loader)."""
    res = annotate(pdb, timeout=timeout)
    rec = list(res.get("receptor_chains") or [])
    lig = list(res.get("ligand_chains") or [])
    if not rec:
        raise ValueError(f"STCRpy found no TCR (receptor) chains in {pdb}")
    return rec, lig


def region_ca_map(pdb: str, timeout: float = 300.0) -> Dict[tuple, "Any"]:
    """{(region, chain, imgt_num): Cα xyz} via STCRpy — same keys as the native loader.

    Feeds :func:`kinapse.benchmarks.structural_agreement` when ``loader='stcrpy'``."""
    import numpy as np

    res = annotate(pdb, timeout=timeout)
    out: Dict[tuple, Any] = {}
    for r in res.get("regions") or []:
        try:
            out[(r["region"], r["chain"], int(r["imgt"]))] = np.asarray(r["xyz"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
    return out


__all__ = ["run_stcrpy", "available", "annotate", "stcrpy_chains", "region_ca_map"]
