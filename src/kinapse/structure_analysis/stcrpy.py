"""Run OPIG **STCRpy** (2025) as a completely separate external model.

STCRpy annotates a TCR / TCR-pMHC structure (chains + CDRs), computes docking
geometry, profiles PLIP-typed interface interactions, and reports TCR-DockQ /
interface-RMSD. It pulls heavy deps (ANARCI models, PLIP, OpenBabel), so kinapse
**never imports it** — it runs in its own environment behind a JSON contract.

Point kinapse at that environment with ``KINAPSE_STCRPY_PYTHON`` (recommended);
otherwise the active interpreter is used if ``stcrpy`` is importable there; else a
clear install hint is raised. Registered as the ``stcrpy`` runner in the
``structure_analysis`` tier (:mod:`kinapse.structure_analysis`).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HINT = (
    "STCRpy not found. It needs its own environment (heavy deps: ANARCI models, PLIP, OpenBabel):\n"
    "  pip install stcrpy && ANARCI --build_models && pip install plip\n"
    "Then point kinapse at that env:\n"
    "  export KINAPSE_STCRPY_PYTHON=/path/to/env/bin/python\n"
    "(or install stcrpy into the active interpreter)."
)
_RUNNER = str(Path(__file__).with_name("_stcrpy_runner.py"))


def _python() -> Optional[List[str]]:
    """The interpreter to run STCRpy with: env var → active interpreter → None."""
    env_py = os.environ.get("KINAPSE_STCRPY_PYTHON")
    if env_py:
        return [env_py]
    if importlib.util.find_spec("stcrpy") is not None:
        return [sys.executable]
    return None


def available() -> bool:
    """True if a STCRpy environment can be located."""
    return _python() is not None


def run_stcrpy(pdb: str, timeout: float = 600.0) -> Dict[str, Any]:
    """Run STCRpy on a PDB (TCR or TCR-pMHC) in its external env.

    Returns a dict with STCRpy's docking-geometry ``angles`` and PLIP-typed
    ``interactions`` (plus a ``status`` key). Raises :class:`ImportError` with an
    install hint if no STCRpy environment is found.
    """
    py = _python()
    if py is None:
        raise ImportError(_HINT)
    try:
        proc = subprocess.run(py + [_RUNNER], input=json.dumps({"pdb": str(pdb)}),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"STCRpy exceeded {timeout}s"}
    if proc.returncode != 0:
        return {"status": "error", "error": (proc.stderr or "").strip()[-800:]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "error": f"stcrpy runner did not emit JSON: {proc.stdout[:300]!r}"}


__all__ = ["run_stcrpy", "available"]
