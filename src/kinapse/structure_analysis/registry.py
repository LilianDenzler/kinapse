"""Registered external structure-analysis tools."""
from __future__ import annotations

from kinapse.runners import RunnerSpec, register
from .stcrpy import run_stcrpy

register(RunnerSpec(
    name="stcrpy", tier="structure_analysis", backend="native", status="stable",
    description="OPIG STCRpy (2025) — TCR / TCR-pMHC chain+CDR annotation, docking geometry, "
                "PLIP-typed interface interactions, TCR-DockQ / interface-RMSD. Runs in its own "
                "external env (set KINAPSE_STCRPY_PYTHON).",
    outputs=("angles", "interactions"),
    homepage="https://github.com/oxpig/STCRpy",
    tags=("tcr", "annotation", "external"),
    func=lambda inputs: run_stcrpy(inputs["pdb"], timeout=inputs.get("timeout", 600.0)),
))
