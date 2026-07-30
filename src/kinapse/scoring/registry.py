"""Scoring runners are provided by ifscore (see :mod:`kinapse.scoring`). They are
declared here so they also appear in the unified runner map
(``kinapse.runners.specs('scoring')``). Execution goes through
``kinapse.scoring.score()`` / ``score_batch()``, not the generic engine.

Statuses reflect what ifscore ships today (``stable``) vs. physics-based
free-energy scorers that are declared/planned and provisioned via an isolated env
(``planned``). See ``docs/scoring.md`` for the cost↔accuracy ladder and how each
works.
"""
from __future__ import annotations

from kinapse.runners import RunnerSpec, register

for _spec in [
    # --- shipped in ifscore today ---
    RunnerSpec(name="geometry_scoring", tier="scoring", backend="native", status="stable",
               description="ifscore interface geometry — BSA / SASA / contacts (no external tool).",
               outputs=("bsa", "sasa_complex", "n_contacts_5A")),
    RunnerSpec(name="dockq", tier="scoring", backend="uvenv", status="stable",
               description="DockQ — docking quality vs a native reference.",
               outputs=("dockq", "irmsd", "lrmsd", "fnat"), needs_native_ref=True),
    RunnerSpec(name="prodigy", tier="scoring", backend="uvenv", status="stable",
               description="PRODIGY — contacts-based predicted binding affinity / ΔG.",
               outputs=("ba_val", "kd")),
    RunnerSpec(name="foldx", tier="scoring", backend="uvenv", status="stable",
               description="FoldX empirical ΔΔG (needs a FoldX academic licence — see docs/scoring.md).",
               outputs=("ddg",), license="academic (time-limited)"),
    RunnerSpec(name="rosetta", tier="scoring", backend="uvenv", status="stable",
               description="Rosetta InterfaceAnalyzer: dG_separated, dSASA, shape complementarity, packstat.",
               outputs=("dG_separated", "dSASA_int", "sc_value", "packstat"),
               license="academic (free, non-commercial)"),
    RunnerSpec(name="haddock", tier="scoring", backend="uvenv", status="stable",
               description="HADDOCK score — empirical docking score of an interface.",
               outputs=("haddock_score",)),

    # --- physics-based free-energy scorers (ensemble / MD; heavy, GPU) ---
    RunnerSpec(name="mmgbsa", tier="scoring", backend="container", status="planned",
               description="MM-GBSA end-point ΔG over short MD (OpenMM/Amber). Ensemble-averaged "
                           "binding-energy RANKING + per-residue hotspot decomposition; not absolute ΔG. "
                           "TCR-validated (Crean 2022): GB-Neck2 (igb=8), ε_int≈6, ~5–15 short replicas, "
                           "entropy usually omitted.",
               outputs=("dG_bind", "dG_gb", "dG_elec", "dG_vdw", "dG_nonpolar", "per_residue"),
               requirements=("openmm", "pdbfixer"),
               tags=("energy", "endpoint", "md", "gpu"),
               homepage="https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA"),
    RunnerSpec(name="mmpbsa", tier="scoring", backend="container", status="planned",
               description="MM-PBSA end-point ΔG (Poisson–Boltzmann solvent). Cross-check for MM-GBSA — "
                           "slower (~60 min/complex) but often better calibrated on charged interfaces.",
               outputs=("dG_bind", "dG_pb", "dG_elec", "dG_vdw", "dG_nonpolar", "per_residue"),
               requirements=("openmm", "pdbfixer"),
               tags=("energy", "endpoint", "md", "gpu")),
    RunnerSpec(name="rosetta_flexddg", tier="scoring", backend="uvenv", status="planned",
               description="Rosetta flex-ddG — backrub-ensemble ΔΔG on interface mutations; cheap, "
                           "orthogonal check to MM-GBSA.",
               outputs=("ddg",), license="academic (free, non-commercial)",
               tags=("energy", "ensemble", "rosetta")),
    RunnerSpec(name="fep", tier="scoring", backend="container", status="planned",
               description="Alchemical FEP/TI relative ΔΔG via OpenFE (OpenMM). Highest rigor "
                           "(~1 kcal/mol); reserve for a few top candidates — very heavy.",
               outputs=("ddg",), requirements=("openfe",),
               tags=("energy", "alchemical", "gpu"),
               homepage="https://github.com/OpenFreeEnergy/openfe"),
]:
    register(_spec)
