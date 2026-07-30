"""Scoring runners are provided by ifscore (see :mod:`kinapse.scoring`). They are
declared here so they also appear in the unified runner map
(``kinapse.runners.specs('scoring')``). Execution goes through
``kinapse.scoring.score()`` / ``score_batch()``, not the generic engine."""
from __future__ import annotations

from kinapse.runners import RunnerSpec, register

for _spec in [
    RunnerSpec(name="geometry_scoring", tier="scoring", backend="native", status="stable",
               description="ifscore interface geometry — BSA / SASA / contacts (no external tool).",
               outputs=("bsa", "sasa_complex", "n_contacts_5A")),
    RunnerSpec(name="dockq", tier="scoring", backend="uvenv", status="stable",
               description="ifscore DockQ — docking quality vs a native reference.",
               outputs=("dockq", "irmsd", "lrmsd", "fnat"), needs_native_ref=True),
    RunnerSpec(name="prodigy", tier="scoring", backend="uvenv", status="stable",
               description="ifscore PRODIGY — predicted binding affinity / interface energy.",
               outputs=("ba_val", "kd")),
]:
    register(_spec)
