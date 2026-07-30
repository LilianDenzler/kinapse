"""Registered TCR-pMHC docking engines. Declared now; wiring is WIP."""
from __future__ import annotations

from kinapse.runners import RunnerSpec, register

for _spec in [
    RunnerSpec(name="haddock", tier="docking", backend="uvenv", status="planned",
               description="HADDOCK — information-driven docking; best classic TCR-pMHC docker.",
               outputs=("pdb", "haddock_score"), homepage="https://www.bonvinlab.org/software/haddock3/"),
    RunnerSpec(name="rosettadock", tier="docking", backend="uvenv", status="planned",
               description="RosettaDock / TCRFlexDock — flexible TCR-pMHC docking.",
               outputs=("pdb", "interface_score"), license="rosetta"),
    RunnerSpec(name="cluspro", tier="docking", backend="uvenv", status="planned",
               description="ClusPro — FFT rigid-body docking (server).",
               outputs=("pdb", "cluster_score"), homepage="https://cluspro.bu.edu"),
]:
    register(_spec)
