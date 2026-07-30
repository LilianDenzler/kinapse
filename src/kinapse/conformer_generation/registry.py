"""Registered conformer generators. DiG is wired via the existing dig_runner;
AlphaFlow/BioEmu are declared for the runner engine."""
from __future__ import annotations

from kinapse.runners import RunnerSpec, register

for _spec in [
    RunnerSpec(name="dig", tier="conformer_generation", backend="uvenv", status="scaffold",
               description="DiG (Distributional Graphormer) diffusion sampler — see dig_runner.",
               outputs=("ensemble_pdbs", "xtc"), homepage="https://github.com/microsoft/AI2BMD"),
    RunnerSpec(name="alphaflow", tier="conformer_generation", backend="uvenv", status="planned",
               description="AlphaFlow — AlphaFold + flow-matching conformational ensembles.",
               outputs=("ensemble_pdbs",), requirements=("torch",),
               homepage="https://github.com/bjing2016/alphaflow"),
    RunnerSpec(name="bioemu", tier="conformer_generation", backend="uvenv", status="planned",
               description="BioEmu — generative equilibrium-ensemble emulator (monomer).",
               outputs=("ensemble_pdbs",), requirements=("torch",),
               homepage="https://github.com/microsoft/bioemu"),
]:
    register(_spec)
