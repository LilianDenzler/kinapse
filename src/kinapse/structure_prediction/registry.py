"""Registered structure-prediction models. Declared now; execution wiring is WIP."""
from __future__ import annotations

from kinapse.runners import RunnerSpec, register

for _spec in [
    RunnerSpec(name="alphafold3", tier="structure_prediction", backend="uvenv", status="planned",
               description="AlphaFold3 — best current TCR-pMHC complex accuracy (median DockQ ~0.64/0.68).",
               outputs=("pdb", "plddt", "ptm", "iptm"),
               homepage="https://github.com/google-deepmind/alphafold3", license="restricted"),
    RunnerSpec(name="boltz2", tier="structure_prediction", backend="uvenv", status="planned",
               description="Boltz-2 — open co-folding + binding-affinity prediction.",
               outputs=("pdb", "affinity"), homepage="https://github.com/jwohlwend/boltz"),
    RunnerSpec(name="tcrdock", tier="structure_prediction", backend="uvenv", status="planned",
               description="TCRdock — AF pipeline specialised for TCR:pMHC docking geometry (Z-score).",
               outputs=("pdb", "zscore"), homepage="https://github.com/phbradley/TCRdock"),
    RunnerSpec(name="tcrmodel2", tier="structure_prediction", backend="uvenv", status="planned",
               description="TCRmodel2 — AF-Multimer adaptation with TCR-specific MSAs/templates.",
               outputs=("pdb", "iptm"), homepage="https://tcrmodel.ibbr.umd.edu"),
    RunnerSpec(name="immunebuilder", tier="structure_prediction", backend="uvenv", status="planned",
               description="ImmuneBuilder/TCRBuilder2 — fast TCR V-domain structure (TCR only).",
               outputs=("pdb",), homepage="https://github.com/oxpig/ImmuneBuilder"),
]:
    register(_spec)
