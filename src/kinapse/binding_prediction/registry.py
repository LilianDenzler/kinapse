"""Registered TCR-pMHC binding/specificity models. Declared now; wiring is WIP."""
from __future__ import annotations

from kinapse.runners import RunnerSpec, register

for _spec in [
    RunnerSpec(name="nettcr", tier="binding_prediction", backend="uvenv", status="planned",
               description="NetTCR-2.2 — CNN on paired CDR3α/β + peptide (sequence).",
               outputs=("score",), homepage="https://github.com/mnielLab/NetTCR-2.2"),
    RunnerSpec(name="tulip", tier="binding_prediction", backend="uvenv", status="planned",
               description="TULIP — unsupervised transformer, aims at unseen epitopes (sequence).",
               outputs=("score",), homepage="https://github.com/barthelemymp/TULIP-TCR"),
    RunnerSpec(name="mixtcrpred", tier="binding_prediction", backend="uvenv", status="planned",
               description="MixTCRpred — transformer, epitope-specific models (sequence).",
               outputs=("score",), homepage="https://github.com/GfellerLab/MixTCRpred"),
    RunnerSpec(name="stag", tier="binding_prediction", backend="uvenv", status="planned",
               description="STAG — GNN on the modeled 3D interface (structure-based).",
               outputs=("score",), tags=("structure",)),
    RunnerSpec(name="tcren", tier="binding_prediction", backend="uvenv", status="planned",
               description="TCRen — TCR-peptide statistical potential; ranks unseen epitopes (structure).",
               outputs=("energy", "rank"), tags=("structure",),
               homepage="https://github.com/antigenomics/tcren"),
]:
    register(_spec)
