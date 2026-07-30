"""Registered sequence-embedding models. `pdb_to_fasta` is a native helper in the
module API; the heavy embedders below are declared for the runner engine."""
from __future__ import annotations

from kinapse.runners import RunnerSpec, register

for _spec in [
    RunnerSpec(name="mmseqs2_msa", tier="sequence_embedding", backend="uvenv", status="scaffold",
               description="MMseqs2-GPU MSA generation (feeds Evoformer).",
               outputs=("a3m",), homepage="https://github.com/soedinglab/MMseqs2"),
    RunnerSpec(name="openfold_evoformer", tier="sequence_embedding", backend="uvenv", status="scaffold",
               description="OpenFold / Evoformer sequence embeddings (needs torch + OpenFold).",
               outputs=("embedding",), requirements=("torch",),
               homepage="https://github.com/aqlaboratory/openfold"),
    RunnerSpec(name="esm2", tier="sequence_embedding", backend="uvenv", status="planned",
               description="ESM-2 protein language-model embeddings (MSA-free).",
               outputs=("embedding",), requirements=("fair-esm",)),
]:
    register(_spec)
