"""The curated rigid-framework reference residues (IMGT), per chain.

These are the STATIC CONSENSUS residues from kinapse's packaged geometry data (identical to the TCR_Metrics
consensus_output): each IMGT position scored by its structural-alignment RMSD across many TCRs, keeping only the
low-RMSD (rigid, conserved) ones. Crucially this EXCLUDES not only the CDR loops but also the mobile framework
loops inside FR3 (e.g. chain-A 82-85, RMSD up to 6 A) -- so it is the correct rigid reference, unlike 'all of FR'.
"""
from __future__ import annotations
import os
import functools


@functools.lru_cache(maxsize=1)
def load_consensus():
    """{'A': set(imgt), 'B': set(imgt)} rigid framework residues, from kinapse's packaged consensus_output."""
    import kinapse.geometry as g
    base = str(g.DATA_PATH)                                # already .../geometry/data/consensus_output
    out = {}
    for ch in "AB":
        txt = open(f"{base}/chain_{ch}/consensus_alignment_residues.txt").read().strip()
        out[ch] = {int(x) for x in txt.split(",") if x.strip()}
    return out
