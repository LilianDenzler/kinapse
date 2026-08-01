"""Structural agreement of a modelled TCR complex vs its ground truth.

Uses the kinapse loader to IMGT-number both structures and identify the CDR/FR
regions, then — the same Kabsch region-superposition used in the ensemble analysis
— computes Cα RMSDs between model and GT:

  * per-CDR RMSD after aligning on that chain's **framework** (α-fwk → α-CDRs,
    β-fwk → β-CDRs),
  * per-CDR **local** RMSD (loop superposed on itself — conformation only),
  * **Cα iRMSD over the 6 CDR loops** (aligned on the whole framework), and
  * framework RMSD.

`assign_tier` maps (Cα iRMSD, DockQ) to an HQ / MQ / AQ / LQ quality tier
(HQ ≤2 Å & DockQ ≥0.8; MQ ≤5 Å & ≥0.49; AQ <5 Å & ≥0.23; else LQ). Residue
correspondence is by IMGT number, so it only compares residues present in both.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

A_FR = ["A_FR1", "A_FR2", "A_FR3", "A_FR4"]
B_FR = ["B_FR1", "B_FR2", "B_FR3", "B_FR4"]
A_CDR = ["A_CDR1", "A_CDR2", "A_CDR3"]
B_CDR = ["B_CDR1", "B_CDR2", "B_CDR3"]
ALL_FR = A_FR + B_FR
ALL_CDR = A_CDR + B_CDR


def _superpose(P: np.ndarray, Q: np.ndarray):
    """Kabsch: rotation R + translation t mapping P onto Q (least squares)."""
    Pm, Qm = P.mean(0), Q.mean(0)
    H = (P - Pm).T @ (Q - Qm)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qm - R @ Pm


def _rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def _ca_by_region(pair) -> Dict[tuple, np.ndarray]:
    """{(region, chain, imgt_num): Cα xyz} for an IMGT-numbered TCRPairView."""
    from kinapse.regions import CDR_FR_RANGES

    out: Dict[tuple, np.ndarray] = {}
    vs = pair.variable_structure                      # Bio.PDB Structure (or Model)
    chains = vs.get_chains() if hasattr(vs, "get_chains") else vs
    for chain in chains:                              # chains renamed A/B, IMGT-numbered
        cid = chain.id
        for res in chain:
            if res.id[0] != " " or "CA" not in res:
                continue
            num = res.id[1]
            for rname, (lo, hi) in CDR_FR_RANGES.items():
                if rname[0] == cid and lo <= num <= hi:
                    out[(rname, cid, num)] = np.asarray(res["CA"].coord, dtype=float)
                    break
    return out


def _aligned_rmsd(mdl, gt, align_regions: List[str], rmsd_regions: List[str]) -> Optional[float]:
    """Superpose model onto GT using `align_regions` Cα, then RMSD over `rmsd_regions`."""
    align_set, rmsd_set = set(align_regions), set(rmsd_regions)
    ak = [k for k in mdl if k in gt and k[0] in align_set]
    rk = [k for k in mdl if k in gt and k[0] in rmsd_set]
    if len(ak) < 3 or len(rk) < 1:
        return None
    R, t = _superpose(np.array([mdl[k] for k in ak]), np.array([gt[k] for k in ak]))
    P = np.array([mdl[k] for k in rk]) @ R.T + t
    return round(_rmsd(P, np.array([gt[k] for k in rk])), 3)


def structural_agreement(model_pdb, gt_pdb, legacy_anarci: bool = False) -> Dict[str, Optional[float]]:
    """Cα RMSD metrics between a modelled complex and its ground truth (first pair)."""
    from kinapse.structures import TCR

    m = TCR(input_pdb=str(model_pdb), legacy_anarci=legacy_anarci)
    g = TCR(input_pdb=str(gt_pdb), legacy_anarci=legacy_anarci)
    if not m.pairs or not g.pairs:
        return {}
    mdl, gt = _ca_by_region(m.pairs[0]), _ca_by_region(g.pairs[0])

    out: Dict[str, Optional[float]] = {}
    for cdr in A_CDR:
        out[f"struct__{cdr}_rmsd"] = _aligned_rmsd(mdl, gt, A_FR, [cdr])   # α-framework aligned
    for cdr in B_CDR:
        out[f"struct__{cdr}_rmsd"] = _aligned_rmsd(mdl, gt, B_FR, [cdr])   # β-framework aligned
    for cdr in ALL_CDR:
        out[f"struct__{cdr}_local_rmsd"] = _aligned_rmsd(mdl, gt, [cdr], [cdr])  # loop-only
    out["struct__fwk_rmsd"] = _aligned_rmsd(mdl, gt, ALL_FR, ALL_FR)
    out["struct__cdr_irmsd"] = _aligned_rmsd(mdl, gt, ALL_FR, ALL_CDR)     # Cα iRMSD over the 6 CDRs
    return out


def assign_tier(irmsd: Optional[float], dockq: Optional[float] = None) -> Optional[str]:
    """HQ / MQ / AQ / LQ from Cα iRMSD (+ DockQ when available)."""
    if irmsd is None or (isinstance(irmsd, float) and math.isnan(irmsd)):
        return None
    dq = None if (dockq is None or (isinstance(dockq, float) and math.isnan(dockq))) else dockq
    if irmsd <= 2.0 and (dq is None or dq >= 0.80):
        return "HQ"
    if irmsd <= 5.0 and (dq is None or dq >= 0.49):
        return "MQ"
    if irmsd < 5.0 and (dq is None or dq >= 0.23):
        return "AQ"
    return "LQ"


__all__ = ["structural_agreement", "assign_tier"]
