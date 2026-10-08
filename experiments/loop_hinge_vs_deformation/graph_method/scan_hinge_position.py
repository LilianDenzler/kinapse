#!/usr/bin/env python
"""Alignment-free scan of the hinge-anchor POSITION, per CDR.

For each CDR loop [lo,hi] we slide a symmetric anchor pair outward from the loop into the framework:
    offset s  ->  left anchor = lo - s ,  right anchor = hi + s
    s <= 0 : anchor sits INSIDE the loop     s = 1 : first residue outside     s >= 1 : walk into the framework
Everything is computed from CA-CA distances only (no alignment, ever):

  rigid_L, rigid_R  : std over the MD of the anchor's distances to the consensus rigid framework, averaged over
                      framework atoms.  LOW = the residue moves rigidly WITH the framework core (a good, solid
                      anchor);  HIGH = the residue is mobile (loop-like).  ==> the SELECTION metric.
  base_width_*      : the anchor-anchor distance (mean, std over MD) -- base geometry / within-MD stability.
  f_hinge           : model-free hinge fraction relative to this base = the fraction of the loop->base distance
                      fluctuation that is NOT explained by the loop's own internal deformation z (z = PCA of the
                      loop's d_LL, which a rigid hinge cannot change).  Rises monotonically as the base moves out,
                      so it is REPORTED, not used to choose (per the rigidity rule).
  in_loop           : True if the anchor is inside the loop (s<=0).
  is_consensus      : True if both anchors are curated rigid-framework residues.

Chosen anchor per CDR (downstream) = the framework offset (s>=1) with minimum mean rigidity.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import argparse, glob, tempfile, warnings, traceback
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, mdtraj as md
from kinapse.structures import load_tcr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as PC
from consensus import load_consensus

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_scan")
CON = load_consensus()
CDR_RANGES = {"A_CDR1": (27, 38), "A_CDR2": (56, 65), "A_CDR3": (105, 117),
              "B_CDR1": (27, 38), "B_CDR2": (56, 65), "B_CDR3": (105, 117)}
OFFSETS = [-2, -1, 0, 1, 2, 3, 4, 5, 6]


def dmat(a, b):
    return np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1)


def hinge_frac(loop, base):
    """model-free, alignment-free: 1 - (fraction of d_LF explained by loop deformation z)."""
    N = loop.shape[1]; iu = np.triu_indices(N, 1)
    dLL = dmat(loop, loop)[:, iu[0], iu[1]]; dLL = dLL - dLL.mean(0)
    if (dLL ** 2).sum() < 1e-9:
        z = np.zeros((len(loop), 0))
    else:
        U, S, _ = np.linalg.svd(dLL, full_matrices=False)
        k = min(8, int((np.cumsum(S ** 2) / (S ** 2).sum() < 0.95).sum()) + 1)
        z = U[:, :k] * S[:k]
    dLF = dmat(loop, base).reshape(len(loop), -1); dLF = dLF - dLF.mean(0)
    if z.shape[1]:
        fit = z @ np.linalg.lstsq(z, dLF, rcond=None)[0]
        f_def = (fit ** 2).sum() / max((dLF ** 2).sum(), 1e-9)
    else:
        f_def = 0.0
    return float(1 - f_def)


def process(sysid):
    pdb, xtc = f"{PC.DATA}/{sysid}/{sysid}.pdb", f"{PC.DATA}/{sysid}/{sysid}.xtc"
    tmp = tempfile.mktemp(suffix=".xtc")
    try:
        md.load(xtc, top=pdb, stride=PC.STRIDE).save_xtc(tmp)
        kw = {"manual_chain_types": PC.CHAIN_OVERRIDES[sysid]} if sysid in PC.CHAIN_OVERRIDES else {}
        tv = load_tcr(pdb, traj=tmp, **kw).pairs[0].traj
        xyz = tv.mdtraj.xyz * 10.0
        imap = {ch: {int(n[1]): int(i) for i, n in zip(*tv.domain_idx([f"{ch}_variable"], atom_names={"CA"}, pass_names=True))} for ch in "AB"}
        fwci = {ch: np.array([imap[ch][a] for a in sorted(CON[ch]) if a in imap[ch]]) for ch in "AB"}
        fwd = {ch: xyz[:, fwci[ch]] for ch in "AB"}                      # framework CA (T,M,3)

        def rigidity(ch, res):
            if res not in imap[ch]:
                return np.nan
            d = dmat(xyz[:, [imap[ch][res]]], fwd[ch])[:, 0, :]          # (T,M)
            return float(d.std(0).mean())

        rows = []
        for cdr, (lo, hi) in CDR_RANGES.items():
            ch = cdr[0]
            li = np.asarray(tv.domain_idx([cdr], atom_names={"CA"}))
            loop = xyz[:, li]
            for s in OFFSETS:
                L, R = lo - s, hi + s
                if L not in imap[ch] or R not in imap[ch]:
                    continue
                base = np.array([imap[ch][L], imap[ch][R]])
                bw = dmat(xyz[:, [imap[ch][L]]], xyz[:, [imap[ch][R]]])[:, 0, 0]
                rl, rr = rigidity(ch, L), rigidity(ch, R)
                rows.append(dict(
                    system=sysid, cdr=cdr, chain=ch, s=s, L=L, R=R,
                    in_loop=bool(s <= 0), is_consensus=bool(L in CON[ch] and R in CON[ch]),
                    rigid_L=round(rl, 3), rigid_R=round(rr, 3), rigid=round((rl + rr) / 2, 3),
                    base_width_mean=round(float(bw.mean()), 3), base_width_std=round(float(bw.std()), 3),
                    f_hinge=round(hinge_frac(loop, xyz[:, base]), 3),
                ))
        pd.DataFrame(rows).to_csv(f"{RES}/{sysid}_scan.csv", index=False)
        return len(rows)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--systems", nargs="*"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); os.makedirs(RES, exist_ok=True)
    systems = a.systems or sorted(os.path.basename(p) for p in glob.glob(f"{PC.DATA}/*")
                                  if len(os.path.basename(p)) == 4 and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    for i, s in enumerate(systems):
        if os.path.exists(f"{RES}/{s}_scan.csv") and not a.force:
            print(f"[{s}] done", flush=True); continue
        try:
            print(f"[{i+1}/{len(systems)}] {s}: {process(s)} rows", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:150]}", flush=True); traceback.print_exc()
    print("SCAN_DONE", flush=True)


if __name__ == "__main__":
    main()
