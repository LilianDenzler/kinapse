#!/usr/bin/env python
"""Direct test of the hinge implication: does each CDR loop's rotation axis pass through its CONSTRAINED stem?

Per TCR/chain/CDR:
  1. superpose every frame on the rigid consensus FRAMEWORK (Kabsch) -> loop coords in the framework frame
  2. fit the loop's rigid-body motion per frame and solve for its common PIVOT p (the fixed point of the
     rotation: least-squares over frames of (I - R_t) p = c_t)
  3. project p onto the N-stem -> C-stem axis:  frac = 0 at the N-stem, 1 at the C-stem
If the loop hinges about its clamped stem, frac should sit on the constrained side:
  CDR1 -> C (frac>0.5),  CDR2 -> N (frac<0.5),  CDR3 -> N (frac<0.5).
Saved/aggregated over the 22 TCRs. Alignment uses only the rigid framework (the reference we trust)."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob, warnings, traceback, json
warnings.simplefilter("ignore")
import numpy as np
from graph_build import load_md, consensus, CDR_RANGES, DATA, HERE

RES = f"{HERE}/results_pivot"
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}


def kabsch(P, Q):
    """Rotation R mapping centered P onto centered Q (min ||R P - Q||)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def loop_pivot_frac(FW, loop, nstem_idx, cstem_idx):
    """FW (T,M,3) framework CA, loop (T,N,3). Returns frac of pivot along N->C stem axis."""
    ref = FW[0] - FW[0].mean(0); refc = FW[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(len(FW)):
        R = kabsch(FW[t] - FW[t].mean(0), ref)            # map frame t framework onto frame-0 framework
        Lf[t] = (loop[t] - FW[t].mean(0)) @ R.T + refc    # loop in the (fixed) framework frame
    Lbar = Lf.mean(0); Lbarc = Lbar.mean(0)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for t in range(len(Lf)):
        Rp = kabsch(Lbar - Lbarc, Lf[t] - Lf[t].mean(0))  # rigid rotation of the mean loop onto frame t
        c = Lf[t].mean(0) - Rp @ Lbarc                    # translation; fixed point solves (I-Rp)p = c
        M = np.eye(3) - Rp
        A += M.T @ M; b += M.T @ c
    p = np.linalg.lstsq(A, b, rcond=None)[0]              # least-squares common pivot
    Nc = Lbar[nstem_idx].mean(0); Cc = Lbar[cstem_idx].mean(0)
    axis = Cc - Nc
    return float(np.dot(p - Nc, axis) / np.dot(axis, axis))


def process(sysid):
    tv, xyz, imap = load_md(sysid, stride=100)             # coarse stride is plenty for a geometric pivot
    con = consensus()
    out = {}
    for ch in "AB":
        fwi = np.array([imap[ch][r] for r in sorted(con[ch]) if r in imap[ch]])
        FW = xyz[:, fwi]
        for cdr, (lo, hi) in CDR_RANGES.items():
            keys = sorted(k for k in imap[ch] if lo <= k <= hi)     # loop residues incl. insertions
            if len(keys) < 4:
                continue
            loop = xyz[:, np.array([imap[ch][k] for k in keys])]
            ni = [keys.index(k) for k in keys[:2]]                  # N-stem = 2 lowest-IMGT loop residues
            ci = [keys.index(k) for k in keys[-2:]]                 # C-stem = 2 highest
            try:
                out[f"{ch}_{cdr}"] = round(loop_pivot_frac(FW, loop, ni, ci), 3)
            except Exception:
                out[f"{ch}_{cdr}"] = None
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    systems = sorted(os.path.basename(p) for p in glob.glob(f"{DATA}/*")
                     if len(os.path.basename(p)) == 4 and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    allres = {}
    for i, s in enumerate(systems):
        try:
            r = process(s); allres[s] = r
            print(f"[{i+1}/{len(systems)}] {s}: " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:120]}", flush=True); traceback.print_exc()
    json.dump(allres, open(f"{RES}/pivot_frac.json", "w"), indent=2)
    # aggregate per CDR
    print("\n=== pivot position along N->C stem axis (0=N stem, 1=C stem), median over TCRs ===")
    print(f"{'CDR':8}{'median frac':>12}{'clamp side':>12}{'pivot on clamp side?':>22}")
    for cdr in ["CDR1", "CDR2", "CDR3"]:
        for ch in "AB":
            vals = [allres[s][f"{ch}_{cdr}"] for s in allres if allres[s].get(f"{ch}_{cdr}") is not None]
            if not vals:
                continue
            m = float(np.median(vals)); side = CLAMP[cdr]
            ok = (m < 0.5) if side == "N" else (m > 0.5)
            print(f"{ch}_{cdr:5}{m:12.2f}{side:>12}{('YES' if ok else 'no'):>22}")
    print("PIVOT_DONE", flush=True)


if __name__ == "__main__":
    main()
