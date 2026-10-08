#!/usr/bin/env python
"""Reference-free CDR internal deformation from the VARIANCE of intra-loop CA-CA distances.

For CDR loop L, over the MD ensemble (no reference frame, no alignment, no fit):
  D_deform = sqrt( mean_{i<j in L} Var_t[d_ij] )                       (Å, absolute)
  eps      = sqrt( mean_{i<j in L} Var_t[d_ij] / <d_ij>_t^2 )          (dimensionless, length-comparable)
Both invariant to rotation/translation; normalized per pair. Insertions included in the loop."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob, warnings, traceback
warnings.simplefilter("ignore")
import numpy as np
from graph_build import load_md, CDR_RANGES, DATA, HERE

RES = f"{HERE}/results_deform"
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]


def process(s):
    tv, xyz, imap = load_md(s)
    save = {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        N = loop.shape[1]; iu = np.triu_indices(N, 1)
        d = np.linalg.norm(loop[:, iu[0]] - loop[:, iu[1]], axis=-1)   # (T, N_LL)
        var = d.var(0); mean = d.mean(0)
        save[f"{cdr}_Ddef"] = np.float32(np.sqrt(var.mean()))
        save[f"{cdr}_eps"] = np.float32(np.sqrt((var / mean ** 2).mean()))
        save[f"{cdr}_N"] = np.int32(N)
    np.savez(f"{RES}/{s}.npz", **save)
    return {cdr: round(float(save[f"{cdr}_eps"]), 3) for cdr in CDRS}


def main():
    os.makedirs(RES, exist_ok=True)
    systems = sorted(os.path.basename(p) for p in glob.glob(f"{DATA}/*")
                     if len(os.path.basename(p)) == 4 and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    for i, s in enumerate(systems):
        try:
            print(f"[{i+1}/{len(systems)}] {s}: {process(s)}", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:120]}", flush=True); traceback.print_exc()
    print("DEFORM_DONE", flush=True)


if __name__ == "__main__":
    main()
