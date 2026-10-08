#!/usr/bin/env python
"""Clamp-anchored rigid-motion amplitude per CDR per TCR: rotation angle theta of the loop's rigid fit
(pivoting near the clamp), axis concentration P1, and pivot location. Robust (all loop atoms, no 3-atom fit)."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob, warnings, traceback
warnings.simplefilter("ignore")
import numpy as np
from motion_characterize import characterize
from geom_hinge import CDRS
from graph_build import load_md, CDR_RANGES, DATA, HERE

RES = f"{HERE}/results_swing"


def process(s, RSET):
    tv, xyz, imap = load_md(s)
    save = {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        ri = np.array([imap[ch][r] for r in RSET[ch] if r in imap[ch]])
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        nstem = [lk.index(k) for k in lk[:2]]; cstem = [lk.index(k) for k in lk[-2:]]
        c = characterize(loop, xyz[:, ri], nstem, cstem)
        for k in ("theta", "theta_p95", "P1", "frac"):
            save[f"{cdr}_{k}"] = np.float32(c[k])
    np.savez(f"{RES}/{s}.npz", **save)
    return {cdr: round(float(save[f"{cdr}_theta"]), 1) for cdr in CDRS}


def main():
    os.makedirs(RES, exist_ok=True)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    RSET = {"A": rig["chain_A"]["ultra_rigid"], "B": rig["chain_B"]["ultra_rigid"]}
    systems = sorted(os.path.basename(p) for p in glob.glob(f"{DATA}/*")
                     if len(os.path.basename(p)) == 4 and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    for i, s in enumerate(systems):
        try:
            print(f"[{i+1}/{len(systems)}] {s}: {process(s, RSET)}", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:120]}", flush=True); traceback.print_exc()
    print("SWING_DONE", flush=True)


if __name__ == "__main__":
    main()
