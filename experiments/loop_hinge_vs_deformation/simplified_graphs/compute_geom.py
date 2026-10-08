#!/usr/bin/env python
"""Batch the geometric rigid-fit decomposition (D_deform, D_rigid, E_nonrigid) over all TCRs, for pooling."""
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob, warnings, traceback
warnings.simplefilter("ignore")
import numpy as np
from geom_hinge import geom_decomp, CDRS
from graph_build import load_md, CDR_RANGES, DATA, HERE

RES = f"{HERE}/results_geom"


def process(s, RSET):
    tv, xyz, imap = load_md(s)
    save = {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        ri = np.array([imap[ch][r] for r in RSET[ch] if r in imap[ch]])
        Dd, Dr, E = geom_decomp(xyz[:, np.array([imap[ch][k] for k in lk])], xyz[:, ri])
        save[f"{cdr}_def"] = Dd.astype(np.float32)
        save[f"{cdr}_rig"] = Dr.astype(np.float32)
        save[f"{cdr}_E"] = E.astype(np.float32)
    np.savez_compressed(f"{RES}/{s}.npz", **save)


def main():
    os.makedirs(RES, exist_ok=True)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    RSET = {"A": rig["chain_A"]["ultra_rigid"], "B": rig["chain_B"]["ultra_rigid"]}
    systems = sorted(os.path.basename(p) for p in glob.glob(f"{DATA}/*")
                     if len(os.path.basename(p)) == 4 and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    for i, s in enumerate(systems):
        try:
            process(s, RSET); print(f"[{i+1}/{len(systems)}] {s} done", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:120]}", flush=True); traceback.print_exc()
    print("GEOM_DONE", flush=True)


if __name__ == "__main__":
    main()
