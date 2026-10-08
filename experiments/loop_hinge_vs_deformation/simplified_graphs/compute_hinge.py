#!/usr/bin/env python
"""Per-frame alignment-free hinge/deformation stats (D_LL, D_LR, D_H) for every TCR, saved for pooling.
Reference and deformation regression B are fit PER TCR (each loop has its own shape/modes)."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob, warnings, traceback
warnings.simplefilter("ignore")
import numpy as np
from hinge_scatter import scatter_stats, CDRS
from graph_build import load_md, CDR_RANGES, DATA, HERE

RES = f"{HERE}/results_hinge"


def process(sysid, RSET):
    tv, xyz, imap = load_md(sysid)
    save = {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        ri = np.array([imap[ch][r] for r in RSET[ch] if r in imap[ch]])
        loop = xyz[:, np.array([imap[ch][k] for k in lk])]
        D_LL, D_LR, D_H, k = scatter_stats(loop, xyz[:, ri])
        save[f"{cdr}_LL"] = D_LL.astype(np.float32)
        save[f"{cdr}_LR"] = D_LR.astype(np.float32)
        save[f"{cdr}_H"] = D_H.astype(np.float32)
    np.savez_compressed(f"{RES}/{sysid}.npz", **save)
    return {cdr: round(float(np.median(save[f"{cdr}_H"])), 2) for cdr in CDRS}


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
    print("HINGE_DONE", flush=True)


if __name__ == "__main__":
    main()
