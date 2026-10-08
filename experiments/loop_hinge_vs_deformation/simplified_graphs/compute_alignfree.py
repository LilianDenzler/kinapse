#!/usr/bin/env python
"""Batch the ALIGNMENT-FREE loop-motion split across all 22 TCRs (distances only, ultra-rigid reference):
  eps_deform = sqrt(mean Var d_LL)   (loop-loop -> pure deformation)
  eps_pose   = sqrt(mean Var d_LR)   (loop -> ultra-rigid -> rigid pose vs framework)
plus rigid-vs-corkscrew twist split. -> results_alignfree.json"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
from graph_build import load_md, CDR_RANGES, HERE
from alignment_free import analyze as af_analyze
import twist_test

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
tcrs = [os.path.basename(f)[:4] for f in sorted(glob.glob(f"{HERE}/results_swing/*.npz"))]
rig = json.load(open(f"{HERE}/rigid_framework.json"))
res = {}
for t in tcrs:
    try:
        tv, xyz, imap = load_md(t)
        row = {}
        for cdr in CDRS:
            ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
            lk = sorted(k for k in imap[ch] if lo <= k <= hi)
            loop = xyz[:, np.array([imap[ch][k] for k in lk])]
            R = xyz[:, np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])]
            ed, ep, _, _ = af_analyze(loop, R)
            twist_test.CLAMP_now = twist_test.CLAMP[cdr[2:]]
            ct, it_, fr, _ = twist_test.analyze(loop, R)
            row[cdr] = dict(eps_deform=ed, eps_pose=ep, twist_rigid=ct, twist_deform=it_, twist_frac_rigid=fr)
        res[t] = row; print(f"{t} ok", flush=True)
    except Exception as e:
        print(f"{t} ERR {type(e).__name__}: {str(e)[:90]}", flush=True)
json.dump(res, open(f"{HERE}/results_alignfree.json", "w"))
print(f"\n{len(res)} TCRs -> results_alignfree.json")
