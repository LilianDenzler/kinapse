#!/usr/bin/env python
"""Batch the equilibrium-fluctuation elasticity fingerprint over all TCRs (for pooling).

For every CDR of every TCR: GLOBAL normalized-strain compliance/stiffness (Σ_q, K_eff,
soft modes, C_mean/k_soft/K_mean) + LOCAL discrete-rod stiffness profile (k_s,k_b,k_t) +
the harmonic-basin (multi-basin) flag. All from Cα geometry, alignment-free.

MD loading/treatment is IDENTICAL to the rest of simplified_graphs: `graph_build.load_md`
(kinapse IMGT renumbering, nm→Å, stride 25), and the loop is selected the same way as
compute_geom.py — `lk = sorted(k for k in imap[ch] if lo≤k≤hi)` — which keeps IMGT
insertion residues (fractional keys like 112.1) instead of dropping them.

Run:  mamba activate kinapse
      python elastic_fluct.py                 # gate first (must pass)
      python compute_elastic_fluct.py [--force] [--systems 3QH3 8YJ3]
Resumable: skips systems whose <ID>_elastic.npz already exists (unless --force).
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":                 # kinapse env must not see user site-packages
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import argparse, glob, warnings, traceback
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import elastic_fluct as EF
from graph_build import load_md, CDR_RANGES, DATA, HERE

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
RES = f"{HERE}/results_elastic"


def process(sysid):
    tv, xyz, imap = load_md(sysid)
    rows, blob = [], {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)         # insertion-safe (fractional keys kept)
        P = xyz[:, np.array([imap[ch][k] for k in lk])]           # (T,N,3) Å loop Cα
        sc, ar = EF.analyse_loop(P)
        sc.update(system=sysid, cdr=cdr, T=int(P.shape[0]))
        rows.append(sc)
        blob[f"{cdr}__resnums"] = np.array(lk, dtype=np.float32)
        for k, v in ar.items():
            blob[f"{cdr}__{k}"] = v
        print(f"  {sysid} {cdr}: N={sc['N']} C_mean={sc['C_mean']:.4f} k_soft={sc['k_soft']:.3g} "
              f"eff_modes={sc['eff_modes']:.1f} bc={sc['soft_mode_bc']:.2f}"
              f"{' MULTI-BASIN' if sc['multibasin'] else ''}", flush=True)
    os.makedirs(RES, exist_ok=True)
    np.savez_compressed(f"{RES}/{sysid}_elastic.npz", **blob)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/{sysid}_percdr.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    systems = args.systems or sorted(
        os.path.basename(p) for p in glob.glob(f"{DATA}/*")
        if len(os.path.basename(p)) == 4 and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    print(f"elastic-fluct compute: {len(systems)} systems\n", flush=True)
    for i, s in enumerate(systems):
        if os.path.exists(f"{RES}/{s}_elastic.npz") and not args.force:
            print(f"[{s}] exists, skip", flush=True); continue
        try:
            process(s); print(f"[{i+1}/{len(systems)}] {s} done", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:150]}", flush=True); traceback.print_exc()
    # (re)build the master table from every per-system csv present
    skip = {"master_percdr.csv", "summary_percdr.csv"}
    csvs = [f for f in sorted(glob.glob(f"{RES}/*_percdr.csv")) if os.path.basename(f) not in skip]
    if csvs:
        master = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
        master.to_csv(f"{RES}/master_percdr.csv", index=False)
        print(f"\nmaster_percdr.csv: {len(master)} rows ({master.system.nunique()} systems)", flush=True)
    print("ELASTIC_DONE", flush=True)


if __name__ == "__main__":
    main()
