#!/usr/bin/env python
"""Per-TCR pairwise CA-CA distance fluctuation (D_std) over the variable domains, per chain.

For each TCR and chain: D_std[i,j] = std over MD frames of |CA_i - CA_j|, indexed by IMGT residue.
Saved per TCR so they can be AVERAGED across all TCRs (a transferable, system-independent fluctuation
profile / rigid-set definition). Alignment-free (distances only)."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob, warnings, traceback
warnings.simplefilter("ignore")
import numpy as np
from graph_build import load_md, DATA, HERE

RES = f"{HERE}/results_fluct"


def process(sysid):
    tv, xyz, imap = load_md(sysid)
    save = {}
    for ch in "AB":
        imgt = np.array(sorted(imap[ch])); ai = np.array([imap[ch][n] for n in imgt])
        P = xyz[:, ai]
        D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
        save[f"{ch}_imgt"] = imgt
        save[f"{ch}_Dstd"] = D.std(0).astype(np.float32)
    np.savez_compressed(f"{RES}/{sysid}_fluct.npz", **save)
    return len(save["A_imgt"]), len(save["B_imgt"])


def main():
    force = "--force" in sys.argv
    os.makedirs(RES, exist_ok=True)
    systems = sorted(os.path.basename(p) for p in glob.glob(f"{DATA}/*")
                     if len(os.path.basename(p)) == 4 and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    for i, s in enumerate(systems):
        if os.path.exists(f"{RES}/{s}_fluct.npz") and not force:
            print(f"[{s}] done", flush=True); continue
        try:
            na, nb = process(s); print(f"[{i+1}/{len(systems)}] {s}: A={na} B={nb} residues", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:150]}", flush=True); traceback.print_exc()
    print("FLUCT_DONE", flush=True)


if __name__ == "__main__":
    main()
