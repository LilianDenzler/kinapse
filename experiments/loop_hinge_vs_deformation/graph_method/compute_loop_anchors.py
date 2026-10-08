#!/usr/bin/env python
"""Per-CDR LOOP-ANCHOR stability, alignment-free (pairwise Ca distances only).

Each CDR has two flanking framework anchor residues (IMGT): CDR1=26/39, CDR2=55/66, CDR3=104/118.
Per TCR, per chain, over the trajectory, save:
  - the 6 anchor Ca pairwise distance matrix (mean, std over frames) with IMGT labels;
  - the 3 anchor-pair midpoints and their 3 mutual distances (mean, std)  [relation of the 3 loop bases].
From this the plots answer: (1) do a loop's two anchors vary within one MD? (2) across TCRs? (3) do the three
anchor pairs move relative to each other within one MD? (4) across TCRs? Resumable.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob, tempfile, warnings
warnings.simplefilter("ignore")
import numpy as np, mdtraj as md
from kinapse.structures import load_tcr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as PC
import graph_fit as GF

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_loopanchor")
os.makedirs(RES, exist_ok=True)
FR = {"A": ["A_FR1", "A_FR2", "A_FR3", "A_FR4"], "B": ["B_FR1", "B_FR2", "B_FR3", "B_FR4"]}
PAIRS = {"CDR1": (26, 39), "CDR2": (55, 66), "CDR3": (104, 118)}          # flanking anchor IMGT per CDR
ANCH_IMGT = [26, 39, 55, 66, 104, 118]                                     # 6 anchor points, ordered


def process(sysid):
    pdb, xtc = f"{PC.DATA}/{sysid}/{sysid}.pdb", f"{PC.DATA}/{sysid}/{sysid}.xtc"
    tmp = tempfile.mktemp(suffix=".xtc")
    try:
        md.load(xtc, top=pdb, stride=PC.STRIDE).save_xtc(tmp)
        kw = {"manual_chain_types": PC.CHAIN_OVERRIDES[sysid]} if sysid in PC.CHAIN_OVERRIDES else {}
        tv = load_tcr(pdb, traj=tmp, **kw).pairs[0].traj
        xyz = tv.mdtraj.xyz * 10.0
        out = {}
        for ch in "AB":
            fr_idx, fr_names = tv.domain_idx(FR[ch], atom_names={"CA"}, pass_names=True)
            imap = {int(n[1]): int(i) for i, n in zip(fr_idx, fr_names)}          # IMGT -> global atom idx
            present = [a for a in ANCH_IMGT if a in imap]
            if len(present) < 4:
                continue
            A = xyz[:, [imap[a] for a in present]]                                # (T, k, 3) anchor coords
            D = GF.dmat(A)                                                        # (T,k,k)
            out[f"{ch}_imgt"] = np.array(present)
            out[f"{ch}_mean"] = D.mean(0).astype(np.float32)
            out[f"{ch}_std"] = D.std(0).astype(np.float32)
            # 3 loop-base midpoints (only for CDRs whose BOTH anchors are present) + their mutual distances
            mids, labels = [], []
            for cdr, (n, c) in PAIRS.items():
                if n in imap and c in imap:
                    mids.append((xyz[:, imap[n]] + xyz[:, imap[c]]) / 2.0)        # (T,3)
                    labels.append(cdr)
            if len(mids) >= 2:
                Mid = np.stack(mids, 1)                                           # (T, nmid, 3)
                Dm = GF.dmat(Mid)
                out[f"{ch}_mid_labels"] = np.array(labels)
                out[f"{ch}_mid_mean"] = Dm.mean(0).astype(np.float32)
                out[f"{ch}_mid_std"] = Dm.std(0).astype(np.float32)
        out["n_frames"] = len(xyz)
        np.savez_compressed(f"{RES}/{sysid}.npz", **out)
        return 1
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    systems = sys.argv[1:] or sorted(os.path.basename(p) for p in glob.glob(f"{PC.DATA}/*")
                                     if len(os.path.basename(p)) == 4
                                     and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    for i, s in enumerate(systems):
        if os.path.exists(f"{RES}/{s}.npz"):
            print(f"[{s}] done", flush=True); continue
        try:
            process(s); print(f"[{i+1}/{len(systems)}] {s} ok", flush=True)
        except Exception as e:
            import traceback; print(f"[{s}] ERR {type(e).__name__}: {str(e)[:120]}", flush=True); traceback.print_exc()
    print("LOOPANCHOR_DONE", flush=True)


if __name__ == "__main__":
    main()
