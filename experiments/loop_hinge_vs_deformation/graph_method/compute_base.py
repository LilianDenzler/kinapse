#!/usr/bin/env python
"""Local-base CDR analysis (patch centroids) + fixed-global vs local-dynamic-base vs rigid-free hinge.

For each CDR loop (IMGT lo..hi) define flanking beta-strand PATCHES a few residues into the framework:
  L = {lo-4,lo-3,lo-2}, R = {hi+2,hi+3,hi+4}  (kept only if consensus rigid residues).
Base points = patch centroids c_L(t), c_R(t); base axis u_base(t)=(c_R-c_L)/||.||.
Report per CDR:
  base_width_mean/std        = ||c_R-c_L|| stability (much tighter than single anchors: centroid averaging)
  axis_dir_std_deg           = std of angle(u_base(t), mean u_base) in the FRAMEWORK frame (axis-direction stability)
  patch_rmsf                 = local base patch rigidity (framework-aligned RMSF)
  f_fixed_global             = run_cdr(loop, CONSENSUS framework).f_hinge   (single axis fixed in the framework)
  f_local_base               = run_cdr(loop, PATCH residues).f_hinge         (hinge relative to the local base)
  f_rigid_free               = any-axis rigid baseline
Interpretation: f_local_base >> f_fixed_global  =>  1D hinge mounted on a moving local base.
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
import graph_decomp as GD, rotations as RO
from consensus import load_consensus

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_base")
CONSENSUS = load_consensus()
CDR_RANGES = {"A_CDR1": (27, 38), "A_CDR2": (56, 65), "A_CDR3": (105, 117),
              "B_CDR1": (27, 38), "B_CDR2": (56, 65), "B_CDR3": (105, 117)}


def process(sysid):
    pdb, xtc = f"{PC.DATA}/{sysid}/{sysid}.pdb", f"{PC.DATA}/{sysid}/{sysid}.xtc"
    tmp = tempfile.mktemp(suffix=".xtc")
    try:
        md.load(xtc, top=pdb, stride=PC.STRIDE).save_xtc(tmp)
        kw = {"manual_chain_types": PC.CHAIN_OVERRIDES[sysid]} if sysid in PC.CHAIN_OVERRIDES else {}
        tv = load_tcr(pdb, traj=tmp, **kw).pairs[0].traj
        xyz = tv.mdtraj.xyz * 10.0
        imap = {}                                      # per chain: IMGT -> CA atom index
        fw_idx = {}
        for ch in "AB":
            vi, vn = tv.domain_idx([f"{ch}_variable"], atom_names={"CA"}, pass_names=True)
            m = {int(n[1]): int(i) for i, n in zip(vi, vn)}
            imap[ch] = m
            fw_idx[ch] = np.array([m[a] for a in sorted(CONSENSUS[ch]) if a in m])
        rows = []; mids = {}
        for cdr, (lo, hi) in CDR_RANGES.items():
            ch = cdr[0]; m = imap[ch]
            Lp = [p for p in (lo-4, lo-3, lo-2) if p in m and p in CONSENSUS[ch]]
            Rp = [p for p in (hi+2, hi+3, hi+4) if p in m and p in CONSENSUS[ch]]
            li = np.asarray(tv.domain_idx([cdr], atom_names={"CA"}))
            if len(Lp) < 2 or len(Rp) < 2 or len(li) < 3:
                continue
            loop = xyz[:, li]
            patch_idx = np.array([m[p] for p in Lp + Rp])
            fw = xyz[:, fw_idx[ch]]
            # base metrics in the framework frame
            Lf = RO.to_framework_frame(xyz[:, patch_idx], fw)          # patches in framework frame
            nL = len(Lp)
            cL = Lf[:, :nL].mean(1); cR = Lf[:, nL:].mean(1)
            dLR = np.linalg.norm(cR - cL, axis=1)
            ub = (cR - cL) / dLR[:, None]; ubar = ub.mean(0); ubar /= np.linalg.norm(ubar)
            ang = np.degrees(np.arccos(np.clip(ub @ ubar, -1, 1)))
            patch_rmsf = float(np.sqrt(((Lf - Lf.mean(0)) ** 2).sum(-1).mean(0)).mean())
            mids[cdr] = (cL + cR) / 2.0                                # loop-base midpoint (framework frame)
            # hinge fractions
            rg = GD.run_cdr(loop, fw)                                  # fixed global axis + rigid-free
            rl = GD.run_cdr(loop, xyz[:, patch_idx])                   # hinge relative to the local base
            rows.append(dict(
                system=sysid, cdr=cdr, chain=ch, loop_length=int(len(li)),
                base_width_mean=round(float(dLR.mean()), 3), base_width_std=round(float(dLR.std()), 3),
                axis_dir_std_deg=round(float(ang.std()), 2), patch_rmsf=round(patch_rmsf, 3),
                f_fixed_global=round(rg["f_hinge"], 3), f_local_base=round(rl["f_hinge"], 3),
                f_rigid_free=round(rg["f_rigid_free"], 3),
                Lp="/".join(map(str, Lp)), Rp="/".join(map(str, Rp)),
            ))
        pd.DataFrame(rows).to_csv(f"{RES}/{sysid}_base.csv", index=False)
        # per-chain: the 3 loop-base midpoints' mutual distances (relation of the 3 patch bases)
        save = {}
        for ch in "AB":
            cc = [c for c in CDR_RANGES if c[0] == ch and c in mids]
            if len(cc) >= 2:
                Mid = np.stack([mids[c] for c in cc], 1)              # (T, k, 3)
                Dm = np.linalg.norm(Mid[:, :, None, :] - Mid[:, None, :, :], axis=-1)
                save[f"{ch}_labels"] = np.array([c[2:] for c in cc])
                save[f"{ch}_mid_mean"] = Dm.mean(0).astype(np.float32)
                save[f"{ch}_mid_std"] = Dm.std(0).astype(np.float32)
        np.savez_compressed(f"{RES}/{sysid}_mid.npz", **save)
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
        if os.path.exists(f"{RES}/{s}_base.csv") and not a.force:
            print(f"[{s}] done", flush=True); continue
        try:
            print(f"[{i+1}/{len(systems)}] {s}: {process(s)} CDRs", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:150]}", flush=True); traceback.print_exc()
    print("BASE_DONE", flush=True)


if __name__ == "__main__":
    main()
