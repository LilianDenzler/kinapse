#!/usr/bin/env python
"""Graph-method MD pipeline: apply the joint hinge+deformation decomposition + single-hinge validity to the
22 TCRs, PER CHAIN, PER CDR. Emits the per-CDR fingerprint + validity table, plus per-CDR q(t) (for F(q)) and
the consensus hinge axis u* (for cross-TCR conservation). Alignment only for GPA rotation extraction; the
decomposition is distance-space. Resumable.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import argparse, glob, json, tempfile, warnings, traceback
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, mdtraj as md
from kinapse.structures import load_tcr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as PC
import graph_fit as GF, graph_decomp as GD, hinge_validity as HV
from consensus import load_consensus

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CORE_MIN, FLANK = 6, 20
CONSENSUS = load_consensus()          # curated rigid framework IMGT residues per chain (excludes mobile FR loops)


def process(sysid):
    pdb, xtc = f"{PC.DATA}/{sysid}/{sysid}.pdb", f"{PC.DATA}/{sysid}/{sysid}.xtc"
    tmp = tempfile.mktemp(suffix=".xtc")
    try:
        md.load(xtc, top=pdb, stride=PC.STRIDE).save_xtc(tmp)
        kw = {"manual_chain_types": PC.CHAIN_OVERRIDES[sysid]} if sysid in PC.CHAIN_OVERRIDES else {}
        tv = load_tcr(pdb, traj=tmp, **kw).pairs[0].traj
        xyz = tv.mdtraj.xyz * 10.0
        anchors = {}
        for ch in "AB":
            vidx, vnames = tv.domain_idx([f"{ch}_variable"], atom_names={"CA"}, pass_names=True)
            sel = [(int(i), int(n[1])) for i, n in zip(vidx, vnames) if int(n[1]) in CONSENSUS[ch]]
            if len(sel) < CORE_MIN:
                continue
            anchors[ch] = (np.array([i for i, _ in sel]), np.array([m for _, m in sel]))   # consensus rigid anchors
        rows, saved = [], {}
        for cdr in PC.CDRS:
            ch = PC.CHAIN_OF[cdr]
            if ch not in anchors:
                continue
            g, a_imgt = anchors[ch]
            lidx, lnames = tv.domain_idx([cdr], atom_names={"CA"}, pass_names=True)
            lidx = np.asarray(lidx)
            if len(lidx) < 3:
                continue
            lo, hi = int(lnames[0][1]), int(lnames[-1][1])
            anchor_ok = bool(((a_imgt >= lo - FLANK) & (a_imgt < lo)).any() and ((a_imgt > hi) & (a_imgt <= hi + FLANK)).any())
            loop, fw = xyz[:, lidx], xyz[:, g]
            r = GD.run_cdr(loop, fw)
            v = HV.single_hinge_report(loop, fw)
            rows.append(dict(
                system=sysid, cdr=cdr, chain=ch, loop_length=int(len(lidx)), n_anchor=int(len(g)), anchor_ok=anchor_ok,
                # intrinsic (LL) + framework-relative (LF) fingerprint
                F_deform=round(r["F_deform"], 3), F_hinge=round(r["F_hinge"], 3),
                F_deform_LF=round(r["F_deform_LF"], 3), F_residual=round(r["F_residual"], 3),
                f_hinge=round(r["f_hinge"], 3), f_deform_LF=round(r["f_deform_LF"], 3),
                f_coupling=round(r["f_coupling"], 3), f_residual=round(r["f_residual"], 3),
                n_deform_modes=r["n_deform_modes"],
                theta_rms_deg=round(float(np.sqrt((r["theta_deg"] ** 2).mean())), 2),
                theta_p95_deg=round(float(np.percentile(np.abs(r["theta_deg"]), 95)), 2),
                # single-hinge validity
                P_1D=round(v["P_1D"], 3), r_perp=round(v["r_perp"], 3), delta_rms_deg=round(v["delta_rms_deg"], 2),
                dE_axis=round(v["dE_axis"], 3), median_alpha_deg=round(v["median_alpha_deg"], 1),
                block_axis_agree_deg=round(v["block_axis_agree_deg"], 1),
            ))
            saved[f"{cdr}__theta_deg"] = r["theta_deg"].astype(np.float32)
            saved[f"{cdr}__u"] = r["u"].astype(np.float32)
            saved[f"{cdr}__p"] = r["p"].astype(np.float32)
        pd.DataFrame(rows).to_csv(f"{RES}/{sysid}_graph.csv", index=False)
        np.savez_compressed(f"{RES}/{sysid}_graph.npz", **saved)
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
        if os.path.exists(f"{RES}/{s}_graph.csv") and not a.force:
            print(f"[{s}] done", flush=True); continue
        try:
            print(f"[{i+1}/{len(systems)}] {s}: {process(s)} CDRs", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:150]}", flush=True); traceback.print_exc()
    print("GRAPH_DECOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
