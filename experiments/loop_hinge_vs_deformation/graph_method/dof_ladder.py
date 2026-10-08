#!/usr/bin/env python
"""How many hinge axes does each CDR use?  Coordinate-space DOF ladder + rotation-axis spectrum.

Everything is measured in the LOCAL BASE frame (align the loop on its two flanking 3-residue beta-strand
patches, so the base is held fixed and we see the loop's motion RELATIVE to its own base).

For the loop CA cloud we fit, per frame, rigid reconstructions of increasing freedom and report the fraction
of the loop's coordinate variance each explains (1 - SS_resid/SS_total):
  f_tether   1 DOF : rotation about the PHYSICAL tether axis u_base=(c_R-c_L) through the base centre
  f_best     1 DOF : rotation about the DOMINANT rotation axis (PC1 of the rotation-vector cloud)
  f_ball     3 DOF : rotation about the base centre, ANY axis (spherical joint, no translation)
  f_free     6 DOF : full per-frame rigid fit (any rotation + translation)  == the rigid ceiling (~parent 0.85)
Plus the model-free axis count from the rotation-vector covariance:
  P1  = lambda1/sum         (fraction of rotation on the single dominant axis; ->1 means a clean 1-D hinge)
  P12 = (lambda1+lambda2)/sum
  ang = angle between the tether axis and the dominant rotation axis (deg)
  Fdef= free-rigid residual RMSD (the truly non-rigid / internal-deformation part), Angstrom

Interpretation: f_ball >> f_best  =>  the rigid reorientation needs >1 axis (multi-axis rock, not a 1-D hinge).
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
import config as PC, rotations as RO
from consensus import load_consensus

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_dof")
CON = load_consensus()
CDR_RANGES = {"A_CDR1": (27, 38), "A_CDR2": (56, 65), "A_CDR3": (105, 117),
              "B_CDR1": (27, 38), "B_CDR2": (56, 65), "B_CDR3": (105, 117)}


def fit_q(a, b, u):                       # closed-form best rotation angle about fixed axis u
    ap = a - (a @ u)[..., None] * u; bp = b - (b @ u)[..., None] * u; A = np.broadcast_to(ap, bp.shape)
    return np.arctan2((np.cross(np.broadcast_to(u, bp.shape), A) * bp).sum((-1, -2)), (A * bp).sum((-1, -2)))


def rot_about(a, q, u):                   # Rodrigues rotation of a (T,N,3) by q(T) about axis u
    c = np.cos(q)[:, None, None]; s = np.sin(q)[:, None, None]
    return c * a + s * np.cross(np.broadcast_to(u, a.shape), a) + (1 - c) * ((a @ u)[..., None] * u)


def frac(Lf, rec):
    tot = ((Lf - Lf.mean(0)) ** 2).sum(); return 1 - ((Lf - rec) ** 2).sum() / tot


def kabsch_about(A, B):                   # per-frame rotation A->B about the origin (3-DOF, no translation)
    H = np.einsum("ni,tnj->tij", A, B); U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.matmul(np.transpose(Vt, (0, 2, 1)), np.transpose(U, (0, 2, 1)))))
    D = np.repeat(np.eye(3)[None], len(B), 0); D[:, 2, 2] = d
    return np.matmul(np.matmul(np.transpose(Vt, (0, 2, 1)), D), np.transpose(U, (0, 2, 1)))


def process(sysid):
    pdb, xtc = f"{PC.DATA}/{sysid}/{sysid}.pdb", f"{PC.DATA}/{sysid}/{sysid}.xtc"
    tmp = tempfile.mktemp(suffix=".xtc")
    try:
        md.load(xtc, top=pdb, stride=PC.STRIDE).save_xtc(tmp)
        kw = {"manual_chain_types": PC.CHAIN_OVERRIDES[sysid]} if sysid in PC.CHAIN_OVERRIDES else {}
        tv = load_tcr(pdb, traj=tmp, **kw).pairs[0].traj
        xyz = tv.mdtraj.xyz * 10.0
        imap = {ch: {int(n[1]): int(i) for i, n in zip(*tv.domain_idx([f"{ch}_variable"], atom_names={"CA"}, pass_names=True))} for ch in "AB"}
        rows = []
        for cdr, (lo, hi) in CDR_RANGES.items():
            ch = cdr[0]; m = imap[ch]
            Lp = [p for p in (lo-4, lo-3, lo-2) if p in m and p in CON[ch]]
            Rp = [p for p in (hi+2, hi+3, hi+4) if p in m and p in CON[ch]]
            li = np.asarray(tv.domain_idx([cdr], atom_names={"CA"}))
            if len(Lp) < 2 or len(Rp) < 2 or len(li) < 3:
                continue
            patch = np.array([m[p] for p in Lp + Rp]); nL = len(Lp)
            Lf = RO.to_framework_frame(xyz[:, li], xyz[:, patch])
            Pf = RO.to_framework_frame(xyz[:, patch], xyz[:, patch])
            cL = Pf[:, :nL].mean(1); cR = Pf[:, nL:].mean(1)
            u = (cR - cL).mean(0); u /= np.linalg.norm(u); pivot = ((cL + cR) / 2).mean(0)
            tmpl, R, _ = RO.gpa(Lf); Rbar = RO.karcher_mean(R)
            Ref = Rbar.apply(tmpl - tmpl.mean(0)) + Lf.mean((0, 1))
            w = RO.relative_rotvecs(R); C = np.cov(w.T)
            ev, evec = np.linalg.eigh(C); ev = ev[::-1] / ev.sum(); pc1 = evec[:, -1]; pc1 /= np.linalg.norm(pc1)
            ang = np.degrees(np.arccos(min(1.0, abs(u @ pc1))))
            A = Ref - pivot; Ab = np.broadcast_to(A, Lf.shape); B = Lf - pivot
            f_tether = frac(Lf, pivot + rot_about(Ab, fit_q(A, B, u), u))
            f_best = frac(Lf, pivot + rot_about(Ab, fit_q(A, B, pc1), pc1))
            Rb = kabsch_about(A, B); f_ball = frac(Lf, pivot + np.einsum("ni,tji->tnj", A, Rb))
            recf = np.einsum("ni,tji->tnj", tmpl - tmpl.mean(0), R); recf += Lf.mean(1, keepdims=True) - recf.mean(1, keepdims=True)
            f_free = frac(Lf, recf); Fdef = np.sqrt(((Lf - recf) ** 2).sum(-1).mean())
            rows.append(dict(system=sysid, cdr=cdr, loop_len=int(len(li)),
                             f_tether=round(f_tether, 3), f_best=round(f_best, 3), f_ball=round(f_ball, 3),
                             f_free=round(f_free, 3), P1=round(float(ev[0]), 3), P12=round(float(ev[0] + ev[1]), 3),
                             ang_tether_pc1=round(ang, 1), F_deform=round(Fdef, 3)))
        pd.DataFrame(rows).to_csv(f"{RES}/{sysid}_dof.csv", index=False)
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
        if os.path.exists(f"{RES}/{s}_dof.csv") and not a.force:
            print(f"[{s}] done", flush=True); continue
        try:
            print(f"[{i+1}/{len(systems)}] {s}: {process(s)} CDRs", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:150]}", flush=True); traceback.print_exc()
    print("DOF_DONE", flush=True)


if __name__ == "__main__":
    main()
