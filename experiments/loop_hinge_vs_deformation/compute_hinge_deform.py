#!/usr/bin/env python
"""Alignment-free hinge-vs-deformation decomposition of TCR CDR loops over MD.

Per system, per CDR loop (relative to the ensemble-mean reference):
  D_deform     RMS change of intra-loop Ca-Ca distances         (pure shape, SE(3)-invariant)
  D_deform_sep same, |i-j|>=SEQ_SEP pairs only
  D_framework  RMS change of loop->rigid-core Ca-Ca distances    (reorientation PROXY, invariant)
  theta        best-fit rigid rotation of the loop in the chain's rigid-core frame  [deg]  (hinge)
  E_nonrigid   loop RMSD residual after that rigid fit           [A]   (deformation the hinge can't explain)
  dtheta       ~ E_nonrigid / lever-arm                          [deg] (honest hinge error bar)
  total_disp   framework-frame loop RMSD to the mean             [A]
  hinge_frac   1 - E_nonrigid^2/total_disp^2                     (fraction of motion a rigid hinge explains)

The rigid core = maximal subset of a chain's framework (FR1-3) Ca whose pairwise distances barely
fluctuate (std <= CORE_STD_A). Only the theta *solver* touches coordinates (one Procrustes in the
core frame); D_deform / D_framework are computed straight from distances. See plan.md. Resumable.
"""
from __future__ import annotations
import os, sys

# user-site has an ABI-broken h5py that crashes MDAnalysis/mdtraj -> re-exec without it
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse, glob, json, tempfile, traceback, warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import mdtraj as md
from kinapse.structures import load_tcr

import config as C


# ---------------------------------------------------------------- geometry --
def dist_mat(X):
    """(F,N,3) -> (F,N,N) pairwise Euclidean distances."""
    return np.linalg.norm(X[:, :, None, :] - X[:, None, :, :], axis=-1)


def cross_dist(X, A):
    """(F,N,3),(F,M,3) -> (F,N,M) loop-to-anchor distances."""
    return np.linalg.norm(X[:, :, None, :] - A[:, None, :, :], axis=-1)


def kabsch_fit(P, q_ref):
    """Best-fit rigid transform of each frame's P (F,M,3) onto a single target q_ref (M,3).
    Returns (R (F,3,3), Pbar (F,1,3), qbar (3,)); apply to any Y as: (Y-Pbar)@R + qbar
    (row-vector convention x' = x R, matching kinapse's superpose_to_mean)."""
    Pbar = P.mean(1, keepdims=True)
    qbar = q_ref.mean(0)
    Pc = P - Pbar
    Qc = q_ref - qbar
    H = np.einsum("fni,nj->fij", Pc, Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.matmul(U, Vt)))
    Dm = np.repeat(np.eye(3)[None], len(P), 0)
    Dm[:, 2, 2] = d
    R = np.matmul(np.matmul(U, Dm), Vt)
    return R, Pbar, qbar


def apply_rt(Y, Pbar, R, qbar):
    return np.einsum("fki,fij->fkj", Y - Pbar, R) + qbar


def rigid_core(A, std_thresh, min_atoms):
    """Maximal subset of framework Ca (A: F,n,3) with all pairwise distance std <= std_thresh.
    Greedy: drop the atom with the largest summed pairwise std until the constraint holds."""
    n = A.shape[1]
    std = dist_mat(A).std(0)                      # (n,n)
    keep = np.ones(n, bool)
    while keep.sum() > min_atoms:
        idx = np.where(keep)[0]
        sub = std[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub, 0.0)
        if sub.max() <= std_thresh:
            break
        keep[idx[sub.sum(1).argmax()]] = False
    idx = np.where(keep)[0]
    sub = std[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, 0.0)
    return idx, float(sub.max() if len(idx) > 1 else 0.0)


def rmsf(Xaln):
    """(F,N,3) framework-aligned -> per-atom RMSF (A)."""
    return np.sqrt(((Xaln - Xaln.mean(0)) ** 2).sum(-1).mean(0))


def hinge_fit(loop_scaf):
    """loop_scaf (F,N,3) in the rigid-core frame. Fit the reference loop shape onto each frame.
    Reference = the MEDOID frame (the real conformation closest to the mean shape) -- NOT the mean
    of coordinates, which would shrink a rotating loop toward its axis and inject spurious
    deformation into E_nonrigid. Returns per-frame arrays: theta[deg], axis(F,3), E_nonrigid[A],
    dtheta[deg], total_disp[A], hinge_frac, and the medoid index."""
    F, N, _ = loop_scaf.shape
    D = dist_mat(loop_scaf)                         # (F,N,N), rigid-invariant
    Dbar = D.mean(0)
    med = int(((D - Dbar[None]) ** 2).reshape(F, -1).sum(1).argmin())
    X0 = loop_scaf[med]                             # real, undistorted reference (shape + pose)
    X0c = X0 - X0.mean(0)
    Xkc = loop_scaf - loop_scaf.mean(1, keepdims=True)
    H = np.einsum("ni,fnj->fij", X0c, Xkc)          # X0c^T Xk_c per frame
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.matmul(U, Vt)))
    Dm = np.repeat(np.eye(3)[None], F, 0)
    Dm[:, 2, 2] = d
    R = np.matmul(np.matmul(U, Dm), Vt)             # x' = X0c @ R  ~ Xkc
    fitted = np.einsum("ni,fij->fnj", X0c, R)
    E = np.sqrt(((fitted - Xkc) ** 2).sum(-1).mean(-1))          # deformation residual (A)
    tr = np.einsum("fii->f", R)
    theta = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
    axis = np.stack([R[:, 2, 1] - R[:, 1, 2],
                     R[:, 0, 2] - R[:, 2, 0],
                     R[:, 1, 0] - R[:, 0, 1]], axis=1)
    an = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.divide(axis, an, out=np.zeros_like(axis), where=an > 1e-9)
    # lever arm: RMS perpendicular distance of reference loop atoms from the (per-frame) axis
    proj = np.einsum("ni,fi->fn", X0c, axis)                     # (F,N)
    perp = X0c[None] - proj[:, :, None] * axis[:, None, :]       # (F,N,3)
    lever = np.sqrt((perp ** 2).sum(-1).mean(-1))                # (F,)
    dtheta = np.degrees(np.divide(E, lever, out=np.zeros_like(E), where=lever > 1e-6))
    total = np.sqrt(((loop_scaf - X0[None]) ** 2).sum(-1).mean(-1))   # framework-frame RMSD to mean
    hinge_frac = 1.0 - np.divide(E ** 2, total ** 2, out=np.zeros_like(E), where=total > 1e-6)
    hinge_frac = np.clip(hinge_frac, 0.0, 1.0)
    return dict(theta=theta, axis=axis, E_nonrigid=E, dtheta=dtheta,
                total_disp=total, hinge_frac=hinge_frac, medoid=med)


def rms_delta(M, seq_sep=None):
    """M: (F,a,b) distance stack. RMS change of each entry vs the frame-mean reference.
    If seq_sep given (square intra-loop case), only keep pairs with |i-j|>=seq_sep."""
    ref = M.mean(0)
    d = M - ref[None]
    if seq_sep is not None:
        a = M.shape[1]
        mask = np.abs(np.subtract.outer(np.arange(a), np.arange(a))) >= seq_sep
        d = d[:, mask]                              # (F, npairs)
        return np.sqrt((d ** 2).mean(-1))
    return np.sqrt((d ** 2).reshape(len(M), -1).mean(-1))


# ------------------------------------------------------------------- driver --
def process(sysid):
    pdb, xtc = f"{C.DATA}/{sysid}/{sysid}.pdb", f"{C.DATA}/{sysid}/{sysid}.xtc"
    tmp = tempfile.mktemp(suffix=".xtc")
    try:
        md.load(xtc, top=pdb, stride=C.STRIDE).save_xtc(tmp)
        kw = {}
        if sysid in C.CHAIN_OVERRIDES:
            kw["manual_chain_types"] = C.CHAIN_OVERRIDES[sysid]
        tv = load_tcr(pdb, traj=tmp, **kw).pairs[0].traj
        traj = tv.mdtraj
        xyz = traj.xyz * 10.0                       # (F, n_atoms, 3) Angstrom
        F = len(xyz)

        core_info, saved, percdr = {}, {}, []
        # per chain: rigid core (shared by that chain's CDRs)
        core_scaf_by_chain = {}
        for ch, fr_regions in C.FR_BY_CHAIN.items():
            fr_g = np.asarray(tv.domain_idx(fr_regions, atom_names={"CA"}))
            if len(fr_g) < C.CORE_MIN_ATOMS:
                core_info[ch] = dict(error="too few framework CA")
                continue
            A_fr = xyz[:, fr_g]                     # (F, n_fr, 3)
            sub, maxstd = rigid_core(A_fr, C.CORE_STD_A, C.CORE_MIN_ATOMS)
            core_g = fr_g[sub]
            A_core = xyz[:, core_g]
            Rc, Pbar, qbar = kabsch_fit(A_core, A_core[0])
            core_scaf = apply_rt(A_core, Pbar, Rc, qbar)
            rmsf_core = rmsf(core_scaf)
            core_info[ch] = dict(n_fr=int(len(fr_g)), n_core=int(len(core_g)),
                                 max_pair_std=round(maxstd, 3),
                                 rmsf_mean=round(float(rmsf_core.mean()), 3),
                                 rmsf_max=round(float(rmsf_core.max()), 3),
                                 core_atoms=core_g.tolist())
            core_scaf_by_chain[ch] = (core_g, Pbar, Rc, qbar)

        # per CDR
        for cdr in C.CDRS:
            ch = C.CHAIN_OF[cdr]
            if ch not in core_scaf_by_chain:
                continue
            core_g, Pbar, Rc, qbar = core_scaf_by_chain[ch]
            lidx = np.asarray(tv.domain_idx([cdr], atom_names={"CA"}))
            if len(lidx) < 3:
                continue
            Xl = xyz[:, lidx]                        # loop raw coords (F,N,3)
            Acore = xyz[:, core_g]                  # anchor raw coords (F,M,3)

            D_deform = rms_delta(dist_mat(Xl))
            D_deform_sep = rms_delta(dist_mat(Xl), seq_sep=C.SEQ_SEP)
            D_framework = rms_delta(cross_dist(Xl, Acore))

            loop_scaf = apply_rt(Xl, Pbar, Rc, qbar)   # loop in the chain's rigid-core frame
            h = hinge_fit(loop_scaf)
            H_dom = D_framework / (D_framework + D_deform + 1e-9)

            saved[f"{cdr}__D_deform"] = D_deform.astype(np.float32)
            saved[f"{cdr}__D_deform_sep"] = D_deform_sep.astype(np.float32)
            saved[f"{cdr}__D_framework"] = D_framework.astype(np.float32)
            saved[f"{cdr}__H"] = H_dom.astype(np.float32)
            saved[f"{cdr}__theta"] = h["theta"].astype(np.float32)
            saved[f"{cdr}__dtheta"] = h["dtheta"].astype(np.float32)
            saved[f"{cdr}__E_nonrigid"] = h["E_nonrigid"].astype(np.float32)
            saved[f"{cdr}__total_disp"] = h["total_disp"].astype(np.float32)
            saved[f"{cdr}__hinge_frac"] = h["hinge_frac"].astype(np.float32)
            saved[f"{cdr}__axis"] = h["axis"].astype(np.float32)

            percdr.append(dict(
                system=sysid, cdr=cdr, chain=ch, length=int(len(lidx)), n_core=int(len(core_g)),
                D_deform=round(float(D_deform.mean()), 3),
                D_deform_sep=round(float(D_deform_sep.mean()), 3),
                D_framework=round(float(D_framework.mean()), 3),
                theta=round(float(h["theta"].mean()), 2),
                theta_p95=round(float(np.percentile(h["theta"], 95)), 2),
                dtheta=round(float(h["dtheta"].mean()), 2),
                E_nonrigid=round(float(h["E_nonrigid"].mean()), 3),
                total_disp=round(float(h["total_disp"].mean()), 3),
                hinge_frac=round(float(h["hinge_frac"].mean()), 3),
                H=round(float(H_dom.mean()), 3),
            ))

        np.savez_compressed(f"{C.RESULTS}/{sysid}_perframe.npz", n_frames=F, **saved)
        with open(f"{C.RESULTS}/{sysid}_core.json", "w") as fh:
            json.dump(dict(system=sysid, n_frames=F, stride=C.STRIDE,
                           core_std_A=C.CORE_STD_A, chains=core_info), fh, indent=2)
        pd.DataFrame(percdr).to_csv(f"{C.RESULTS}/{sysid}_percdr.csv", index=False)
        return F, len(percdr)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(C.RESULTS, exist_ok=True)

    systems = args.systems or C.SYSTEMS or sorted(
        os.path.basename(p) for p in glob.glob(f"{C.DATA}/*")
        if len(os.path.basename(p)) == 4
        and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    print(f"{len(systems)} systems: {' '.join(systems)}", flush=True)

    for i, s in enumerate(systems):
        out = f"{C.RESULTS}/{s}_percdr.csv"
        if os.path.exists(out) and not args.force:
            print(f"[{s}] done", flush=True)
            continue
        try:
            F, n = process(s)
            print(f"[{i+1}/{len(systems)}] {s}: {F} frames, {n} CDRs", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:160]}", flush=True)
            traceback.print_exc()
    print("HINGE_DEFORM_DONE", flush=True)


if __name__ == "__main__":
    main()
