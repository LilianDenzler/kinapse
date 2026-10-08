"""Compute the three-formalism elasticity fingerprint per CDR loop, per system.

For every CDR of every TCR:
  Method 1  finite-strain field   -> per-residue shear / volumetric / non-affine (D2min)
  Method 2  elastic network (GNM) -> per-residue mobility; ANM soft-mode spectrum
  Method 3  data-driven stiffness -> compliance Sigma soft modes, K_eff, QH entropy

All metrics are INTRINSIC to the loop (referenced to its own d_LL medoid) => invariant
to rigid motion and immune to the merged-trajectory split-domain issue; no framework
anchoring or box gating is needed. Rigid motion is the zero-strain null space (see plan.md).

Guarded by the IMGT insertion-code preflight (assert_no_dropped_residues): a system whose
mdtraj CA selection loses an insertion residue is refused, never silently truncated.

Run:  mamba activate kinapse
      python compute_elasticity.py [--systems 3QH3 8YJ3] [--force] [--stride N]
Resumable: skips systems whose <ID>_elasticity.npz already exists (unless --force).
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse, glob, tempfile, traceback, warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import mdtraj as md
from kinapse.structures import load_tcr

import config as C
import elast as E
from preflight import assert_no_dropped_residues, discover


def load_system(sid, stride):
    pdb = os.path.join(C.DATA, sid, f"{sid}.pdb")
    xtc = os.path.join(C.DATA, sid, f"{sid}.xtc")
    kw = {"manual_chain_types": C.CHAIN_OVERRIDES[sid]} if sid in C.CHAIN_OVERRIDES else {}
    with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as tf:
        tmp = tf.name
    try:
        md.load(xtc, top=pdb, stride=stride).save_xtc(tmp)                 # stride first (memory)
        pair = load_tcr(pdb, traj=tmp, **kw).pairs[0]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return pair


def run_cdr(X):
    """X: (T,N,3) loop CA coords in Angstrom. Returns (scalars dict, arrays dict)."""
    T, N, _ = X.shape
    ndof = max(3 * N - 6, 1)
    ref = E.medoid_reference(X)
    Xref = X[ref]

    # --- Method 1: finite-strain field -------------------------------------
    neigh = E.neighborhoods(Xref, rc=C.STRAIN_RC, kmin=C.STRAIN_KMIN)
    sf = E.strain_field(Xref, X, neigh)                                    # each (T,N)
    shear_by_res = sf["shear"].mean(0)                                     # (N,)
    vol_by_res = sf["vol"].mean(0)
    d2_by_res = sf["d2min"].mean(0)
    shear_frame = np.sqrt((sf["shear"] ** 2).mean(1))                     # (T,) RMS over residues
    d2_frame = sf["d2min"].mean(1)                                         # (T,)
    hotspot = int(np.argmax(d2_by_res))

    # --- Method 2: elastic network (GNM) -----------------------------------
    G = E.gnm_kirchhoff(Xref, rc=C.GNM_RC)
    gnm_msf, _ = E.gnm_msf(G, kT=1.0)                                      # (N,) predicted mobility
    # observed alignment-free per-residue mobility = mean_j Var_t(d_ij)
    dfull = E.dmat(X)                                                      # (T,N,N)
    obs_mob = dfull.var(0).sum(1) / (N - 1)                                # (N,)
    gm = (gnm_msf - gnm_msf.mean()) / (gnm_msf.std() + 1e-12)
    om = (obs_mob - obs_mob.mean()) / (obs_mob.std() + 1e-12)
    gnm_obs_corr = float(np.mean(gm * om))                                 # shape agreement [-1,1]
    Hn = E.anm_hessian(Xref, rc=C.ANM_RC)
    anm_w = np.sort(np.linalg.eigvalsh(Hn))                                # 6 zeros then soft->stiff
    anm_soft = float(anm_w[6]) if len(anm_w) > 6 else float("nan")        # softest nonrigid eigenvalue

    # --- Method 3: data-driven stiffness -----------------------------------
    d, pairs = E.dll_series(X)                                             # (T,P)
    Sigma, dmean = E.compliance(d)
    wS, VS = E.soft_modes(Sigma)                                           # eigvals desc
    tot = wS[wS > 0].sum()
    soft_frac1 = float(wS[0] / tot) if tot > 0 else float("nan")          # % variance in softest mode
    res_foot = E.edge_mode_to_residue(VS[:, 0], pairs, N)                 # softest-mode residue footprint
    soft_pr = E.participation_ratio(res_foot)                             # low => localized (hinge-like)
    Keff = E.stiffness(Sigma, kT=C.KT, ridge=C.RIDGE)
    kmin = float(np.min(np.linalg.eigvalsh(Keff)))                        # softest stiffness eigenvalue
    entropy = E.qh_entropy_from_cov(Sigma, ndof=ndof)                     # QH entropy (nats), real DOF only

    scalars = dict(
        N=N, ref_frame=ref,
        shear_mean=float(shear_by_res.mean()), shear_max=float(shear_by_res.max()),
        vol_absmean=float(np.abs(vol_by_res).mean()),
        d2min_mean=float(d2_by_res.mean()), d2min_max=float(d2_by_res.max()),
        nonaffine_hotspot_idx=hotspot,
        gnm_obs_corr=gnm_obs_corr, anm_soft_eig=anm_soft,
        soft_frac1=soft_frac1, soft_participation=soft_pr,
        compliance_total=float(tot), stiffness_min=kmin, qh_entropy=entropy,
    )
    arrays = dict(
        shear_by_res=shear_by_res, vol_by_res=vol_by_res, d2min_by_res=d2_by_res,
        gnm_msf=gnm_msf, obs_mobility=obs_mob, soft_mode_residue=res_foot,
        shear_frame=shear_frame.astype(np.float32), d2min_frame=d2_frame.astype(np.float32),
        compliance_eig=wS[:min(len(wS), 10)],
    )
    return scalars, arrays


def process(sid, stride, force):
    npz = os.path.join(C.RESULTS, f"{sid}_elasticity.npz")
    if os.path.exists(npz) and not force:
        print(f"{sid}: exists, skip"); return None
    pair = load_system(sid, stride)
    assert_no_dropped_residues(pair)                                       # hard IMGT-insertion guard
    tv = pair.traj
    rows, blob = [], {}
    for cdr in C.CDRS:
        idx, names = tv.domain_idx([cdr], atom_names={"CA"}, pass_names=True)
        idx = np.asarray(idx)
        resnums = np.array([r for (_, r, _) in names])
        X = tv.mdtraj.xyz[:, idx, :] * 10.0                                # (T,N,3) Angstrom
        sc, ar = run_cdr(X)
        sc.update(system=sid, cdr=cdr, T=X.shape[0])
        rows.append(sc)
        blob[f"{cdr}__resnums"] = resnums
        for k, v in ar.items():
            blob[f"{cdr}__{k}"] = v
        print(f"  {sid} {cdr}: N={sc['N']} shear={sc['shear_mean']:.3f} "
              f"d2min={sc['d2min_mean']:.3f} soft_frac1={sc['soft_frac1']:.2f} "
              f"soft_PR={sc['soft_participation']:.1f} S={sc['qh_entropy']:.1f}")
    os.makedirs(C.RESULTS, exist_ok=True)
    np.savez_compressed(npz, **blob)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS, f"{sid}_percdr.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=C.STRIDE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    systems = args.systems or (C.SYSTEMS or discover())
    os.makedirs(C.RESULTS, exist_ok=True)
    print(f"elasticity compute: {len(systems)} systems, stride {args.stride}\n")
    all_df = []
    for sid in systems:
        try:
            df = process(sid, args.stride, args.force)
            if df is not None:
                all_df.append(df)
        except Exception as e:
            print(f"{sid}: FAIL {type(e).__name__}: {e}")
            traceback.print_exc()
    # (re)build the master table from every per-system csv present
    # (exclude the aggregate tables themselves, which also match *_percdr.csv)
    skip = {"master_percdr.csv", "summary_percdr.csv"}
    csvs = [f for f in sorted(glob.glob(os.path.join(C.RESULTS, "*_percdr.csv")))
            if os.path.basename(f) not in skip]
    if csvs:
        master = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
        master.to_csv(os.path.join(C.RESULTS, "master_percdr.csv"), index=False)
        print(f"\nmaster_percdr.csv: {len(master)} rows ({master.system.nunique()} systems)")


if __name__ == "__main__":
    main()
