#!/usr/bin/env python
"""Open feature discovery: what flank/loop structural & sequence features govern the hinge / rigid-deform / DOF /
entropy of each CDR loop? Computes a broad feature panel per (TCR,CDR) from MD, to be correlated against the motion
targets (results_angleamp/rigid_deform/modes_atlas json). Inspired by Singh et al. 2024 (868 TCR G119P removes a
cross-stem backbone H-bond) — but the H-bond is just ONE feature among many here.
Stage 1 (this script): compute + save results_features.json.  Stage 2 (analyze()): merge targets + rank correlations."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
from scipy.spatial import cKDTree
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}
NSUB = 300
AROM = {"PHE", "TRP", "TYR", "HIS"}; CHG = {"ASP", "GLU", "LYS", "ARG"}
HPHOB = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-9)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1); y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)


def circvar(ang):
    return float(1 - np.abs(np.exp(1j * ang).mean()))


def bb_atoms(top, ca_idx):
    res = top.atom(int(ca_idx)).residue
    d = {a.name: a.index for a in res.atoms}
    return d, res.name


def feats_for_cdr(sysid, ch, cdr, supr, imap, top, rigset, rng):
    lo, hi = CDR_RANGES[cdr[2:]]
    lk = sorted(k for k in imap[ch] if lo <= k <= hi); Nres = len(lk)
    ca = np.array([imap[ch][k] for k in lk]); loop = supr[:, ca]; T = loop.shape[0]
    # reference geometry (medoid)
    ii, jj = np.triu_indices(Nres, 1); dLL = np.linalg.norm(loop[:, ii] - loop[:, jj], axis=-1)
    med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1))); L0 = loop[med]
    fN, fC = L0[0], L0[-1]; m = 0.5 * (fN + fC); cc = fC - fN; feet_len = float(np.linalg.norm(cc)); cc /= feet_len
    perp = (L0 - m) - ((L0 - m) @ cc)[:, None] * cc
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); tip_lever = float(np.linalg.norm(perp[ai]))
    rg = float(np.sqrt(((L0 - L0.mean(0)) ** 2).sum(1).mean()))
    clamp_i = Nres - 1 if CLAMP[cdr[2:]] == "C" else 0
    free_i = 0 if clamp_i == Nres - 1 else Nres - 1
    # roles by IMGT key
    Nfl = [k for k in imap[ch] if lo - 4 <= k < lo]; Cfl = [k for k in imap[ch] if hi < k <= hi + 4]
    loopset = set(lk); Nset, Cset = set(Nfl), set(Cfl)

    def role(k):
        if k in loopset: return "loop"
        if k in Nset: return "Nflank"
        if k in Cset: return "Cflank"
        return "fw"
    # sequence composition (loop)
    names = {k: bb_atoms(top, imap[ch][k])[1] for k in imap[ch]}
    loopn = [names[k] for k in lk]
    n_gly = sum(n == "GLY" for n in loopn); n_pro = sum(n == "PRO" for n in loopn)
    n_arom = sum(n in AROM for n in loopn); n_chg = sum(n in CHG for n in loopn)
    frac_hphob = float(np.mean([n in HPHOB for n in loopn]))
    jmotif_gly = int(any(names.get(k) == "GLY" for k in Cfl))        # 868-like Gly just C-term of CDR3
    gly_flank = int(any(names.get(k) == "GLY" for k in Nfl + Cfl)); pro_flank = int(any(names.get(k) == "PRO" for k in Nfl + Cfl))
    # flank backbone dihedral circular variance (subsampled frames)
    idx = rng.choice(T, min(NSUB, T), replace=False)
    flankkeys = Nfl + Cfl
    cvs = []
    allk = sorted(imap[ch])
    for k in flankkeys:
        if k - 1 not in imap[ch] or k + 1 not in imap[ch]:
            continue
        a_prevC = bb_atoms(top, imap[ch][k - 1])[0].get("C"); a = bb_atoms(top, imap[ch][k])[0]
        a_nextN = bb_atoms(top, imap[ch][k + 1])[0].get("N")
        if not all(x is not None for x in (a_prevC, a.get("N"), a.get("CA"), a.get("C"), a_nextN)):
            continue
        phi = dihedral(supr[idx, a_prevC], supr[idx, a["N"]], supr[idx, a["CA"]], supr[idx, a["C"]])
        psi = dihedral(supr[idx, a["N"]], supr[idx, a["CA"]], supr[idx, a["C"]], supr[idx, a_nextN])
        cvs += [circvar(phi), circvar(psi)]
    flank_dihvar = float(np.mean(cvs)) if cvs else float("nan")
    # backbone H-bonds (N...O < 3.5 A) among loop+flanks+chain, subsampled, occupancy by class
    chainkeys = [k for k in allk if 1 <= k <= 128]
    Nat = {k: bb_atoms(top, imap[ch][k])[0].get("N") for k in chainkeys}
    Oat = {k: bb_atoms(top, imap[ch][k])[0].get("O") for k in chainkeys}
    don = [(k, Nat[k]) for k in chainkeys if Nat[k] is not None]
    acc = [(k, Oat[k]) for k in chainkeys if Oat[k] is not None]
    dk = [k for k, _ in don]; di = np.array([a for _, a in don]); ak = [k for k, _ in acc]; aii = np.array([a for _, a in acc])
    cls = {"cross_stem": 0.0, "stem_fw": 0.0, "intra_loop": 0.0, "loop_fw": 0.0}
    for t in idx:
        Dp = supr[t, di]; Ap = supr[t, aii]
        tree = cKDTree(Ap)
        for p, k_d in enumerate(dk):
            for q in tree.query_ball_point(Dp[p], 3.5):
                k_a = ak[q]
                if abs(k_d - k_a) < 2:
                    continue
                r_d, r_a = role(k_d), role(k_a); pair = {r_d, r_a}
                if pair == {"Nflank", "Cflank"}: cls["cross_stem"] += 1
                elif ("Nflank" in pair or "Cflank" in pair) and "fw" in pair: cls["stem_fw"] += 1
                elif pair == {"loop"}: cls["intra_loop"] += 1
                elif "loop" in pair and "fw" in pair: cls["loop_fw"] += 1
    for c in cls:
        cls[c] /= len(idx)                                           # mean count per frame (occupancy-weighted)
    # dynamics: RMSF and apex-anchor DCCM (framework-superposed CA)
    def rmsf(cai):
        P = supr[:, cai]; return float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))
    clamp_rmsf = rmsf(ca[clamp_i]); free_rmsf = rmsf(ca[free_i]); tip_rmsf = rmsf(ca[ai])
    loop_rmsf = float(np.mean([rmsf(c) for c in ca]))
    # DCCM apex vs anchors (flank CAs + clamp foot)
    anchor_ca = [imap[ch][k] for k in (Nfl + Cfl) if k in imap[ch]] + [ca[clamp_i]]
    def disp(cai):
        P = supr[:, cai]; return P - P.mean(0)
    da = disp(ca[ai]); na = np.sqrt((da ** 2).sum(1).mean())
    dccs = []
    for cai in anchor_ca:
        db = disp(cai); nb = np.sqrt((db ** 2).sum(1).mean())
        if na > 1e-6 and nb > 1e-6:
            dccs.append(float((da * db).sum(1).mean() / (na * nb)))
    apex_anchor_dccm = float(np.mean(dccs)) if dccs else float("nan")
    return dict(N=Nres, feet_len=feet_len, tip_lever=tip_lever, rg=rg,
                n_gly=n_gly, n_pro=n_pro, n_arom=n_arom, n_chg=n_chg, frac_hphob=frac_hphob,
                jmotif_gly=jmotif_gly, gly_flank=gly_flank, pro_flank=pro_flank, flank_dihvar=flank_dihvar,
                hb_cross_stem=cls["cross_stem"], hb_stem_fw=cls["stem_fw"], hb_intra_loop=cls["intra_loop"], hb_loop_fw=cls["loop_fw"],
                clamp_rmsf=clamp_rmsf, free_rmsf=free_rmsf, tip_rmsf=tip_rmsf, loop_rmsf=loop_rmsf,
                apex_anchor_dccm=apex_anchor_dccm)


def main(only=None):
    tcrs = sorted(os.path.basename(f)[:4] for f in glob.glob(f"{HERE}/results_swing/*.npz"))
    if only:
        tcrs = [only]
    rig = json.load(open(f"{HERE}/rigid_framework.json")); rng = np.random.default_rng(0); res = {}
    for n, s in enumerate(tcrs):
        try:
            tv, xyz, imap = load_md(s); top = tv.mdtraj.topology; cache = {}; res[s] = {}
            for cdr in CDRS:
                ch = cdr[0]
                if ch not in cache:
                    fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
                    cache[ch] = superpose_all(xyz, fw)
                res[s][cdr] = feats_for_cdr(s, ch, cdr, cache[ch], imap, top, set(rig["chain_" + ch]["ultra_rigid"]), rng)
            b3 = res[s]["B_CDR3"]
            print(f"[{n+1}/{len(tcrs)}] {s}: B_CDR3 hb_cross={b3['hb_cross_stem']:.2f} dccm={b3['apex_anchor_dccm']:+.2f} "
                  f"flank_dihvar={b3['flank_dihvar']:.2f} jmotif_gly={b3['jmotif_gly']}", flush=True)
        except Exception as e:
            print(f"[{n+1}/{len(tcrs)}] {s}: FAILED {type(e).__name__}: {e}", flush=True)
    if not only:
        json.dump(res, open(f"{HERE}/results_features.json", "w"), indent=1, default=float)
        print("saved results_features.json", flush=True)
    else:
        print(json.dumps(res[only]["B_CDR3"], indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
