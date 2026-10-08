#!/usr/bin/env python
"""TORSION-SPACE fixed-boundary elastic-network model of a single CDR loop (iMod/ilmode recipe).
Degrees of freedom = backbone phi/psi dihedrals of the INTERIOR loop residues (not the feet).
Elastic-network potential on backbone atoms (parameter-free: all pairs, spring = 1/d^2 -> no cutoff, avoids the
'tip effect'). Both feet clamped: the N-foot is upstream of all variable dihedrals (fixed automatically); the
C-foot is held by a LOOP-CLOSURE constraint (its backbone atoms must not move) -> work in the null space of that
constraint. Equilibrium per-residue deformation = diag of the dihedral-covariance pushed to Cartesian (kT K^-1),
compared to the MD deformation profile. Mode shapes = soft constrained torsional modes.
-> figures/anm_torsion_<sys>.png + printed correlations.  No masses (equilibrium fluctuations are mass-independent)."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy.stats import pearsonr
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all

BB = ("N", "CA", "C")


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1); c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


def backbone_indices(top, ca_idx_per_res):
    """return (3*Nres) atom indices ordered N,CA,C per residue, or None if any missing."""
    idx = []
    for ca in ca_idx_per_res:
        res = top.atom(int(ca)).residue
        names = {a.name: a.index for a in res.atoms}
        if not all(n in names for n in BB):
            return None
        idx += [names[n] for n in BB]
    return np.array(idx)


def pf_hessian(X):
    """parameter-free ANM Cartesian Hessian (all pairs, gamma = 1/d^2)."""
    n = len(X); H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            d = X[j] - X[i]; r = np.linalg.norm(d)
            if r < 1e-6:
                continue
            e = d / r; k = (1.0 / r ** 2) * np.outer(e, e)
            H[3*i:3*i+3, 3*j:3*j+3] -= k; H[3*j:3*j+3, 3*i:3*i+3] -= k
            H[3*i:3*i+3, 3*i:3*i+3] += k; H[3*j:3*j+3, 3*j:3*j+3] += k
    return H


def lever_arm(X, Nres):
    """J (3*natoms x ndih): Cartesian displacement of each atom per unit dihedral rotation.
    Dihedrals = phi_i (bond N_i-CA_i) and psi_i (bond CA_i-C_i) for interior residues i=1..Nres-2.
    Atom local layout: residue r -> N=3r, CA=3r+1, C=3r+2."""
    nat = 3 * Nres; cols = []; meta = []
    for i in range(1, Nres - 1):
        Ni, CAi, Ci = 3 * i, 3 * i + 1, 3 * i + 2
        # phi_i: axis N_i->CA_i, pivot CA_i, downstream = C_i and all atoms of residues > i
        for name, (a0, a1, down) in {
            "phi": (Ni, CAi, [Ci] + list(range(3 * (i + 1), nat))),
            "psi": (CAi, Ci, list(range(3 * (i + 1), nat))),
        }.items():
            axis = X[a1] - X[a0]; nrm = np.linalg.norm(axis)
            if nrm < 1e-6:
                continue
            axis = axis / nrm; piv = X[a1]
            col = np.zeros(nat * 3)
            for k in down:
                col[3*k:3*k+3] = np.cross(axis, X[k] - piv)
            cols.append(col); meta.append((i, name))
    return (np.array(cols).T if cols else np.zeros((nat * 3, 0))), meta


def torsion_modes(Xbb, Nres):
    """return per-atom predicted fluctuation (nat,) and (Kc eigen) for the closed-loop torsional ENM."""
    H = pf_hessian(Xbb)
    J, meta = lever_arm(Xbb, Nres)
    if J.shape[1] == 0:
        return np.zeros(len(Xbb)), 0, 0
    K = J.T @ H @ J                                              # torsional stiffness
    # loop-closure: C-foot (last residue) backbone atoms must not move
    cfoot = [3 * (Nres - 1) + a for a in range(3)]              # local atom indices of N,CA,C of C-foot
    rows = np.array([3 * k + a for k in cfoot for a in range(3)])
    Ccon = J[rows, :]                                            # (9 x ndih) closure constraint
    # null space of the constraint (allowed dihedral motions keep the C-foot fixed)
    u, s, vt = np.linalg.svd(Ccon)
    tol = max(Ccon.shape) * (s.max() if s.size else 0) * 1e-10
    rank = int((s > tol).sum())
    Z = vt[rank:].T                                             # (ndih x ndof) null-space basis
    ndof = Z.shape[1]
    if ndof == 0:
        return np.zeros(len(Xbb)), 0, 0
    Kc = Z.T @ K @ Z
    w, V = np.linalg.eigh(Kc)
    keep = w > 1e-9 * w.max()
    Cinv_c = (V[:, keep] / w[keep]) @ V[:, keep].T              # constrained dihedral covariance (kT=1)
    Sigma_theta = Z @ Cinv_c @ Z.T
    # Cartesian covariance diagonal -> per-atom mean-square fluctuation
    JT = J @ Sigma_theta                                        # (3nat x ndih)
    var = np.einsum("ij,ij->i", JT, J)                          # diag(J Sigma J^T)
    fluct = var.reshape(-1, 3).sum(1)                           # per-atom
    return fluct, ndof, int(keep.sum())


def md_deform(loop_ca):
    T, N, _ = loop_ca.shape
    ii, jj = np.triu_indices(N, 1); dLL = np.linalg.norm(loop_ca[:, ii] - loop_ca[:, jj], axis=-1)
    med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1))); L0 = loop_ca[med]
    acc = np.zeros(N)
    for t in range(T):
        R, Pc, Qc = robust_kabsch(L0, loop_ca[t]); rig = (L0 - Pc) @ R.T + Qc
        acc += ((loop_ca[t] - rig) ** 2).sum(1)
    return np.sqrt(acc / T), med


def main(sysid="3SKN", cdrs=("B_CDR1", "A_CDR3", "B_CDR3")):
    tv, xyz, imap = load_md(sysid); top = tv.mdtraj.topology
    rig = json.load(open(f"{HERE}/rigid_framework.json")); cache = {}; out = {}
    print(f"\n{sysid}: TORSION-space feet-anchored loop ENM vs MD deformation")
    for cdr in cdrs:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        if ch not in cache:
            fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
            cache[ch] = superpose_all(xyz, fw)
        supr = cache[ch]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi); Nres = len(lk)
        ca = np.array([imap[ch][k] for k in lk])
        loop_ca = supr[:, ca]
        md, med = md_deform(loop_ca)
        bb = backbone_indices(top, ca)
        if bb is None:
            print(f"  {cdr:8} N={Nres}: missing backbone atoms, skipped"); continue
        Xbb = supr[med][bb]                                      # reference backbone (medoid frame)
        fluct_at, ndof, nmode = torsion_modes(Xbb, Nres)
        ca_pred = np.sqrt(np.maximum(fluct_at.reshape(Nres, 3)[:, 1], 0))   # CA = local atom 1 per residue
        interior = list(range(1, Nres - 1))
        if ndof >= 1 and len(interior) >= 3 and ca_pred[interior].std() > 0:
            r, p = pearsonr(md[interior], ca_pred[interior])
        else:
            r, p = float("nan"), float("nan")
        print(f"  {cdr:8} N={Nres:2d} interior={len(interior):2d} torsion-DOF={ndof:2d} modes={nmode:2d}  "
              f"Pearson(ENM,MD interior)={r:+.2f} (p={p:.2f})", flush=True)
        out[cdr] = {"lk": lk, "md": md, "pred": ca_pred, "ndof": ndof, "r": r}
    plot(sysid, out)


def plot(sysid, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cdrs = list(out)
    if not cdrs:
        print("nothing to plot"); return
    fig, axes = plt.subplots(1, len(cdrs), figsize=(6.5 * len(cdrs), 4.8))
    if len(cdrs) == 1:
        axes = [axes]
    for ax, cdr in zip(axes, cdrs):
        row = out[cdr]; lk = row["lk"]; x = np.arange(len(lk))
        axr = ax.twinx()
        ax.plot(x, row["md"], "-D", color="#C0392B", lw=2.4, label="MD deformation (Å)")
        pr = row["pred"] / (row["pred"].max() + 1e-9)
        axr.plot(x, pr, "-o", color="#3B6EA5", lw=1.8, ms=4, label=f"torsion-ENM (r={row['r']:+.2f})")
        ax.set_xticks(x); ax.set_xticklabels([str(k) for k in lk], fontsize=7, rotation=90)
        ax.set_xlabel("loop residue (IMGT)"); ax.set_ylabel("MD deformation (Å)", color="#C0392B")
        axr.set_ylabel("torsion-ENM predicted (norm.)", color="#3B6EA5")
        ax.set_title(f"{sysid} {cdr}: torsion-space feet-anchored ENM (DOF={row['ndof']})", fontsize=10, fontweight="bold")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    fig.suptitle("Torsion-space (phi/psi) fixed-boundary loop ENM vs MD deformation profile", y=1.02, fontsize=11, fontweight="bold")
    out_png = f"{HERE}/figures/anm_torsion_{sysid}.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out_png, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "3SKN", tuple(a[1:]) if len(a) > 1 else ("B_CDR1", "A_CDR3", "B_CDR3"))
