#!/usr/bin/env python
"""Rigid vs DEFORMATION across ALL TCRs (no PyMOL). For every system x CDR:
  V_rigid  = Σ_res mean_t ‖(robust rigid fit of L0)  - L0‖²   (Å²)   -- pure rigid motion, deform-excluded
  V_deform = Σ_res mean_t ‖loop_t - (rigid fit)‖²             (Å²)   -- residual the rigid fit can't reproduce
  V_total  = V_rigid + V_deform (orthogonal by construction).
Saves results_rigid_deform.json and figures/rigid_vs_deform_all.png:
  A) per-TCR stacked Å² (rigid+deform, summed over 6 CDRs), sorted by deform fraction
  B) heatmap of deform-fraction (%) per TCR x CDR."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
NSUB = 500


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1); c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


def one(sysid, rng):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    out = {}
    supr_cache = {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        if ch not in supr_cache:
            fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
            supr_cache[ch] = superpose_all(xyz, fw)
        supr = supr_cache[ch]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = supr[:, np.array([imap[ch][k] for k in lk])]; T, N, _ = loop.shape
        dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
        L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
        idx = rng.choice(T, min(NSUB, T), replace=False)
        sr = np.zeros(N); sd = np.zeros(N); st = np.zeros(N)
        for t in idx:
            R, Pc, Qc = robust_kabsch(L0, loop[t])
            rigid = (L0 - Pc) @ R.T + Qc
            sr += ((rigid - L0) ** 2).sum(1)
            sd += ((loop[t] - rigid) ** 2).sum(1)
            st += ((loop[t] - L0) ** 2).sum(1)
        n = len(idx)
        out[cdr] = dict(rigid=float((sr / n).sum()), deform=float((sd / n).sum()), total=float((st / n).sum()), N=int(N))
    return out


def main():
    tcrs = sorted(os.path.basename(f)[:4] for f in glob.glob(f"{HERE}/results_swing/*.npz"))
    rng = np.random.default_rng(0)
    res = {}
    for i, s in enumerate(tcrs):
        try:
            res[s] = one(s, rng)
            tot = sum(v["total"] for v in res[s].values()); dfr = 100 * sum(v["deform"] for v in res[s].values()) / tot
            print(f"[{i+1}/{len(tcrs)}] {s}: total={tot:.1f} Å²  deform={dfr:.0f}%", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(tcrs)}] {s}: FAILED {type(e).__name__}: {e}", flush=True)
    json.dump(res, open(f"{HERE}/results_rigid_deform.json", "w"), indent=1, default=float)
    plot(res)


def plot(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res.keys())
    Rg = np.array([sum(res[s][c]["rigid"] for c in CDRS) for s in tcrs])
    Df = np.array([sum(res[s][c]["deform"] for c in CDRS) for s in tcrs])
    Tt = Rg + Df; frac = Df / Tt
    o = np.argsort(frac)                                      # sort by deform fraction (rigid-most first)
    tS = [tcrs[i] for i in o]; RgS = Rg[o]; DfS = Df[o]; frS = frac[o]
    fig, ax = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={"width_ratios": [1.35, 1.0]})
    x = np.arange(len(tcrs))
    ax[0].bar(x, RgS, 0.8, color="#3B6EA5", label="rigid (Å²)")
    ax[0].bar(x, DfS, 0.8, bottom=RgS, color=(0.75, 0.22, 0.17), label="deformation (Å²)")
    for xi, (r, d, fr) in enumerate(zip(RgS, DfS, frS)):
        ax[0].text(xi, r + d + max(Tt) * 0.01, f"{fr*100:.0f}%", ha="center", va="bottom", fontsize=7.5, color=(0.6, 0.15, 0.1))
    ax[0].set_xticks(x); ax[0].set_xticklabels(tS, rotation=90, fontsize=8.5)
    ax[0].set_ylabel("summed displacement variance over 6 CDRs (Å²)")
    ax[0].set_title("a  rigid vs deformation per TCR  (Σ over 6 CDRs; % = deformation share; left = most rigid)", fontsize=10, fontweight="bold")
    ax[0].legend(fontsize=9, loc="upper left")
    # B: heatmap deform fraction per TCR x CDR
    M = np.array([[res[s][c]["deform"] / res[s][c]["total"] * 100 for c in CDRS] for s in tS])
    im = ax[1].imshow(M, aspect="auto", cmap="RdBu_r", vmin=0, vmax=60)
    ax[1].set_xticks(range(6)); ax[1].set_xticklabels(CDRS, rotation=30, fontsize=8.5)
    ax[1].set_yticks(range(len(tS))); ax[1].set_yticklabels(tS, fontsize=8)
    for yi in range(len(tS)):
        for xi in range(6):
            ax[1].text(xi, yi, f"{M[yi, xi]:.0f}", ha="center", va="center", fontsize=6.5,
                       color="white" if (M[yi, xi] > 40 or M[yi, xi] < 12) else "black")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02); cb.set_label("deformation share (%)", fontsize=9)
    ax[1].set_title("b  deformation share per TCR x CDR (%)  (blue = rigid-dominated, red = deform-dominated)", fontsize=10, fontweight="bold")
    fig.suptitle("Rigid vs deformation across all TCRs  (rigid = robust deform-excluded fit; deform = orthogonal residual, = total − rigid)",
                 y=1.01, fontsize=12, fontweight="bold")
    out = f"{HERE}/figures/rigid_vs_deform_all.png"
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


if __name__ == "__main__":
    main()
