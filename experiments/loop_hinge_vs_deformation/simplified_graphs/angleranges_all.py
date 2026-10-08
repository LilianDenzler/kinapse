#!/usr/bin/env python
"""Angle RANGES of the three rigid rotation modes (hinge ĉ / twist ĥ / sway n̂) for ALL TCRs.
Per frame the robust rigid fit gives rotation vector w; the signed angle about each loop-frame axis is w·axis (deg).
Axes are rebuilt per TCR from its own loop geometry -> same DEFINITION everywhere, so ranges are comparable.
For each TCR x CDR x mode we store percentiles p2.5/25/50/75/97.5 (deg) and the 95% swing width.
-> results_angleranges.json + figures/angle_ranges_all.png (per-CDR; every TCR's floating range + the common band)."""
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
RGB = {"hinge": (0.18, 0.49, 0.20), "twist": (0.48, 0.31, 0.64), "sway": (0.94, 0.63, 0.19)}
MODES = ["hinge", "twist", "sway"]
NSUB = 500
PCTS = [2.5, 25, 50, 75, 97.5]


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1); c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


def geom(loop):
    T, N, _ = loop.shape
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    fN, fC = L0[0], L0[-1]; m = 0.5 * (fN + fC); cc = fC - fN; cc /= np.linalg.norm(cc)
    perp = (L0 - m) - ((L0 - m) @ cc)[:, None] * cc
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(cc, h); n /= np.linalg.norm(n)
    return L0, {"hinge": cc, "twist": h, "sway": n}


def one(sysid, rng):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    out = {}; supr_cache = {}
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        if ch not in supr_cache:
            fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
            supr_cache[ch] = superpose_all(xyz, fw)
        supr = supr_cache[ch]
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = supr[:, np.array([imap[ch][k] for k in lk])]; T = loop.shape[0]
        L0, axv = geom(loop)
        idx = rng.choice(T, min(NSUB, T), replace=False)
        ang = {k: np.empty(len(idx)) for k in MODES}
        for s, t in enumerate(idx):
            R, _, _ = robust_kabsch(L0, loop[t]); w = Rot.from_matrix(R).as_rotvec()
            for k in MODES:
                ang[k][s] = np.degrees(w @ axv[k])
        out[cdr] = {k: {"pct": [float(np.percentile(ang[k], p)) for p in PCTS],
                        "rms": float(np.std(ang[k]))} for k in MODES}
    return out


def main():
    tcrs = sorted(os.path.basename(f)[:4] for f in glob.glob(f"{HERE}/results_swing/*.npz"))
    rng = np.random.default_rng(0); res = {}
    for i, s in enumerate(tcrs):
        try:
            res[s] = one(s, rng)
            b3 = res[s]["B_CDR3"]
            print(f"[{i+1}/{len(tcrs)}] {s}: B_CDR3 swing(95%) hinge={b3['hinge']['pct'][4]-b3['hinge']['pct'][0]:.0f} "
                  f"twist={b3['twist']['pct'][4]-b3['twist']['pct'][0]:.0f} sway={b3['sway']['pct'][4]-b3['sway']['pct'][0]:.0f}°", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(tcrs)}] {s}: FAILED {type(e).__name__}: {e}", flush=True)
    json.dump(res, open(f"{HERE}/results_angleranges.json", "w"), indent=1, default=float)
    plot(res)
    # numeric summary: per CDR per mode, median 95% swing width and its [min,max] across TCRs
    print("\nmedian 95% swing width (deg) across TCRs  [min-max]:")
    print(f"{'CDR':9}" + "".join(f"{m:>22}" for m in MODES))
    for cdr in CDRS:
        row = f"{cdr:9}"
        for m in MODES:
            w = np.array([res[s][cdr][m]["pct"][4] - res[s][cdr][m]["pct"][0] for s in res])
            row += f"{np.median(w):8.0f}  [{w.min():.0f}-{w.max():.0f}]".rjust(22)
        print(row, flush=True)


def plot(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    tcrs = list(res.keys()); nt = len(tcrs)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), sharey=False)
    for ax, cdr in zip(axes.flat, CDRS):
        for j, m in enumerate(MODES):
            p = np.array([res[s][cdr][m]["pct"] for s in tcrs])          # (nt,5): p2.5,25,50,75,97.5
            xs = j + np.linspace(-0.32, 0.32, nt)
            # common band = median across TCRs of [p2.5, p97.5]
            lo, hi = np.median(p[:, 0]), np.median(p[:, 4]); c50 = np.median(p[:, 2])
            ax.add_patch(Rectangle((j - 0.42, lo), 0.84, hi - lo, facecolor=RGB[m], alpha=0.13, edgecolor="none", zorder=0))
            ax.hlines(c50, j - 0.42, j + 0.42, color=RGB[m], lw=1.2, zorder=1)
            for i in range(nt):
                ax.vlines(xs[i], p[i, 0], p[i, 4], color=RGB[m], alpha=0.35, lw=1.1, zorder=2)   # 95% range
                ax.vlines(xs[i], p[i, 1], p[i, 3], color=RGB[m], alpha=0.85, lw=2.2, zorder=3)   # IQR
                ax.plot(xs[i], p[i, 2], ".", color="k", ms=2.5, zorder=4)                        # median
        ax.axhline(0, color="0.6", ls=":", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["hinge\n(ĉ)", "twist\n(ĥ)", "sway\n(n̂)"], fontsize=9)
        ax.set_ylabel("rotation angle about axis (deg)"); ax.set_title(cdr, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.6, 2.6)
    fig.suptitle("Angle ranges of the three rigid modes across all 22 TCRs  "
                 "(faint line = per-TCR 95% swing · bold = IQR · dot = median · shaded = common band across TCRs)",
                 y=1.0, fontsize=12, fontweight="bold")
    out = f"{HERE}/figures/angle_ranges_all.png"
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


if __name__ == "__main__":
    main()
