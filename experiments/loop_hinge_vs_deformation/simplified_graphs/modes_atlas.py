#!/usr/bin/env python
"""All-22x6 essential-dynamics ATLAS + quasi-harmonic ENTROPY (numbers only, no PyMOL).
Per TCR x CDR:
  - Cartesian PCA (framework-ref, rigid preserved) spectrum -> PR_cart, top-3 modes (%var, rigid%, hinge/twist/sway)
  - CA-CA distance PCA (alignment-free, deformation-only) spectrum -> PR_dist
  - S_rot : non-parametric (Kozachenko-Leonenko kNN) differential entropy of the 3-D rotation vectors (handles large/
            multimodal swings) x R  -> cal/mol/K
  - S_deform : Schlitter quasi-harmonic entropy of the Cartesian deformation-residual covariance (CA, carbon mass) -> cal/mol/K
Saves results_modes_atlas.json FIRST (crash-safe), then renders:
  figures/atlas_dof.png (PR + mode-1 rigid% heatmaps), figures/conserved_motions_ternary.png,
  figures/conserved_motions_bars.png, figures/entropy_atlas.png
Honest framing: S is UNBOUND loop configurational entropy = upper bound on entropy lost if the loop fully freezes on
binding (NOT ΔS; absolute values carry a reference-state offset, so RELATIVE comparison + the rigid/deform split are the robust outputs)."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all
from compute_modes_pca import pca_cart, dist_spectrum, participation, rigid_basis, hts_shares, geom_axes, CLAMP

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
RGB = {"hinge": (0.18, 0.49, 0.20), "twist": (0.48, 0.31, 0.64), "sway": (0.94, 0.63, 0.19)}
CCOL = {"A_CDR1": "#1f77b4", "A_CDR2": "#2ca02c", "A_CDR3": "#d62728", "B_CDR1": "#17becf", "B_CDR2": "#9467bd", "B_CDR3": "#ff7f0e"}
R_GAS = 1.987204            # cal/mol/K
NSUB = 1200


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1); c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


def kl_entropy(X, k=5):
    """Kozachenko-Leonenko differential entropy (nats). X (n,d) in radians."""
    n, d = X.shape
    r = cKDTree(X).query(X, k + 1)[0][:, k]
    r = np.maximum(r, 1e-12)
    cd = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)
    return float(digamma(n) - digamma(k) + cd + (d / n) * np.sum(np.log(r)))


def schlitter(cov, mass_amu=12.011, T=300.0):
    """Schlitter QH entropy (cal/mol/K) of a Cartesian covariance (Å²)."""
    kB = 1.380649e-23; hbar = 1.054571817e-34; amu = 1.66053906660e-27
    ev = np.linalg.eigvalsh(cov * 1e-20)                       # Å² -> m²
    ev = ev[ev > 1e-25]
    kappa = kB * T * np.e ** 2 / hbar ** 2
    return float((R_GAS / 2) * np.sum(np.log1p(kappa * mass_amu * amu * ev)))


def analyze(loop, cdr, rng):
    T, N, _ = loop.shape
    axv = geom_axes(loop)                                      # loop-frame axes (medoid-derived internally)
    mean, lam_c, V = pca_cart(loop)
    piv = mean[-1] if CLAMP[cdr[2:]] == "C" else mean[0]       # PCA part is centred on the mean structure
    lam_d = dist_spectrum(loop)
    # medoid reference for the per-frame rigid fit (consistent with composite/angleranges)
    ii, jj = np.triu_indices(N, 1); dLL = np.linalg.norm(loop[:, ii] - loop[:, jj], axis=-1)
    L0 = loop[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    Q6 = rigid_basis(mean, piv); tot = lam_c[lam_c > 0].sum()
    top = []
    for m in range(min(3, 3 * N)):
        v = V[:, m]; frig = float((Q6.T @ v) @ (Q6.T @ v)); ch = hts_shares(v, mean, piv, axv)
        top.append(dict(var=float(100 * lam_c[m] / tot), rms=float(np.sqrt(lam_c[m])), rigid=frig,
                        hinge=ch["hinge"], twist=ch["twist"], sway=ch["sway"]))
    idx = rng.choice(T, min(NSUB, T), replace=False)
    sub = loop[idx]
    W = np.empty((len(idx), 3)); resid = np.empty((len(idx), 3 * N))
    for s, t in enumerate(idx):
        R, Pc, Qc = robust_kabsch(L0, loop[t]); W[s] = Rot.from_matrix(R).as_rotvec()
        resid[s] = (loop[t] - ((L0 - Pc) @ R.T + Qc)).reshape(-1)
    # Schlitter-consistent split (same quantum reference -> addable, real cal/mol/K):
    S_deform = schlitter(np.cov(resid.T))                        # QH internal (deformation-only) entropy
    S_total = schlitter(np.cov(sub.reshape(len(idx), 3 * N).T))  # QH entropy of ALL loop motion
    S_rigid = S_total - S_deform                                 # rigid-body contribution (by difference)
    # non-parametric rotational entropy (handles large/multimodal swings) — RELATIVE cross-check, NOT addable to Schlitter:
    S_rot_np = R_GAS * kl_entropy(W)
    return dict(N=int(N), PR_cart=float(participation(lam_c)), PR_dist=float(participation(lam_d)),
                lam_cart=[float(x) for x in lam_c[:12]], lam_dist=[float(x) for x in lam_d[:12]],
                top=top, S_deform=S_deform, S_rigid=S_rigid, S_total=S_total, S_rot_np=S_rot_np)


def main():
    tcrs = sorted(os.path.basename(f)[:4] for f in glob.glob(f"{HERE}/results_swing/*.npz"))
    rng = np.random.default_rng(0); res = {}
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    for i, s in enumerate(tcrs):
        try:
            tv, xyz, imap = load_md(s); res[s] = {}; cache = {}
            for cdr in CDRS:
                ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
                if ch not in cache:
                    fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
                    cache[ch] = superpose_all(xyz, fw)
                lk = sorted(k for k in imap[ch] if lo <= k <= hi)
                loop = cache[ch][:, np.array([imap[ch][k] for k in lk])]
                res[s][cdr] = analyze(loop, cdr, rng)
            d = res[s]
            print(f"[{i+1}/{len(tcrs)}] {s}: PR_cart {np.mean([d[c]['PR_cart'] for c in CDRS]):.1f} "
                  f"S_tot {np.mean([d[c]['S_total'] for c in CDRS]):.0f} cal/mol/K", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(tcrs)}] {s}: FAILED {type(e).__name__}: {e}", flush=True)
    json.dump(res, open(f"{HERE}/results_modes_atlas.json", "w"), indent=1, default=float)
    print("saved results_modes_atlas.json", flush=True)
    for fn in (fig_dof, fig_ternary, fig_bars, fig_entropy):
        try:
            fn(res)
        except Exception as e:
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}", flush=True)


def _heat(ax, M, tcrs, title, cmap, vmin, vmax, cbar_label, fmt="{:.1f}"):
    import matplotlib.pyplot as plt
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(6)); ax.set_xticklabels(CDRS, rotation=30, fontsize=8)
    ax.set_yticks(range(len(tcrs))); ax.set_yticklabels(tcrs, fontsize=7)
    for y in range(len(tcrs)):
        for x in range(6):
            ax.text(x, y, fmt.format(M[y, x]), ha="center", va="center", fontsize=6)
    ax.set_title(title, fontsize=10, fontweight="bold")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label(cbar_label, fontsize=8)


def fig_dof(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res.keys())
    PRc = np.array([[res[t][c]["PR_cart"] for c in CDRS] for t in tcrs])
    PRd = np.array([[res[t][c]["PR_dist"] for c in CDRS] for t in tcrs])
    rig1 = np.array([[100 * res[t][c]["top"][0]["rigid"] for c in CDRS] for t in tcrs])
    fig, ax = plt.subplots(1, 3, figsize=(20, 9))
    _heat(ax[0], PRc, tcrs, "a  effective DOF — Cartesian (rigid+internal)\nlow = few-DOF rigid rotor, high = many-DOF", "viridis", 1, 7, "PR_cart")
    _heat(ax[1], PRd, tcrs, "b  effective INTERNAL DOF — CA-CA distance\n(alignment-free, deformation-only)", "magma", 1, 7, "PR_dist")
    _heat(ax[2], rig1, tcrs, "c  mode-1 rigid %  (is the dominant motion a rigid swing?)", "RdYlBu", 0, 100, "rigid %", fmt="{:.0f}")
    fig.suptitle("Effective-DOF atlas: which loops are few-DOF rigid rotors vs many-DOF deformers (all 22 TCRs x 6 CDRs)", y=1.0, fontsize=13, fontweight="bold")
    out = f"{HERE}/figures/atlas_dof.png"; fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


def _tern_xy(h, t, s):
    tot = h + t + s + 1e-12; t, s = t / tot, s / tot
    return t + 0.5 * s, (np.sqrt(3) / 2) * s


def fig_ternary(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, cdr in zip(axes.flat, CDRS):
        for cx, cy, lab in [(0, 0, "hinge"), (1, 0, "twist"), (0.5, np.sqrt(3) / 2, "sway")]:
            ax.plot([cx], [cy], marker="^", color=RGB[lab], ms=1)
            ax.annotate(lab, (cx, cy), fontsize=10, fontweight="bold", color=RGB[lab],
                        ha="center", va="top" if lab != "sway" else "bottom")
        tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
        ax.plot(tri[:, 0], tri[:, 1], "k-", lw=0.8)
        for t in tcrs:
            m = res[t][cdr]["top"][0]
            x, y = _tern_xy(m["hinge"], m["twist"], m["sway"])
            ax.scatter(x, y, s=40, c=[m["rigid"]], cmap="RdYlBu", vmin=0, vmax=1, edgecolor="k", lw=0.4, zorder=3)
        ax.set_title(cdr, fontsize=11, fontweight="bold"); ax.axis("off"); ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap="RdYlBu", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02).set_label("mode-1 rigid fraction", fontsize=9)
    fig.suptitle("Conserved or idiosyncratic? mode-1 motion type per CDR across all 22 TCRs (tight cluster = conserved axis)", y=1.0, fontsize=13, fontweight="bold")
    out = f"{HERE}/figures/conserved_motions_ternary.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


def fig_bars(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res.keys()); x = np.arange(len(tcrs))
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    for ax, cdr in zip(axes.flat, CDRS):
        h = np.array([res[t][cdr]["top"][0]["hinge"] for t in tcrs])
        tw = np.array([res[t][cdr]["top"][0]["twist"] for t in tcrs])
        sw = np.array([res[t][cdr]["top"][0]["sway"] for t in tcrs])
        ax.bar(x, h, 0.85, color=RGB["hinge"], label="hinge")
        ax.bar(x, tw, 0.85, bottom=h, color=RGB["twist"], label="twist")
        ax.bar(x, sw, 0.85, bottom=h + tw, color=RGB["sway"], label="sway")
        ax.set_xticks(x); ax.set_xticklabels(tcrs, rotation=90, fontsize=6.5)
        ax.set_ylim(0, 1); ax.set_title(cdr, fontsize=11, fontweight="bold"); ax.set_ylabel("mode-1 share")
    axes.flat[0].legend(fontsize=8, loc="upper right", ncol=3)
    fig.suptitle("mode-1 hinge/twist/sway composition per TCR (consistent stack within a CDR = conserved dominant motion)", y=1.0, fontsize=13, fontweight="bold")
    out = f"{HERE}/figures/conserved_motions_bars.png"; fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


def fig_entropy(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res.keys())
    St = np.array([[res[t][c]["S_total"] for c in CDRS] for t in tcrs])
    fig, ax = plt.subplots(1, 3, figsize=(20, 9))
    _heat(ax[0], St, tcrs, "a  total QH configurational entropy S_total (cal/mol/K)\nUNBOUND = upper bound on freeze-on-binding cost", "YlOrRd", float(St.min()), float(St.max()), "S (cal/mol/K)", fmt="{:.0f}")
    for c in CDRS:
        sr = [res[t][c]["S_rigid"] for t in tcrs]; sd = [res[t][c]["S_deform"] for t in tcrs]
        ax[1].scatter(sr, sd, s=30, color=CCOL[c], label=c, alpha=0.8)
    ax[1].set_xlabel("S_rigid (rigid-body, = S_total − S_deform) cal/mol/K"); ax[1].set_ylabel("S_deform (QH internal) cal/mol/K")
    ax[1].set_title("b  rigid-body vs internal entropy per loop (both Schlitter → addable)", fontsize=10, fontweight="bold"); ax[1].legend(fontsize=7, ncol=2)
    mr = [np.mean([res[t][c]["S_rigid"] for t in tcrs]) for c in CDRS]
    md = [np.mean([res[t][c]["S_deform"] for t in tcrs]) for c in CDRS]
    xx = np.arange(6)
    ax[2].bar(xx, mr, 0.8, color="#3B6EA5", label="S_rigid (body)")
    ax[2].bar(xx, md, 0.8, bottom=mr, color=(0.75, 0.22, 0.17), label="S_deform (internal)")
    ax[2].set_xticks(xx); ax[2].set_xticklabels(CDRS, rotation=30, fontsize=8); ax[2].set_ylabel("mean S (cal/mol/K)")
    ax[2].set_title("c  mean entropy by CDR (rigid-body + internal)", fontsize=10, fontweight="bold"); ax[2].legend(fontsize=8)
    fig.suptitle("Quasi-harmonic configurational entropy per loop (UNBOUND upper bound; CA/carbon-mass Schlitter — read relative differences + the rigid/internal split)", y=1.0, fontsize=12, fontweight="bold")
    out = f"{HERE}/figures/entropy_atlas.png"; fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


if __name__ == "__main__":
    main()
