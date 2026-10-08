#!/usr/bin/env python
"""Aggregate + plot the hinge-vs-deformation results. Run after compute_hinge_deform.py."""
from __future__ import annotations
import os, sys, glob
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

FIG = os.path.join(C.ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
CDRS = C.CDRS
OUTLIER = "1KGC"                                   # 10x ensemble-breadth outlier (MD_flexibility)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 130, "axes.axisbelow": True})
COL = {"A_CDR1": "#4C72B0", "A_CDR2": "#5A8FBF", "A_CDR3": "#2E5A88",
       "B_CDR1": "#C44E52", "B_CDR2": "#D07A6B", "B_CDR3": "#8C2D2F"}


def load():
    paths = [p for p in sorted(glob.glob(f"{C.RESULTS}/*_percdr.csv"))
             if not os.path.basename(p).startswith(("master", "summary"))]   # don't re-ingest aggregates
    master = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    master.to_csv(f"{C.RESULTS}/master_percdr.csv", index=False)
    return master


def perframe(sysid):
    f = f"{C.RESULTS}/{sysid}_perframe.npz"
    return np.load(f) if os.path.exists(f) else None


def pooled(cdr, key, exclude_outlier=True, sub=1500):
    out = []
    for p in sorted(glob.glob(f"{C.RESULTS}/*_perframe.npz")):
        s = os.path.basename(p)[:4]
        if exclude_outlier and s == OUTLIER:
            continue
        z = np.load(p)
        k = f"{cdr}__{key}"
        if k in z.files:
            v = z[k]
            if len(v) > sub:
                v = v[np.linspace(0, len(v) - 1, sub).astype(int)]
            out.append(v)
    return np.concatenate(out) if out else np.array([])


# ---- fig1: (D_framework, D_deform) 2D map per CDR --------------------------
def fig1(master):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), sharex=True, sharey=True)
    for ax, cdr in zip(axes.ravel(), CDRS):
        x = pooled(cdr, "D_framework"); y = pooled(cdr, "D_deform")
        if len(x) == 0:
            continue
        hb = ax.hexbin(x, y, gridsize=45, bins="log", cmap="viridis", mincnt=1)
        lim = np.nanpercentile(np.r_[x, y], 99.5)
        ax.plot([0, lim], [0, lim], "w--", lw=1, alpha=.7)          # y=x: equal hinge/deform
        mx, my = np.median(x), np.median(y)
        ax.axvline(mx, color="w", lw=.6, alpha=.4); ax.axhline(my, color="w", lw=.6, alpha=.4)
        ax.set_title(f"{cdr}   median H={master.loc[master.cdr==cdr,'H'].median():.2f}", color=COL[cdr])
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    for ax in axes[-1]:
        ax.set_xlabel(r"$D_{\rm framework}$  (reorientation proxy, Å)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$D_{\rm deform}$  (Å)")
    fig.suptitle("Loop motion is hinge-dominated: points hug the $D_{framework}$ axis, "
                 "far below the $y=x$ (equal) line   ·  pooled 21 TCRs (1KGC excl.)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_2Dmap.png", bbox_inches="tight"); plt.close(fig)


# ---- fig2: hinge fraction + deform vs framework per CDR --------------------
def fig2(master):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    order = CDRS
    # (a) hinge_frac strip across systems
    for i, cdr in enumerate(order):
        d = master[master.cdr == cdr].reset_index(drop=True)
        xj = np.random.default_rng(i).normal(i, 0.06, len(d))
        norm = d[d.system != OUTLIER]
        ax[0].scatter(xj, d.hinge_frac, s=16, c=COL[cdr], alpha=.75, edgecolor="none")
        ax[0].hlines(norm.hinge_frac.median(), i-0.3, i+0.3, color="k", lw=2, zorder=5)
        out = d[d.system == OUTLIER]
        if len(out):
            ax[0].scatter([i], out.hinge_frac, marker="x", c="red", s=40, zorder=6)
    ax[0].axhline(0.92, color="gray", ls="--", lw=1)
    ax[0].text(5.4, 0.925, "MD_flex 92%", ha="right", fontsize=8, color="gray")
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=30)
    ax[0].set_ylabel("hinge fraction  $1-E_{nonrigid}^2/{\\rm total}^2$"); ax[0].set_ylim(0, 1)
    ax[0].set_title("Hinge fraction per CDR (dot=TCR, bar=median; red× = 1KGC)")
    # (b) mean D_deform vs D_framework per CDR
    for cdr in order:
        d = master[(master.cdr == cdr) & (master.system != OUTLIER)]
        ax[1].scatter(d.D_framework, d.D_deform, s=20, c=COL[cdr], label=cdr, alpha=.8)
    lim = master[master.system != OUTLIER][["D_framework", "D_deform"]].max().max() * 1.05
    ax[1].plot([0, lim], [0, lim], "k--", lw=1, alpha=.5)
    ax[1].set_xlabel(r"$D_{\rm framework}$ (Å)"); ax[1].set_ylabel(r"$D_{\rm deform}$ (Å)")
    ax[1].set_title("Per-TCR means: deformation ≪ framework displacement"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_hinge_fraction.png", bbox_inches="tight"); plt.close(fig)


# ---- fig3: theta distribution + dtheta per CDR ----------------------------
def fig3(master):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, cdr in enumerate(CDRS):
        th = pooled(cdr, "theta")
        if len(th) == 0:
            continue
        parts = ax.violinplot(th, positions=[i], widths=0.8, showmedians=True)
        for b in parts["bodies"]:
            b.set_facecolor(COL[cdr]); b.set_alpha(.6)
        dth = master.loc[master.cdr == cdr, "dtheta"].median()
        ax.text(i, np.percentile(th, 98) + 1, f"δθ≈{dth:.0f}°", ha="center", fontsize=7, color="dimgray")
    ax.set_xticks(range(len(CDRS))); ax.set_xticklabels(CDRS, rotation=20)
    ax.set_ylabel("hinge angle θ vs central conformation (deg)")
    ax.set_title("Per-frame hinge angle (pooled 21 TCRs); δθ = median hinge uncertainty from deformation")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_theta.png", bbox_inches="tight"); plt.close(fig)


# ---- fig4: PMF over (theta, D_deform) for the CDR3 loops -------------------
def pmf(ax, x, y, title, color):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    xr = (0, np.percentile(x, 99)); yr = (0, np.percentile(y, 99))
    Hh, xe, ye = np.histogram2d(x, y, bins=60, range=[xr, yr], density=True)
    F = -np.log(Hh + Hh[Hh > 0].min() * 1e-3); F -= F.min()
    F = np.ma.masked_where(Hh.T == 0, F.T)
    pc = ax.pcolormesh(xe, ye, F, cmap="magma_r", vmax=6, shading="auto")
    ax.set_title(title, color=color); ax.set_xlabel("θ (deg)"); ax.set_ylabel(r"$D_{\rm deform}$ (Å)")
    return pc


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, cdr in zip(axes, ["A_CDR3", "B_CDR3"]):
        pc = pmf(ax, pooled(cdr, "theta"), pooled(cdr, "D_deform"), f"PMF  {cdr}", COL[cdr])
        fig.colorbar(pc, ax=ax, label=r"$-\ln P$ ($k_BT$)")
    fig.suptitle("Free-energy landscape: loops explore a wide hinge angle at ~constant (low) deformation", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_pmf.png", bbox_inches="tight"); plt.close(fig)


def summary(master):
    g = master[master.system != OUTLIER].groupby("cdr")
    tab = g.agg(n=("system", "size"),
                D_deform=("D_deform", "mean"), D_framework=("D_framework", "mean"),
                theta=("theta", "mean"), theta_p95=("theta_p95", "mean"),
                dtheta=("dtheta", "mean"), hinge_frac=("hinge_frac", "mean"),
                H=("H", "mean")).round(3).reindex(CDRS)
    tab.to_csv(f"{C.RESULTS}/summary_percdr.csv")
    print(tab.to_string())
    print(f"\nAll-CDR mean hinge_frac (excl 1KGC): {master[master.system!=OUTLIER].hinge_frac.mean():.3f}")
    print(f"Mean D_deform {master[master.system!=OUTLIER].D_deform.mean():.3f} A  vs  "
          f"D_framework {master[master.system!=OUTLIER].D_framework.mean():.3f} A")


def main():
    master = load()
    print(f"{master.system.nunique()} systems, {len(master)} CDR rows\n")
    fig1(master); fig2(master); fig3(master); fig4(); summary(master)
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
