"""Aggregate the elasticity fingerprint over all systems and render figures.

Reads results/master_percdr.csv + results/<ID>_elasticity.npz. Robust to a partial
set of completed systems. Run after compute_elasticity.py.
  python plot_elasticity.py
"""
from __future__ import annotations
import os, glob, warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

CDRS = C.CDRS
COLORS = {c: col for c, col in zip(CDRS, plt.cm.tab10(np.linspace(0, 1, len(CDRS))))}
OUTLIER = "1KGC"                       # ~10x breadth outlier (kept in results, flagged in pooled)


def load_master():
    m = pd.read_csv(os.path.join(C.RESULTS, "master_percdr.csv"))
    return m


def load_profiles(cdr, key, ngrid=20):
    """Interpolate each system's per-residue profile onto a common [0,1] loop-position
    grid (loop lengths differ). Returns (grid, stacked (S,ngrid) array, system list)."""
    grid = np.linspace(0, 1, ngrid)
    rows, sysl = [], []
    for f in sorted(glob.glob(os.path.join(C.RESULTS, "*_elasticity.npz"))):
        sid = os.path.basename(f).split("_")[0]
        if sid == OUTLIER:
            continue
        d = np.load(f)
        k = f"{cdr}__{key}"
        if k not in d.files:
            continue
        y = d[k]
        if len(y) < 3:
            continue
        x = np.linspace(0, 1, len(y))
        rows.append(np.interp(grid, x, y))
        sysl.append(sid)
    return grid, (np.array(rows) if rows else np.zeros((0, ngrid))), sysl


def fig1_strain_profiles(m):
    """Per-CDR mean per-residue non-affinity (D2min) and shear along the loop."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, cdr in zip(axes.ravel(), CDRS):
        g, Dp, _ = load_profiles(cdr, "d2min_by_res")
        _, Sp, _ = load_profiles(cdr, "shear_by_res")
        if len(Dp):
            ax.plot(g, Dp.mean(0), color="C3", lw=2, label="non-affine D²min")
            ax.fill_between(g, Dp.mean(0) - Dp.std(0), Dp.mean(0) + Dp.std(0), color="C3", alpha=.15)
        if len(Sp):
            ax.plot(g, Sp.mean(0), color="C0", lw=2, label="shear ‖dev E‖")
            ax.fill_between(g, Sp.mean(0) - Sp.std(0), Sp.mean(0) + Sp.std(0), color="C0", alpha=.15)
        ax.set_title(cdr); ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("loop position (N→C, normalized)")
    fig.suptitle("Fig 1 — finite-strain field along each CDR (mean ± SD over TCRs; 1KGC excluded)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "fig1_strain_profiles.png"), dpi=140)
    plt.close(fig)


def fig2_character_map(m):
    """Strain character: non-affinity vs shear, per loop, colored by CDR."""
    mm = m[m.system != OUTLIER]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for cdr in CDRS:
        s = mm[mm.cdr == cdr]
        ax.scatter(s.shear_mean, s.d2min_mean, s=38, color=COLORS[cdr], label=cdr, alpha=.8, edgecolor="k", lw=.3)
    ax.set_xlabel("affine shear  ⟨‖dev E‖⟩"); ax.set_ylabel("non-affine  ⟨D²min⟩")
    ax.set_title("Fig 2 — deformation character per loop"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "fig2_character_map.png"), dpi=140)
    plt.close(fig)


def fig3_gnm(m):
    """Does uniform-spring elasticity (GNM) explain the loop's mobility? corr per CDR."""
    mm = m[m.system != OUTLIER]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [mm[mm.cdr == c].gnm_obs_corr.values for c in CDRS]
    bp = ax.boxplot(data, labels=CDRS, patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], CDRS):
        patch.set_facecolor(COLORS[c]); patch.set_alpha(.6)
    ax.axhline(0, color="k", lw=.8, ls="--")
    ax.set_ylabel("corr(GNM mobility, observed)"); ax.set_ylim(-1, 1)
    ax.set_title("Fig 3 — uniform-spring elasticity explains CDR1/2, not CDR3"); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "fig3_gnm_explains.png"), dpi=140)
    plt.close(fig)


def fig4_softmode(m):
    """Compliance soft mode: localized (low participation) vs distributed; % variance."""
    mm = m[m.system != OUTLIER]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for cdr in CDRS:
        s = mm[mm.cdr == cdr]
        a1.scatter(s.soft_participation, s.soft_frac1, color=COLORS[cdr], label=cdr, s=36, alpha=.8, edgecolor="k", lw=.3)
    a1.set_xlabel("softest-mode participation (low = localized hinge-like)")
    a1.set_ylabel("variance fraction in softest mode"); a1.legend(fontsize=8); a1.grid(alpha=.3)
    a1.set_title("Fig 4a — softest elastic mode: localized vs distributed")
    means = mm.groupby("cdr").qh_entropy.mean().reindex(CDRS)
    a2.bar(range(len(CDRS)), means.values, color=[COLORS[c] for c in CDRS], alpha=.75)
    a2.set_xticks(range(len(CDRS))); a2.set_xticklabels(CDRS, rotation=30)
    a2.set_ylabel("quasi-harmonic entropy (nats)"); a2.grid(alpha=.3, axis="y")
    a2.set_title("Fig 4b — configurational entropy per CDR")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "fig4_softmode_entropy.png"), dpi=140)
    plt.close(fig)


def summary(m):
    mm = m[m.system != OUTLIER]
    g = mm.groupby("cdr").agg(
        N=("N", "mean"), shear=("shear_mean", "mean"), d2min=("d2min_mean", "mean"),
        gnm_corr=("gnm_obs_corr", "mean"), soft_frac1=("soft_frac1", "mean"),
        soft_PR=("soft_participation", "mean"), entropy=("qh_entropy", "mean"),
    ).reindex(CDRS).round(3)
    g.to_csv(os.path.join(C.RESULTS, "summary_percdr.csv"))
    print(g.to_string())
    return g


def main():
    os.makedirs(C.FIGURES, exist_ok=True)
    m = load_master()
    print(f"loaded {len(m)} rows, {m.system.nunique()} systems\n")
    fig1_strain_profiles(m)
    fig2_character_map(m)
    fig3_gnm(m)
    fig4_softmode(m)
    summary(m)
    print(f"\nfigures -> {C.FIGURES}")


if __name__ == "__main__":
    main()
