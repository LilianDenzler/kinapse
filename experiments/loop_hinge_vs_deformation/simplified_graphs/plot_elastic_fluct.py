#!/usr/bin/env python
"""Pool the equilibrium-fluctuation elasticity over all TCRs → figures + ELASTIC_FINDINGS.md.

Figures:
  fig_elastic_charactermap  global: compliance C_mean vs mode-spread eff_modes, colour k_soft
  fig_elastic_profiles      local : k_bend & k_torsion stiffness PROFILE along the loop contour
                                    (the "clamped stems, soft apex" test), pooled per CDR
  fig_elastic_spectrum      global: compliance soft-mode spectrum k_k=kT/λ_k per CDR
  fig_elastic_basins        caveat: soft-mode bimodality (which loops are multi-basin)

Importable helpers (pooled_profile, pooled_files) do NOT trigger plotting — run as a script.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from graph_build import HERE

RES = f"{HERE}/results_elastic"
FIG = f"{HERE}/figures"
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
COL = {"CDR1": "#4E79A7", "CDR2": "#59A14F", "CDR3": "#E15759"}
KT = 0.001987 * 300.0
EXCLUDE = {"1KGC"}                                   # ~10× breadth outlier (kept in results, out of pooled means)
files = sorted(glob.glob(f"{RES}/*_elastic.npz"))
pooled_files = [f for f in files if os.path.basename(f)[:4] not in EXCLUDE]


def anchor_positions(kind, N):
    """Contour position s∈[0,1] (0=N-stem, 1=C-stem) of each local-coordinate entry."""
    if kind == "bend":     return np.arange(1, N - 1) / (N - 1)
    if kind == "torsion":  return (np.arange(N - 3) + 1.5) / (N - 1)
    if kind == "stretch":  return (np.arange(N - 1) + 0.5) / (N - 1)
    raise ValueError(kind)


def pooled_profile(cdr, key, kind, ngrid=24):
    """Interpolate each loop's local-stiffness profile onto a common contour grid (log-space)
    and stack. Returns grid (ngrid,), M (n_loops, ngrid) of log10(stiffness)."""
    grid = np.linspace(0, 1, ngrid); M = []
    for f in pooled_files:
        z = np.load(f)
        N = len(z[f"{cdr}__resnums"]); val = z[f"{cdr}__{key}"].astype(float)
        pos = anchor_positions(kind, N)
        m = np.isfinite(val) & (val > 0)
        if m.sum() < 2:
            continue
        M.append(np.interp(grid, pos[m], np.log10(val[m])))
    return grid, np.array(M)


def main():
    # ------------------------------------------------------------ summary table
    master = pd.read_csv(f"{RES}/master_percdr.csv")
    pool = master[~master.system.isin(EXCLUDE)]
    agg = pool.groupby("cdr").agg(
        N=("N", "mean"), C_mean=("C_mean", "mean"), k_soft=("k_soft", "median"),
        eff_modes=("eff_modes", "mean"), soft_frac1=("soft_frac1", "mean"),
        k_bend_med=("k_bend_med", "median"), k_bend_min=("k_bend_min", "median"),
        k_tors_med=("k_tors_med", "median"), multibasin=("multibasin", "mean"),
    ).reindex(CDRS)
    print(f"pooled over {pool.system.nunique()} TCRs (excl. {sorted(EXCLUDE)}); {len(pool)} loops\n")
    print(agg.round(4).to_string())

    # ======================================================== fig1 charactermap
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for cdr in CDRS:
        d = pool[pool.cdr == cdr]; c = COL[cdr[2:]]; mk = "o" if cdr[0] == "A" else "^"
        ax[0].scatter(d.C_mean, d.eff_modes, s=46, marker=mk, color=c, alpha=.8, ec="k", lw=.4, label=cdr)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("C_mean = tr(Σ_q)/m   (mean normalized-strain compliance →  softer)")
    ax[0].set_ylabel("eff_modes  (participation ratio of Σ_q spectrum)\nlow = one soft mode · high = distributed")
    ax[0].set_title("Global elasticity character map (one point = one loop)")
    ax[0].legend(fontsize=8, ncol=2, loc="upper left")
    data = [pool[pool.cdr == cdr].k_soft.values for cdr in CDRS]
    bp = ax[1].boxplot(data, tick_labels=CDRS, showfliers=False, patch_artist=True, widths=.6)
    for patch, cdr in zip(bp["boxes"], CDRS):
        patch.set_facecolor(COL[cdr[2:]]); patch.set_alpha(.7)
    for med in bp["medians"]:
        med.set_color("k")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("k_soft = kT/λ_max   (kcal/mol) — stiffness of the EASIEST internal mode")
    ax[1].set_title("Softest-mode stiffness per CDR")
    ax[1].tick_params(axis="x", rotation=35)
    fig.suptitle("Equilibrium-fluctuation elasticity — GLOBAL:  K_eff = kT·Σ_q⁺  from normalized strain q_ij=(d_ij−⟨d_ij⟩)/⟨d_ij⟩",
                 y=1.02, fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_elastic_charactermap.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("\nfig ->", f"{FIG}/fig_elastic_charactermap.png")

    # ======================================================== fig2 local profiles
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    for row, key, kind, lab in [(0, "k_bend", "bend", "bending  k_b(i)=kT/Var(θ_i)"),
                                (1, "k_torsion", "torsion", "torsion  k_t(i)=kT/Var(φ_i)")]:
        for col, cdrbase in enumerate(["CDR1", "CDR2", "CDR3"]):
            ax = axes[row, col]
            for ch, ls in [("A", "-"), ("B", "--")]:
                cdr = f"{ch}_{cdrbase}"; grid, M = pooled_profile(cdr, key, kind)
                if not len(M):
                    continue
                med = np.nanmedian(M, 0); lo = np.nanpercentile(M, 25, 0); hi = np.nanpercentile(M, 75, 0)
                ax.plot(grid, med, ls, color=COL[cdrbase], lw=2, label=f"{cdr} (n={len(M)})")
                ax.fill_between(grid, lo, hi, color=COL[cdrbase], alpha=.13)
            ax.set_title(f"{cdrbase} — {lab}", fontsize=10)
            ax.legend(fontsize=8); ax.grid(alpha=.25)
            if col == 0:
                ax.set_ylabel("log₁₀ stiffness\n(kcal/mol/rad²)")
            if row == 1:
                ax.set_xlabel("contour position   0 = N-stem  →  1 = C-stem")
    fig.suptitle("Equilibrium-fluctuation elasticity — LOCAL discrete-rod stiffness PROFILE along each CDR\n"
                 "(pooled median ± IQR; low = soft/compliant.  Clamped stems ⇒ stiff ends; a soft apex ⇒ dip in the middle)",
                 y=1.0, fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_elastic_profiles.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", f"{FIG}/fig_elastic_profiles.png")

    # ======================================================== fig3 spectrum
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    for cdr in CDRS:
        ks = []
        for f in pooled_files:
            z = np.load(f); lam = z[f"{cdr}__lam"].astype(float); lam = lam[lam > 1e-10]
            ks.append(KT / lam[:8])
        L = min(len(x) for x in ks); K = np.array([x[:L] for x in ks])
        ax[0].plot(range(1, L + 1), np.median(K, 0), "o-" if cdr[0] == "A" else "^--",
                   color=COL[cdr[2:]], label=cdr, alpha=.85)
    ax[0].set_yscale("log"); ax[0].set_xlabel("mode index k (soft → stiff)")
    ax[0].set_ylabel("k_k = kT/λ_k   (kcal/mol)")
    ax[0].set_title("Deformation-mode stiffness spectrum (pooled median)")
    ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=.25)
    data = [pool[pool.cdr == cdr].soft_frac1.values for cdr in CDRS]
    bp = ax[1].boxplot(data, tick_labels=CDRS, showfliers=False, patch_artist=True, widths=.6)
    for patch, cdr in zip(bp["boxes"], CDRS):
        patch.set_facecolor(COL[cdr[2:]]); patch.set_alpha(.7)
    for med in bp["medians"]:
        med.set_color("k")
    ax[1].set_ylabel("soft_frac1 — variance fraction in the single softest mode")
    ax[1].set_title("Is flexibility in ONE mode (high) or spread out (low)?")
    ax[1].tick_params(axis="x", rotation=35)
    fig.suptitle("Deformation modes = spectral decomposition of the compliance Σ_q (not a predictive PCA fit)", y=1.02, fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_elastic_spectrum.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", f"{FIG}/fig_elastic_spectrum.png")

    # ======================================================== fig4 basins (caveat)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    data = [pool[pool.cdr == cdr].soft_mode_bc.values for cdr in CDRS]
    bp = ax.boxplot(data, tick_labels=CDRS, showfliers=True, patch_artist=True, widths=.6)
    for patch, cdr in zip(bp["boxes"], CDRS):
        patch.set_facecolor(COL[cdr[2:]]); patch.set_alpha(.7)
    for med in bp["medians"]:
        med.set_color("k")
    ax.axhline(5 / 9, color="crimson", ls="--", lw=1.3, label="BC = 5/9 ≈ 0.555  (multi-basin threshold)")
    ax.axhline(1 / 3, color="0.4", ls=":", lw=1.2, label="BC = 1/3  (Gaussian / harmonic)")
    ax.set_ylabel("softest-mode bimodality coefficient (Sarle BC)")
    ax.set_title("Harmonic-basin check: where kT/Var is a LOCAL (not global) stiffness\n"
                 "BC>0.555 ⇒ the soft mode is multi-modal (report K_eff as effective/per-basin)")
    ax.tick_params(axis="x", rotation=35); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_elastic_basins.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", f"{FIG}/fig_elastic_basins.png")

    agg.round(4).to_csv(f"{RES}/summary_percdr.csv")
    print("\nsummary ->", f"{RES}/summary_percdr.csv")


if __name__ == "__main__":
    main()
