#!/usr/bin/env python
"""Cluster the unbound TCRs by CDR-loop flexibility + inter-domain geometry.

Each TCR is described by a feature vector that fuses the two analyses in this
experiment:

  loop flexibility (from ``results_flex_vs_geometry/flex_components.csv``)
    * per-CDR total RMSF (all six loops)              — how mobile each loop is
    * CDR3 α & β: angle RMSF, deform RMSF, angle_frac, max excursion
                                                       — reorientation-vs-shape character

  geometry exploration (from ``results/summary/system_summary.csv``)
    * std of the six Vα/Vβ docking parameters (BA, AC1/2, BC1/2, dc)
    * std of the CDR3 α & β bend angle
                                                       — how much the domain packing
                                                         and loop hinge move

Features are z-scored, then Ward-clustered; the number of clusters is chosen by
silhouette (override with ``--k``). Outputs (``results_flex_vs_geometry/clustering/``):

  * ``tcr_clusters.csv``            — cluster label per TCR
  * ``cluster_feature_means.csv``   — mean (raw) feature per cluster (what defines them)
  * ``tcr_dendrogram.png``          — Ward dendrogram, coloured by cluster
  * ``tcr_cluster_pca.png``         — PCA of the feature space, coloured by cluster
  * ``tcr_feature_heatmap.png``     — z-scored features × TCRs, ordered by cluster

Run:  python cluster_tcrs.py [--k N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
FLEX = HERE / "results_flex_vs_geometry" / "flex_components.csv"
GEOM = HERE / "results" / "summary" / "system_summary.csv"
OUT = HERE / "results_flex_vs_geometry" / "clustering"

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
GEOM_STD = ["BA_std", "AC1_std", "AC2_std", "BC1_std", "BC2_std", "dc_std",
            "alpha_cdr3_bend_deg_std", "beta_cdr3_bend_deg_std"]
# feature blocks (for the heatmap grouping / interpretation)
BLOCKS = [("loop flexibility", None), ("CDR3 character", None), ("geometry", None)]


def build_features():
    """Per-TCR feature table fusing flexibility + geometry."""
    flex = pd.read_csv(FLEX)
    feats = {}
    piv_tot = flex.pivot_table(index="system", columns="cdr", values="rmsf_total_A")
    for c in CDRS:                                    # loop-flexibility block
        feats[f"flex:{c}"] = piv_tot[c]
    for c in ("A_CDR3", "B_CDR3"):                    # CDR3-character block
        for col, name in (("rmsf_angle_A", "angle"), ("rmsf_deform_A", "deform"),
                          ("angle_frac", "anglefrac"), ("max_total_A", "max")):
            feats[f"char:{c}_{name}"] = flex[flex["cdr"] == c].set_index("system")[col]
    F = pd.DataFrame(feats)

    geom = pd.read_csv(GEOM).set_index("system")
    have = [g for g in GEOM_STD if g in geom.columns]
    G = geom[have].rename(columns=lambda x: "geom:" + x.replace("_deg", "").replace("_std", "_std"))
    X = F.join(G, how="inner").dropna()
    return X


def _pick_k(Z, link, forced=None):
    if forced:
        return forced
    scored = []
    for k in range(2, min(7, len(Z))):
        lab = fcluster(link, k, criterion="maxclust")
        if len(set(lab)) > 1:
            scored.append((silhouette_score(Z, lab), k))
    return max(scored)[1] if scored else 3


def _palette(k):
    return plt.cm.tab10(np.linspace(0, 1, 10))[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=None, help="force number of clusters")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    X = build_features()
    systems = X.index.to_list()
    features = X.columns.to_list()
    Z = StandardScaler().fit_transform(X.values)
    link = linkage(Z, method="ward")
    k = _pick_k(Z, link, args.k)
    labels = fcluster(link, k, criterion="maxclust")
    sil = silhouette_score(Z, labels) if len(set(labels)) > 1 else float("nan")
    print(f"[cluster] {len(systems)} TCRs · {len(features)} features · k={k} "
          f"(silhouette {sil:.2f})")

    # relabel clusters 1..k by size (1 = largest) for stable colouring
    order_lab = pd.Series(labels).value_counts().index.to_list()
    remap = {old: i + 1 for i, old in enumerate(order_lab)}
    labels = np.array([remap[x] for x in labels])
    colors = _palette(k)

    assign = pd.DataFrame({"system": systems, "cluster": labels}).sort_values(
        ["cluster", "system"]).reset_index(drop=True)
    assign.to_csv(OUT / "tcr_clusters.csv", index=False)
    Xc = X.copy(); Xc["cluster"] = labels
    Xc.groupby("cluster").mean().round(3).to_csv(OUT / "cluster_feature_means.csv")

    # ---- dendrogram ----
    ct = link[-(k - 1), 2] - 1e-9 if k > 1 else None
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(systems)), 5))
    dendrogram(link, labels=systems, color_threshold=ct, above_threshold_color="0.6", ax=ax)
    ax.set_ylabel("Ward distance")
    ax.set_title(f"TCR clustering by flexibility + geometry  (k={k}, silhouette {sil:.2f})")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    fig.tight_layout(); fig.savefig(OUT / "tcr_dendrogram.png", dpi=150); plt.close(fig)

    # ---- PCA scatter ----
    pca = PCA(n_components=2).fit(Z)
    Y = pca.transform(Z); evr = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    for c in range(1, k + 1):
        m = labels == c
        ax.scatter(Y[m, 0], Y[m, 1], color=colors[c - 1], s=55, edgecolors="k",
                   linewidths=0.4, label=f"cluster {c} (n={m.sum()})")
    for i, s in enumerate(systems):
        ax.annotate(s, (Y[i, 0], Y[i, 1]), fontsize=6, alpha=0.7,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({evr[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({evr[1]:.0f}%)")
    ax.set_title("TCR feature-space (PCA) coloured by cluster")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "tcr_cluster_pca.png", dpi=150); plt.close(fig)

    # ---- feature heatmap (z-scored), rows ordered by dendrogram, cols by block ----
    leaves = dendrogram(link, no_plot=True)["leaves"]
    Zdf = pd.DataFrame(Z, index=systems, columns=features).iloc[leaves]
    lab_ord = labels[leaves]
    col_order = ([f for f in features if f.startswith("flex:")] +
                 [f for f in features if f.startswith("char:")] +
                 [f for f in features if f.startswith("geom:")])
    Zdf = Zdf[col_order]
    fig, (cax, ax) = plt.subplots(1, 2, figsize=(0.34 * len(col_order) + 3, 0.32 * len(systems) + 2),
                                  gridspec_kw={"width_ratios": [0.04, 1]})
    cax.imshow(lab_ord.reshape(-1, 1), aspect="auto",
               cmap=plt.cm.colors.ListedColormap(colors))
    cax.set_xticks([]); cax.set_yticks(range(len(systems)))
    cax.set_yticklabels(Zdf.index, fontsize=7); cax.set_title("cl.", fontsize=8)
    im = ax.imshow(Zdf.values, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([c.split(":", 1)[1] for c in col_order], rotation=90, fontsize=7)
    ax.set_yticks([])
    # block dividers
    nflex = sum(c.startswith("flex:") for c in col_order)
    nchar = sum(c.startswith("char:") for c in col_order)
    for xb in (nflex - 0.5, nflex + nchar - 0.5):
        ax.axvline(xb, color="k", lw=1.2)
    fig.colorbar(im, ax=ax, label="z-score", fraction=0.025)
    ax.set_title("Per-TCR features (z-scored) — grouped: flexibility | CDR3 char | geometry")
    fig.tight_layout(); fig.savefig(OUT / "tcr_feature_heatmap.png", dpi=150); plt.close(fig)

    # ---- console summary: what defines each cluster ----
    means = Xc.groupby("cluster").mean()
    gmean, gstd = X.mean(), X.std(ddof=0) + 1e-9
    print("\nDefining features per cluster (top z-scored deviations):")
    for c in range(1, k + 1):
        z = ((means.loc[c] - gmean) / gstd).sort_values()
        hi = "  ".join(f"{f.split(':',1)[1]}↑{z[f]:+.1f}" for f in z.index[-3:][::-1])
        lo = "  ".join(f"{f.split(':',1)[1]}↓{z[f]:+.1f}" for f in z.index[:3])
        mem = ", ".join(assign.loc[assign.cluster == c, "system"])
        print(f"  cluster {c} (n={(labels==c).sum()}): {mem}")
        print(f"      high: {hi}\n      low : {lo}")
    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
