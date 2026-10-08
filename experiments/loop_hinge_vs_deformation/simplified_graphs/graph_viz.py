#!/usr/bin/env python
"""Visualise simplified CDR / framework distance graphs (networkx + block distance matrices).

draw_cdr5(g, out) -> 5 panels for a CDR graph:
    1  node-link overview            loop / flank / framework nodes coloured, at real (PCA-2D) positions
    2  intra-loop distance matrix    loop x loop      (d_LL, the loop's internal shape)
    3  loop vs framework             loop x frame     (d_LF, the loop's pose vs the rigid scaffold)
    4  loop vs flanking              loop x flank     (the transition residues)
    5  flanking vs framework         flank x frame
Matrices show D_mean (mean CA-CA distance over the MD, Angstrom).

draw_framework(g, out) -> 3 panels for a framework-only graph: node-link, D_mean, D_std.
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import networkx as nx

COL = {"loop": "#DD8452", "flank": "#55A868", "frame": "#4C72B0", "frame2": "#8172B3"}


def pca2d(X):
    Xc = np.asarray(X) - np.asarray(X).mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def _node_colors(g):
    lc = g["chain"]                                   # loop chain (or 'AB' for framework graph)
    out = []
    for t, ch in zip(g["node_type"], g["node_chain"]):
        if t == "frame" and len(set(g["node_chain"].tolist())) > 1 and ch != lc[0]:
            out.append(COL["frame2"])                 # framework from the OTHER chain (alpha in a beta-CDR fig)
        else:
            out.append(COL[t])
    return out


def _nodelink(ax, g, label_mode="all"):
    N = len(g["node_type"]); pos = {i: xy for i, xy in enumerate(pca2d(g["mean_xyz"]))}
    G = nx.Graph(); G.add_nodes_from(range(N))
    Dstd = np.asarray(g["D_std"]); iu = np.triu_indices(N, 1)
    edges = list(zip(iu[0].tolist(), iu[1].tolist()))                  # COMPLETE graph: every pair is an edge
    alpha = float(min(.4, 22.0 / max(N, 1)))                           # fade edges as the graph gets denser
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=Dstd[iu], edge_cmap=plt.cm.viridis,
                           width=.4, alpha=alpha, ax=ax)               # colour by fluctuation (dark=rigid edge)
    nx.draw_networkx_nodes(G, pos, node_color=_node_colors(g), node_size=170,
                           edgecolors="k", linewidths=.3, ax=ax)
    if label_mode != "none":
        want = range(N) if label_mode == "all" else \
            [i for i in range(N) if g["node_type"][i] in ("loop", "flank")]   # only label loop/flank when dense
        nx.draw_networkx_labels(G, pos, {i: str(g["node_label"][i]) for i in want}, font_size=6, ax=ax)
    ax.set_axis_off()
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", ls="", mfc=COL[k], mec="k", ms=9, label=v) for k, v in
           [("loop", f"loop ({g['n_loop']})"), ("flank", f"flank ({g.get('n_flank', 0)})"),
            ("frame", f"framework ({g['n_frame']})")]]
    if len(set(g["node_chain"].tolist())) > 1:
        leg.append(Line2D([0], [0], marker="o", ls="", mfc=COL["frame2"], mec="k", ms=9, label="framework (other chain)"))
    ax.legend(handles=leg, loc="upper right", fontsize=8, framealpha=.9)


def _block(ax, g, rmask, cmask, title, cmap="viridis"):
    Dm = np.asarray(g["D_mean"]); lab = g["node_label"]
    ri, ci = np.where(rmask)[0], np.where(cmask)[0]
    if len(ri) == 0 or len(ci) == 0:
        ax.text(.5, .5, f"{title}\n\n(no flanking residues\nfor this CDR)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="0.4"); ax.set_axis_off(); return
    A = Dm[np.ix_(ri, ci)]
    im = ax.imshow(A, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=.046, pad=.02, label="mean CA–CA dist (Å)")
    ax.set_yticks(range(len(ri))); ax.set_yticklabels([lab[i] for i in ri], fontsize=6)
    if len(ci) > 25:
        step = max(1, len(ci) // 20); tk = list(range(0, len(ci), step))
        ax.set_xticks(tk); ax.set_xticklabels([lab[ci[t]] for t in tk], rotation=90, fontsize=5)
    else:
        ax.set_xticks(range(len(ci))); ax.set_xticklabels([lab[i] for i in ci], rotation=90, fontsize=6)
    ax.set_title(title, fontsize=9)


def draw_cdr5(g, out, cutoff=14.0):
    tL = g["node_type"] == "loop"; tK = g["node_type"] == "flank"; tF = g["node_type"] == "frame"
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1, 1], hspace=.28, wspace=.32)
    axG = fig.add_subplot(gs[:, 0])
    _nodelink(axG, g, label_mode="all" if len(g["node_type"]) <= 45 else "loopflank")
    fwtxt = "beta only" if len(set(g["node_chain"].tolist())) == 1 else "alpha+beta combined"
    axG.set_title(f"{g['sysid']}  {g['chain']}_{g['cdr']}  distance graph  ·  framework = {fwtxt}\n"
                  f"{g['n_loop']} loop + {g.get('n_flank',0)} flank + {g['n_frame']} framework nodes  ·  "
                  f"{g['n_frames']} frames", fontsize=10)
    _block(fig.add_subplot(gs[0, 1]), g, tL, tL, "① intra-loop  (loop × loop, d_LL)")
    _block(fig.add_subplot(gs[0, 2]), g, tL, tF, "② loop × framework  (d_LF)")
    _block(fig.add_subplot(gs[1, 1]), g, tL, tK, "③ loop × flanking")
    _block(fig.add_subplot(gs[1, 2]), g, tK, tF, "④ flanking × framework")
    fig.suptitle("Simplified alignment-free CDR distance graph — nodes = CA atoms, edges = CA–CA distances",
                 y=1.01, fontsize=13)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def draw_framework(g, out):
    allres = g["cdr"] == "ALLRES"
    what = "ALL variable-domain residues" if allres else "framework rigid residues"
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.8), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    _nodelink(ax[0], g, label_mode="none")
    ax[0].set_title(f"{g['sysid']}  {what} (chain {g['chain']})\n{len(g['node_type'])} nodes · complete graph (all-to-all)"
                    + ("  · orange = CDR" if allres else ""), fontsize=10)
    lab = g["node_label"]; N = len(lab)
    for k, (M, ttl, cm) in enumerate([(np.asarray(g["D_mean"]), "D_mean · distance (Å)", "magma"),
                                       (np.asarray(g["D_std"]), "D_std · fluctuation (Å) — small = rigid", "inferno")]):
        im = ax[k + 1].imshow(M, cmap=cm); fig.colorbar(im, ax=ax[k + 1], fraction=.046, pad=.02)
        step = max(1, N // 25); tk = list(range(0, N, step))
        ax[k + 1].set_xticks(tk); ax[k + 1].set_xticklabels([lab[t] for t in tk], rotation=90, fontsize=5)
        ax[k + 1].set_yticks(tk); ax[k + 1].set_yticklabels([lab[t] for t in tk], fontsize=5)
        ax[k + 1].set_title(ttl, fontsize=10)
    fig.suptitle(f"Distance-graph scaffold — chain {g['chain']}, {what}"
                 + ("  (CDR loops show as bright D_std stripes)" if allres else "  (D_std small ⇒ super-rigid)"),
                 y=1.03, fontsize=12)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out
