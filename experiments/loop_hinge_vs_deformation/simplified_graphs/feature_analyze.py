#!/usr/bin/env python
"""Stage 2 of feature discovery: merge the feature panel (results_features.json) with the motion TARGETS
(results_angleamp / results_rigid_deform / results_modes_atlas), then rank which features predict the hinge /
deformation / DOF / entropy. Univariate Spearman (pooled + per-CDR), permutation p, Benjamini-Hochberg FDR.
Honest for n~22: hypothesis-generating. -> figures/feature_discovery.png + printed ranked table."""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
FEATS = ["N", "feet_len", "tip_lever", "rg", "n_gly", "n_pro", "n_arom", "n_chg", "frac_hphob",
         "jmotif_gly", "gly_flank", "pro_flank", "flank_dihvar",
         "hb_cross_stem", "hb_stem_fw", "hb_intra_loop", "hb_loop_fw",
         "clamp_rmsf", "free_rmsf", "tip_rmsf", "loop_rmsf", "apex_anchor_dccm"]


def load_targets():
    F = json.load(open(f"{HERE}/results_features.json"))
    aa = json.load(open(f"{HERE}/results_angleamp.json"))
    rd = json.load(open(f"{HERE}/results_rigid_deform.json"))
    at = json.load(open(f"{HERE}/results_modes_atlas.json"))
    rows = []  # (tcr, cdr, feats dict, targets dict)
    for t in F:
        for c in CDRS:
            if c not in F[t]:
                continue
            tg = {}
            try:
                tg["hinge_amp"] = aa[t][c]["hinge"]["p95"]; tg["twist_amp"] = aa[t][c]["twist"]["p95"]
            except Exception:
                tg["hinge_amp"] = tg["twist_amp"] = np.nan
            try:
                tg["deform_frac"] = rd[t][c]["deform"] / rd[t][c]["total"]
            except Exception:
                tg["deform_frac"] = np.nan
            try:
                tg["PR_dist"] = at[t][c]["PR_dist"]; tg["S_deform"] = at[t][c]["S_deform"]; tg["mode1_rigid"] = at[t][c]["top"][0]["rigid"]
            except Exception:
                tg["PR_dist"] = tg["S_deform"] = tg["mode1_rigid"] = np.nan
            rows.append((t, c, F[t][c], tg))
    return rows


def perm_p(x, y, rho, n=2000, seed=0):
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n):
        if abs(spearmanr(x, rng.permutation(y))[0]) >= abs(rho):
            cnt += 1
    return (cnt + 1) / (n + 1)


def bh(pvals):
    p = np.array(pvals); m = len(p); order = np.argsort(p); q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k); q[i] = prev
    return q


def analyze(rows, scope, targets):
    sel = [r for r in rows if (scope == "all" or r[1] in scope)]
    out = []
    for tgt in targets:
        yv = np.array([r[3].get(tgt, np.nan) for r in sel], float)
        for f in FEATS:
            xv = np.array([r[2].get(f, np.nan) for r in sel], float)
            mask = np.isfinite(xv) & np.isfinite(yv)
            if mask.sum() < 6 or np.nanstd(xv[mask]) == 0:
                continue
            rho, _ = spearmanr(xv[mask], yv[mask])
            if np.isnan(rho):
                continue
            out.append([tgt, f, float(rho), int(mask.sum()), xv[mask], yv[mask]])
    # permutation p + FDR
    for row in out:
        row.append(perm_p(row[4], row[5], row[2]))
    q = bh([row[-1] for row in out])
    for row, qi in zip(out, q):
        row.append(float(qi))
    return out


def main():
    rows = load_targets()
    targets = ["hinge_amp", "twist_amp", "deform_frac", "PR_dist", "S_deform", "mode1_rigid"]
    print(f"loaded {len(rows)} (TCR,CDR) rows\n")
    for scope_name, scope in [("CDR3 (A_CDR3+B_CDR3)", ["A_CDR3", "B_CDR3"]), ("all 6 CDRs", "all")]:
        res = analyze(rows, scope, targets)
        res.sort(key=lambda r: -abs(r[2]))
        print(f"==== scope: {scope_name} ====  (ranked |Spearman rho|; q=BH-FDR)")
        print(f"{'target':12}{'feature':16}{'rho':>7}{'n':>4}{'perm_p':>9}{'q(FDR)':>9}")
        for tgt, f, rho, n, _, _, p, q in res[:18]:
            star = "*" if q < 0.1 else ""
            print(f"{tgt:12}{f:16}{rho:+7.2f}{n:4d}{p:9.3f}{q:9.3f} {star}", flush=True)
        print()
    plot(rows, targets)


def plot(rows, targets):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    scope = ["A_CDR3", "B_CDR3"]; sel = [r for r in rows if r[1] in scope]
    M = np.full((len(FEATS), len(targets)), np.nan)
    for ti, tgt in enumerate(targets):
        yv = np.array([r[3].get(tgt, np.nan) for r in sel], float)
        for fi, f in enumerate(FEATS):
            xv = np.array([r[2].get(f, np.nan) for r in sel], float)
            mask = np.isfinite(xv) & np.isfinite(yv)
            if mask.sum() >= 6 and np.nanstd(xv[mask]) > 0:
                M[fi, ti] = spearmanr(xv[mask], yv[mask])[0]
    fig, ax = plt.subplots(figsize=(9, 11))
    import numpy.ma as ma
    im = ax.imshow(ma.masked_invalid(M), aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(len(targets))); ax.set_xticklabels(targets, rotation=30, fontsize=9)
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=8)
    for fi in range(len(FEATS)):
        for ti in range(len(targets)):
            if np.isfinite(M[fi, ti]):
                ax.text(ti, fi, f"{M[fi,ti]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[fi, ti]) > 0.5 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("Spearman rho (CDR3 loops, n=44)", fontsize=9)
    ax.set_title("Feature discovery: which flank/loop features predict loop motion?\n(CDR3α+CDR3β, 22 TCRs)", fontsize=11, fontweight="bold")
    out = f"{HERE}/figures/feature_discovery.png"
    fig.tight_layout(); fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out, flush=True)


if __name__ == "__main__":
    main()
