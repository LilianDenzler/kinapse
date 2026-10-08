#!/usr/bin/env python
"""Express EVERY loop-motion mode as an RMS Cartesian displacement in ANGSTROMS, so hinge / twist / sway /
translation / deformation are all directly comparable and (near-)additive: D_total^2 ≈ sum of the pieces^2.

Coordinate space, framework-aligned, relative to the medoid reference L0 (centroid L0cen):
  rigid rotation about the centroid, angle ω(t)=rotvec(Kabsch); split ω onto loop-frame axes ĉ/ĥ/n̂.
  A rotation of ω_k about axis û_k moves atom i by  ω_k * (û_k × r_i')   (r_i' = L0_i - L0cen).
  -> D_k = std_t(ω_k) * sqrt(mean_i |û_k × r_i'|^2)         [rad * Å = Å],  the RMS displacement of that mode.
  hinge = about ĉ (tip elevation), twist = about ĥ, sway = about n̂.
  translation D_trans = RMS displacement of the loop centroid (the rigid DOF the rotation misses).
  deformation D_deform = RMS residual after the full 6-DOF rigid fit (what NO rigid motion explains).
-> results_modesA.json + figures/modes_angstrom.png"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from geom_hinge import kabsch

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}      # anchored stem foot; the loop pivots here (not the centroid)


def analyze(loop, R, clampside):
    T, N, _ = loop.shape
    Rref = R[0] - R[0].mean(0)
    Lf = np.empty_like(loop)
    for t in range(T):
        Lf[t] = (loop[t] - R[t].mean(0)) @ kabsch(R[t] - R[t].mean(0), Rref).T + R[0].mean(0)
    dLL = np.linalg.norm(loop[:, np.triu_indices(N, 1)[0]] - loop[:, np.triu_indices(N, 1)[1]], axis=-1)
    L0 = Lf[int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1)))]
    L0cen = L0.mean(0); L0c = L0 - L0cen
    fN, fC = L0[0], L0[-1]
    piv = fC if clampside == "C" else fN                                # PIVOT = clamp foot (physical anchor)
    fi = N - 1 if clampside == "C" else 0
    c = fC - fN; c /= np.linalg.norm(c)
    perp = (L0 - piv) - ((L0 - piv) @ c)[:, None] * c
    ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
    n = np.cross(c, h); n /= np.linalg.norm(n)
    rel0 = L0 - piv                                                     # lever arms measured from the clamp
    Dk = {"c": np.zeros(N), "h": np.zeros(N), "n": np.zeros(N)}; Edef = np.zeros(N)   # PER-RESIDUE accumulation
    for t in range(T):
        Q = kabsch(L0c, Lf[t] - Lf[t].mean(0)); w = Rot.from_matrix(Q).as_rotvec()
        for k, u in (("c", c), ("h", h), ("n", n)):                     # displacement if ONLY this mode's rotation applied about the clamp
            Rk = Rot.from_rotvec((w @ u) * u).as_matrix()
            Dk[k] += ((rel0 @ Rk.T - rel0) ** 2).sum(1)                 # per-atom squared displacement
        Edef += ((Lf[t] - (L0c @ Q.T + Lf[t].mean(0))) ** 2).sum(1)

    def summarise(sq):                                                  # sq = Σ_frames per-atom squared displacement
        rms_i = np.sqrt(sq / T)                                         # per-residue RMS displacement (N,)
        return float(np.sqrt((rms_i ** 2).mean())), float(rms_i.sum())  # (per-residue RMS ; TOTAL over loop = Σ_i)
    D_hinge, T_hinge = summarise(Dk["c"]); D_twist, T_twist = summarise(Dk["h"]); D_sway, T_sway = summarise(Dk["n"])
    D_deform, T_deform = summarise(Edef)
    foot = Lf[:, fi]                                                    # the clamp ATOM's own motion vs framework = honest clamp slide
    D_slide = float(np.sqrt(((foot - foot.mean(0)) ** 2).sum(1).mean()))
    D_total = float(np.sqrt(((Lf - L0) ** 2).sum(-1).mean()))
    D_rigid = float(np.sqrt(max(D_total ** 2 - D_deform ** 2, 0.0)))    # clean: rigid ⟂ deform, adds to total
    return dict(hinge=D_hinge, twist=D_twist, sway=D_sway, clamp_slide=D_slide, deform=D_deform,
                rigid=D_rigid, total=D_total, nres=int(N),
                hinge_tot=T_hinge, twist_tot=T_twist, sway_tot=T_sway, deform_tot=T_deform)


def main():
    tcrs = [os.path.basename(f)[:4] for f in sorted(glob.glob(f"{HERE}/results_swing/*.npz"))]
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    res = {}
    for t in tcrs:
        try:
            tv, xyz, imap = load_md(t)
            row = {}
            for cdr in CDRS:
                ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
                lk = sorted(k for k in imap[ch] if lo <= k <= hi)
                loop = xyz[:, np.array([imap[ch][k] for k in lk])]
                Rr = xyz[:, np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])]
                row[cdr] = analyze(loop, Rr, CLAMP[cdr[2:]])
            res[t] = row; print(f"{t} ok", flush=True)
        except Exception as e:
            print(f"{t} ERR {type(e).__name__}: {str(e)[:90]}", flush=True)
    json.dump(res, open(f"{HERE}/results_modesA.json", "w"))
    print(f"\n{len(res)} TCRs -> results_modesA.json")
    plot(res)


def _bar(res, keys, title, ylab, out, extra=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tcrs = list(res); x = np.arange(6); w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(13, 6))
    for j, (k, col, lab) in enumerate(keys):
        vals = [np.median([res[t][c][k] for t in tcrs]) for c in CDRS]
        ax.bar(x + (j - (len(keys) - 1) / 2) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(CDRS, fontsize=10)
    ax.set_ylabel(ylab)
    ax.legend(fontsize=9, ncol=len(keys), loc="upper center", bbox_to_anchor=(0.5, 1.08))
    ax.set_title(title, y=1.10, fontsize=11)
    if extra:
        for xi, c in enumerate(CDRS):
            ax.annotate(extra(c, tcrs), (xi, 0), (0, -28), textcoords="offset points", ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


def plot(res):
    tcrs = list(res)
    G, P, O, GY, R = "#2E7D32", "#7B4FA3", "#F0A030", "#888888", "#C0392B"
    def nres(c, ts):
        ns = [res[t][c]["nres"] for t in ts]
        return f"n={int(min(ns))}–{int(max(ns))}\n(med {int(np.median(ns))})"
    # (1) per-residue RMS (length-independent)
    _bar(res, [("hinge", G, "hinge (ĉ)"), ("twist", P, "twist (ĥ)"), ("sway", O, "sway (n̂)"),
               ("clamp_slide", GY, "clamp slide"), ("deform", R, "deformation")],
         "PER-RESIDUE RMS displacement (Å) — length-independent, clamp-pivoted", "Å per residue (median over 22 TCRs)",
         f"{HERE}/figures/modes_angstrom.png")
    # (2) TOTAL over the whole loop (Σ over residues; length-dependent)
    _bar(res, [("hinge_tot", G, "hinge (ĉ)"), ("twist_tot", P, "twist (ĥ)"), ("sway_tot", O, "sway (n̂)"),
               ("deform_tot", R, "deformation")],
         "TOTAL displacement summed over the whole loop (Å) — length-DEPENDENT", "Σ over residues, Å (median over 22 TCRs)",
         f"{HERE}/figures/modes_total_angstrom.png", extra=nres)
    # (3) TOTAL normalised by loop length (Σ / n_res)  -> length-comparable
    resn = {t: {c: {**res[t][c],
                    "hinge_n": res[t][c]["hinge_tot"] / res[t][c]["nres"],
                    "twist_n": res[t][c]["twist_tot"] / res[t][c]["nres"],
                    "sway_n": res[t][c]["sway_tot"] / res[t][c]["nres"],
                    "deform_n": res[t][c]["deform_tot"] / res[t][c]["nres"]} for c in CDRS} for t in tcrs}
    _bar(resn, [("hinge_n", G, "hinge (ĉ)"), ("twist_n", P, "twist (ĥ)"), ("sway_n", O, "sway (n̂)"),
                ("deform_n", R, "deformation")],
         "TOTAL ÷ loop length (Å per residue) — length-normalised, comparable across CDRs", "Å per residue (median over 22 TCRs)",
         f"{HERE}/figures/modes_norm_angstrom.png")
    print(f"\n{'CDR':8}{'n':>4}{'  hinge_tot':>11}{'twist_tot':>10}{'sway_tot':>9}{'deform_tot':>11}{'  totalRMS':>10}")
    for c in CDRS:
        v = {k: np.median([res[t][c][k] for t in tcrs]) for k in ("nres", "hinge_tot", "twist_tot", "sway_tot", "deform_tot", "total")}
        print(f"{c:8}{int(v['nres']):>4}{v['hinge_tot']:11.2f}{v['twist_tot']:10.2f}{v['sway_tot']:9.2f}{v['deform_tot']:11.2f}{v['total']:10.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(json.load(open(f"{HERE}/results_modesA.json")))
    else:
        main()
