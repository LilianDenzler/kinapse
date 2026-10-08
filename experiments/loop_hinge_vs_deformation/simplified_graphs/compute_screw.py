#!/usr/bin/env python
"""Loop RIGID-motion descriptors (screw / Chasles parameters) for one TCR, per frame, deformation-excluded.
For each frame the robust (fluctuation-weighted) rigid fit gives a transform x->Rx+τ; describe it invariantly as a SCREW:
  angle θ        = rotation amount (deg)                     -> the swing amplitude
  axis direction = |û·ĉ|,|û·ĥ|,|û·n̂|  -> hinge / twist / sway  (which way it rotates; ONE direction, no carving)
  pitch          = τ·û  (Å)  -> translation ALONG the axis    -> 0 = pure rotation (hinge); ≠0 = screw-glide
  axis→clamp     = distance from the clamp to the screw axis line (Å) -> where the loop pivots
DEFORMATION (separate, clean) = RMS deviation of d_LL from the reference. -> figures/screw_<sys>.png + results_screw_<sys>.npz"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from graph_build import load_md, CDR_RANGES, HERE
from viz_ensemble import superpose_all

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
CLAMP = {"CDR1": "C", "CDR2": "N", "CDR3": "N"}
NSUB = 200


def robust_kabsch(P, Q, iters=6):
    w = np.ones(len(P))
    for _ in range(iters):
        ws = w.sum(); Pc = (w[:, None] * P).sum(0) / ws; Qc = (w[:, None] * Q).sum(0) / ws
        A = P - Pc; B = Q - Qc
        U, _, Vt = np.linalg.svd((w[:, None] * A).T @ B)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        resid = np.linalg.norm(A @ R.T - B, axis=1)
        c = np.median(resid) + 1e-6; w = c ** 2 / (resid ** 2 + c ** 2)
    return R, Pc, Qc


def main(sysid="3SKN"):
    tv, xyz, imap = load_md(sysid)
    rig = json.load(open(f"{HERE}/rigid_framework.json"))
    rng = np.random.default_rng(0); I = np.eye(3)
    res = {}
    print(f"{sysid}  RIGID screw descriptors (deformation-excluded via robust fit)")
    print(f"{'CDR':8}{'θ(°)':>7}{'deform(Å)':>10}{'pitch(Å)':>9}{'axis→clamp(Å)':>14}{'  hinge/twist/sway':>19}")
    for cdr in CDRS:
        ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
        fw = np.array([imap[ch][r] for r in rig["chain_" + ch]["ultra_rigid"] if r in imap[ch]])
        supr = superpose_all(xyz, fw)
        lk = sorted(k for k in imap[ch] if lo <= k <= hi)
        loop = supr[:, np.array([imap[ch][k] for k in lk])]
        T, N, _ = loop.shape; ii, jj = np.triu_indices(N, 1)
        dLL = np.linalg.norm(loop[:, ii] - loop[:, jj], axis=-1)
        med = int(np.argmin(((dLL - dLL.mean(0)) ** 2).sum(1))); L0 = loop[med]; dref = dLL[med]
        fN, fC = L0[0], L0[-1]; clamp = fC if CLAMP[cdr[2:]] == "C" else fN
        m = 0.5 * (fN + fC); c = fC - fN; c /= np.linalg.norm(c)
        perp = (L0 - m) - ((L0 - m) @ c)[:, None] * c
        ai = int(np.argmax(np.linalg.norm(perp, axis=1))); h = perp[ai]; h /= np.linalg.norm(h)
        n = np.cross(c, h); n /= np.linalg.norm(n)
        idx = rng.choice(T, min(NSUB, T), replace=False)
        theta = np.empty(len(idx)); deform = np.empty(len(idx)); pitch = np.empty(len(idx))
        axclamp = np.empty(len(idx)); char = np.empty((len(idx), 3))
        for s, t in enumerate(idx):
            deform[s] = np.sqrt(((dLL[t] - dref) ** 2).mean())
            R, Pc, Qc = robust_kabsch(L0, loop[t])
            w = Rot.from_matrix(R).as_rotvec(); th = np.linalg.norm(w)
            theta[s] = np.degrees(th)
            u = w / (th + 1e-12)
            tau = Qc - R @ Pc                                  # translation of the origin under x->Rx+τ
            pitch[s] = float(tau @ u)                          # screw pitch: translation ALONG the axis
            tau_perp = tau - (tau @ u) * u
            p0 = -np.linalg.pinv(R - I) @ tau_perp             # point on the screw axis closest to origin
            d = clamp - p0; axclamp[s] = float(np.linalg.norm(d - (d @ u) * u))
            char[s] = [abs(u @ c), abs(u @ h), abs(u @ n)]
        cf = char / char.sum(1, keepdims=True)
        res[cdr] = dict(theta=theta, deform=deform, pitch=pitch, axclamp=axclamp, char=cf)
        mc = np.median(cf, 0)
        print(f"{cdr:8}{np.median(theta):7.1f}{np.median(deform):10.2f}{np.median(np.abs(pitch)):9.2f}{np.median(axclamp):14.1f}   {mc[0]:.2f}/{mc[1]:.2f}/{mc[2]:.2f}")
    np.savez(f"{HERE}/results_screw_{sysid}.npz", **{f"{c}_{k}": res[c][k] for c in CDRS for k in res[c]})
    plot(sysid, res)


def plot(sysid, res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.arange(6)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    def box(a, key, ylab, title, absv=False):
        data = [(np.abs(res[c][key]) if absv else res[c][key]) for c in CDRS]
        bp = a.boxplot(data, positions=x, widths=.6, patch_artist=True, showfliers=False, medianprops=dict(color="k"))
        for p in bp["boxes"]:
            p.set_facecolor("#3B6EA5"); p.set_alpha(.4)
        a.set_xticks(x); a.set_xticklabels(CDRS, rotation=15, fontsize=8); a.set_ylabel(ylab); a.set_title(title, fontsize=10, fontweight="bold")

    box(ax[0, 0], "theta", "θ (deg)", "a  Rigid rotation angle (swing amplitude)")
    box(ax[0, 1], "deform", "d_LL deviation (Å)", "b  DEFORMATION (separate, clean)")
    box(ax[0, 2], "pitch", "|pitch| (Å)", "c  Screw pitch = glide along axis\n(≈0 ⇒ pure rotation, no real translation)", absv=True)
    box(ax[1, 0], "axclamp", "axis→clamp distance (Å)", "d  Where the axis sits (dist. from clamp)")
    # axis character stacked
    G, P, O = "#2E7D32", "#7B4FA3", "#F0A030"
    mc = np.array([np.median(res[c]["char"], 0) for c in CDRS])
    ax[1, 1].bar(x, mc[:, 0], color=G, label="hinge (ĉ)")
    ax[1, 1].bar(x, mc[:, 1], bottom=mc[:, 0], color=P, label="twist (ĥ)")
    ax[1, 1].bar(x, mc[:, 2], bottom=mc[:, 0] + mc[:, 1], color=O, label="sway (n̂)")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(CDRS, rotation=15, fontsize=8); ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_ylabel("axis-direction share"); ax[1, 1].legend(fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(.5, 1.0))
    ax[1, 1].set_title("e  Axis direction (which way it rotates)", fontsize=10, fontweight="bold", y=1.08)
    # rigid vs deform scale note
    ax[1, 2].scatter([np.median(res[c]["deform"]) for c in CDRS], [np.median(res[c]["theta"]) for c in CDRS], s=60, color="#3B6EA5")
    for c in CDRS:
        ax[1, 2].annotate(c, (np.median(res[c]["deform"]), np.median(res[c]["theta"])), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax[1, 2].set_xlabel("deformation (Å)"); ax[1, 2].set_ylabel("rigid rotation θ (deg)")
    ax[1, 2].set_title("f  Rigid rotation vs deformation (per CDR)", fontsize=10, fontweight="bold")
    fig.suptitle(f"{sysid}: loop RIGID-motion descriptors (screw, deformation-excluded) + deformation.  "
                 "Rigid = rotation θ about an axis (direction=hinge/twist/sway, located near clamp), pitch≈0 ⇒ pure rotation.", y=1.02, fontsize=11)
    out = f"{HERE}/figures/screw_{sysid}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("fig ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3SKN")
