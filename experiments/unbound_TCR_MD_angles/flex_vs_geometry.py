#!/usr/bin/env python
"""Decompose unbound-TCR CDR loop flexibility into ANGLE vs DEFORMATION.

For every unbound TCR MD system in ``CORY_ORIOL_MERGED_MD`` this measures how
mobile each of the six CDR loops is, as three Cα RMSF quantities (Å):

* **total**   — superpose every frame on that domain's FRAMEWORK Cα (α-framework
                for α CDRs, β-framework for β CDRs), then the loop's RMSF about its
                mean: all of the loop's motion relative to the domain. This is the
                "loop flexibility" of ``BSC_clinical_TCRs/analyse_flexibility.py``,
                here on MD.
* **deform**  — superpose every frame on the CDR LOOP *itself*; the residual RMSF
                is the loop's genuine internal shape change.
* **angle**   — take the AVERAGE loop and rigidly re-pose it to each frame: a
                Kabsch fit registered on the loop **plus its two anchor residues**
                (the conserved stem, e.g. 104/118 for CDR3) so it hinges about the
                anchored base. The RMSF of that re-posed *average* loop is pure
                reorientation — because the moved body is the rigid average shape
                it carries **no deformation by construction**. Measured directly,
                not as ``sqrt(total² − deform²)``.

``angle`` and ``deform`` are thus independent measurements. ``angle² + deform²``
recovers ~80–95 % of ``total²``; the remainder is genuine angle–deformation
coupling (reported per loop as ``quad_closure``), which a quadrature split would
have silently forced to zero. Superposition reference throughout is the **medoid**
(the real frame closest to the average), not an arbitrary frame 0.

Broken/split frames (domains not in contact) are dropped with the same Vα–Vβ
contact cutoff as the angle geometry, so the two analyses see the same frames.

The angle component is then cross-checked against the *independently* measured
CDR3 bend-angle exploration (its per-frame std from ``results/``): loops whose
flexibility is angle-dominated should be the ones whose bend angle explores most.

Outputs (``results_flex_vs_geometry/``):
* ``flex_components.csv``            — per (system, CDR): total/angle/deform RMSF +
                                       max-excursion (Å).
* ``flex_components_bars.png``       — per-CDR grouped RMSF bars: total/angle/deform.
* ``flex_by_tcr_cdr3.png``           — per-TCR CDR3, α and β of each TCR adjacent.
* ``flex_angle_vs_deform.png``       — angle vs deform scatter (ratio + magnitude).
* ``flex_exploration.png``           — typical (RMSF) vs maximum excursion, per CDR.
* ``flex_exploration_by_tcr.png``    — per-TCR CDR3 RMSF→max dumbbell.
* ``cdr_flexibility_heatmap.png``    — total loop RMSF, system × CDR.
* ``flex_vs_geometry.png`` / ``.csv``— CDR3 angle component vs measured bend std.

All bar charts are in RMSF (Å). ``angle`` and ``deform`` are independent direct
measurements (see above), so they need not sum in quadrature to ``total``.

Run (kinapse env)::

    python flex_vs_geometry.py                      # all systems
    python flex_vs_geometry.py --systems 1KGC 8YJ3
    python flex_vs_geometry.py --angles-dir results # geometry std source (default)
"""
from __future__ import annotations

import os
import sys

# exclude the ABI-broken user-site h5py (same guard as angle_exploration)
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MDAnalysis as mda
from tqdm import tqdm

from angle_exploration import (
    Consensus, resolve_template, _pick_cdr3,
    DATA_DIR, discover_systems, CONTACT_CUTOFF,
)

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results_flex_vs_geometry"
DEFAULT_ANGLES = HERE / "results"          # where <ID>_angles.csv live (bend std)

# IMGT loop definitions (Cα of these residues make up each CDR)
CDR_RANGES = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}
# IMGT framework (scaffold) regions, between the CDRs
FR_RANGES = {"FR1": (1, 26), "FR2": (39, 55), "FR3": (66, 104), "FR4": (118, 128)}
# Conserved IMGT framework residues flanking each loop (the loop's stem/base).
# The angle fit is registered on these so the loop hinges about its anchored base.
ANCHORS = {"CDR1": (26, 39), "CDR2": (55, 66), "CDR3": (104, 118)}
CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
FRS = [f"{c}_{fr}" for c in "AB" for fr in ("FR1", "FR2", "FR3", "FR4")]
REGION_SEQ = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]   # N→C order
CDR_COLORS = {"CDR1": "#4a5568", "CDR2": "#2b6cb0", "CDR3": "#c53030"}


# --------------------------------------------------------------------------- #
# RMSF with a batched Kabsch superposition
# --------------------------------------------------------------------------- #
def _kabsch_batch(P, Q):
    """Rigid transforms mapping each mobile ``P[f]`` onto the fixed ``Q``.

    P : (F, N, 3) mobile reference atoms per frame; Q : (N, 3) fixed reference.
    Returns R : (F, 3, 3), t : (F, 3) with ``R @ P[f] + t ≈ Q``.
    """
    Pm = P.mean(1)                                  # (F,3)
    Qm = Q.mean(0)                                  # (3,)
    Pc = P - Pm[:, None, :]
    Qc = Q - Qm[None, :]
    H = np.einsum("fni,nj->fij", Pc, Qc)            # (F,3,3)
    U, _, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1))
    Ut = np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", V, Ut)))   # (F,)
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("fij,fjk,fkl->fil", V, D, Ut)      # (F,3,3)
    t = Qm[None, :] - np.einsum("fij,fj->fi", R, Pm)
    return R, t


def _rmsf(coords, ref_local, target_local):
    """Per-Cα RMSF (Å) of ``target`` after superposing every frame on ``ref``.

    ``coords`` : (F, U, 3) Å; ``ref_local``/``target_local`` index into U.

    Reference = the **medoid** (the real frame closest to the ensemble average),
    not an arbitrary frame 0: a first pass aligns to frame 0 only to locate the
    average, its nearest real frame becomes the superposition target, then RMSF
    is the fluctuation about the average of the superposed ensemble.
    """
    ref = coords[:, ref_local]                                   # (F, R, 3)
    # pass 1: provisional align to frame 0 → average → medoid (frame ≈ average)
    R0, t0 = _kabsch_batch(ref, ref[0])
    ref0 = np.einsum("fij,fnj->fni", R0, ref) + t0[:, None, :]
    medoid = int(np.argmin(((ref0 - ref0.mean(0)) ** 2).sum(-1).mean(1)))
    # pass 2: superpose all frames onto that medoid (a real conformation)
    R, t = _kabsch_batch(ref, ref[medoid])
    aln = np.einsum("fij,ftj->fti", R, coords[:, target_local]) + t[:, None, :]
    return np.sqrt(((aln - aln.mean(0)) ** 2).sum(-1).mean(0))   # (T,) about the average


def _atom_mean(x):
    """Mean over the loop's Cα of a per-atom RMSF array (matches analyse_flexibility.py)."""
    return float(np.mean(x))


def _rmsf_aligned(X):
    """Per-atom RMSF (Å) of already-superposed coords ``X`` (F,N,3), about the mean."""
    return np.sqrt(((X - X.mean(0)) ** 2).sum(-1).mean(0))


def _fw_transforms(coords, fw_local):
    """Per-frame transforms superposing each frame on the framework medoid.

    Medoid = the real frame closest to the framework average (a first align to
    frame 0 only locates that average). Returns R (F,3,3), t (F,3).
    """
    fw = coords[:, fw_local]
    R0, t0 = _kabsch_batch(fw, fw[0])
    fw0 = np.einsum("fij,fnj->fni", R0, fw) + t0[:, None, :]
    med = int(np.argmin(((fw0 - fw0.mean(0)) ** 2).sum(-1).mean(1)))
    return _kabsch_batch(fw, fw[med])


def _kabsch_ref_to_frames(ref, tgt):
    """Rigid transforms mapping a fixed ``ref`` (N,3) onto each per-frame ``tgt`` (F,N,3).

    Returns R (F,3,3), t (F,3) with ``R[f] @ ref + t[f] ≈ tgt[f]``.
    """
    rc = ref.mean(0); Rr = ref - rc
    tc = tgt.mean(1); Tt = tgt - tc[:, None, :]
    H = np.einsum("ni,fnj->fij", Rr, Tt)
    U, _, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1)); Ut = np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", V, Ut)))
    D = np.zeros((len(tgt), 3, 3)); D[:, 0, 0] = D[:, 1, 1] = 1.0; D[:, 2, 2] = d
    R = np.einsum("fij,fjk,fkl->fil", V, D, Ut)
    t = tc - np.einsum("fij,j->fi", R, rc)          # rc is (3,)
    return R, t


def _superpose(coords, ref_local, target_local):
    """Superpose ``target`` on ``ref`` (medoid reference); return aligned target (F,T,3)."""
    ref = coords[:, ref_local]
    R0, t0 = _kabsch_batch(ref, ref[0])
    ref0 = np.einsum("fij,fnj->fni", R0, ref) + t0[:, None, :]
    med = int(np.argmin(((ref0 - ref0.mean(0)) ** 2).sum(-1).mean(1)))
    R, t = _kabsch_batch(ref, ref[med])
    return np.einsum("fij,ftj->fti", R, coords[:, target_local]) + t[:, None, :]


def _excursion(X):
    """Per-frame RMSD (Å) of superposed coords ``X`` (F,N,3) from the average.

    Its mean over frames ~ the (RMS-)RMSF; its **max** over frames is the single
    most extreme conformation the loop reaches — the tip of its exploration.
    """
    dev = X - X.mean(0)
    return np.sqrt((dev ** 2).sum(-1).mean(1))          # (F,)


# --------------------------------------------------------------------------- #
# Per-system flexibility decomposition
# --------------------------------------------------------------------------- #
def _region_indices(u, tmpl):
    """Global Cα atom indices for framework, each CDR loop + anchors, and each FR."""
    reg = {"fwA": np.asarray(tmpl.idxA), "fwB": np.asarray(tmpl.idxB)}
    for chain, md_chain, cmap in (("A", tmpl.a_md, tmpl.map_a),
                                  ("B", tmpl.b_md, tmpl.map_b)):
        for cdr, (lo, hi) in CDR_RANGES.items():
            reg[f"{chain}_{cdr}"] = _pick_cdr3(u, md_chain, cmap, lo, hi)[0]
            a_lo, a_hi = ANCHORS[cdr]                       # the two flanking stem residues
            an = [_pick_cdr3(u, md_chain, cmap, a, a)[0] for a in (a_lo, a_hi)]
            reg[f"{chain}_{cdr}_anchor"] = np.concatenate([x for x in an if len(x)]) \
                if any(len(x) for x in an) else np.array([], int)
        for fr, (lo, hi) in FR_RANGES.items():
            reg[f"{chain}_{fr}"] = _pick_cdr3(u, md_chain, cmap, lo, hi)[0]
    return reg


def _region_metrics(C, fw, region_local, anchor_local):
    """total/angle/deform RMSF + excursion stats + per-frame traces for one region.

    ``fw`` is either a precomputed ``(R, t)`` framework transform (reused across a
    chain's CDRs) or an array of framework Cα local indices to superpose on (used
    for FRs, each aligned on the *rest* of the framework). ``anchor_local`` pins
    the base for the direct-angle re-pose (empty for FRs → the region is its own base).
    Returns (metrics_dict, D_total, D_angle, D_deform) — the D_* are per-frame.
    """
    R, t = fw if isinstance(fw, tuple) else _fw_transforms(C, fw)
    reg_fw = np.einsum("fij,fnj->fni", R, C[:, region_local]) + t[:, None, :]     # framework-aligned
    reg_al = _superpose(C, region_local, region_local)                            # region-aligned (deform)
    base_fw = (np.concatenate([np.einsum("fij,fnj->fni", R, C[:, anchor_local]) + t[:, None, :], reg_fw], axis=1)
               if len(anchor_local) else reg_fw)
    Rr, tr = _kabsch_ref_to_frames(base_fw.mean(0), base_fw)
    ang_c = np.einsum("fij,nj->fni", Rr, reg_fw.mean(0)) + tr[:, None, :]         # re-posed average (angle)

    rt, ra, rd = _rmsf_aligned(reg_fw), _rmsf_aligned(ang_c), _rmsf_aligned(reg_al)
    Dt, Da, Dd = _excursion(reg_fw), _excursion(ang_c), _excursion(reg_al)
    pct = lambda D, q: float(np.percentile(D, q))
    msf_t, msf_a, msf_d = float((rt ** 2).sum()), float((ra ** 2).sum()), float((rd ** 2).sum())
    m = {
        "rmsf_total_A": _atom_mean(rt), "rmsf_angle_A": _atom_mean(ra), "rmsf_deform_A": _atom_mean(rd),
        "max_total_A": float(Dt.max()), "max_angle_A": float(Da.max()), "max_deform_A": float(Dd.max()),
        "exc_angle_p10": pct(Da, 10), "exc_angle_p50": pct(Da, 50), "exc_angle_p90": pct(Da, 90),
        "exc_deform_p10": pct(Dd, 10), "exc_deform_p50": pct(Dd, 50), "exc_deform_p90": pct(Dd, 90),
        "angle_frac": (msf_a / msf_t) if msf_t > 0 else np.nan,
        "quad_closure": ((msf_a + msf_d) / msf_t) if msf_t > 0 else np.nan,
    }
    return m, Dt, Da, Dd


def process_system(pdbid, con, target_frames, stride_override, contact_cutoff):
    d = DATA_DIR / pdbid
    u = mda.Universe((d / f"{pdbid}.pdb").as_posix(), (d / f"{pdbid}.xtc").as_posix())
    n_frames = len(u.trajectory)
    stride = stride_override or max(1, n_frames // target_frames)

    # renumber once from frame 0 (robust α/β pairing)
    u.trajectory[0]
    tmp = DEFAULT_OUT / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    frame0 = tmp / f"{pdbid}_f0.pdb"
    u.atoms.write(frame0.as_posix())
    ref_positions = u.atoms.positions.astype(float)
    tmpl, how = resolve_template(u, frame0.as_posix(), con, ref_positions)
    frame0.unlink(missing_ok=True)

    reg = _region_indices(u, tmpl)
    union = np.unique(np.concatenate([v for v in reg.values() if len(v)]))
    loc = {k: np.searchsorted(union, v) for k, v in reg.items()}   # union-local idx
    idxA, idxB = np.asarray(tmpl.idxA), np.asarray(tmpl.idxB)

    # gather Cα coords (union only) + per-frame Vα–Vβ contact for validity
    coords, valid, frames = [], [], []
    for ts in tqdm(u.trajectory[::stride], total=int(np.ceil(n_frames / stride)),
                   desc=pdbid, leave=False):
        P = u.atoms.positions.astype(float)
        coords.append(P[union])
        valid.append(float(cdist(P[idxA], P[idxB]).min()) < contact_cutoff)
        frames.append(int(ts.frame))
    coords = np.asarray(coords)
    valid = np.asarray(valid, bool)
    C = coords[valid]
    if len(C) < 10:
        raise RuntimeError(f"only {len(C)} valid frames")
    exc = {"frame": np.asarray(frames)[valid]}       # per-frame excursions (filled below)

    # framework superposition once per chain (medoid reference)
    fwT = {"A": _fw_transforms(C, loc["fwA"]), "B": _fw_transforms(C, loc["fwB"])}

    rows = []
    # --- CDR loops: framework = whole chain framework; anchors = the CDR stem ---
    for cdr in CDRS:
        loop = loc[cdr]
        if len(loop) < 3:
            continue
        chain = cdr[0]
        m, Dt, Da, Dd = _region_metrics(C, fwT[chain], loop, loc.get(f"{cdr}_anchor", np.array([], int)))
        # keep the per-frame excursion traces for the density maps / clustering
        exc[f"{cdr}_angle"], exc[f"{cdr}_deform"], exc[f"{cdr}_total"] = Da, Dd, Dt
        rows.append({"system": pdbid, "cdr": cdr, "region_type": "CDR", **m,
                     "n_loop": int(len(loop)), "n_valid": int(valid.sum()), "pairing": how})

    # --- framework regions: measured in the SAME whole-framework frame as the CDRs
    #     (a standard framework-aligned RMSF profile; FRs are part of the scaffold
    #     reference, so their RMSF is naturally low → the rigid-scaffold baseline) ---
    for chain in ("A", "B"):
        for fr in (f"{chain}_{r}" for r in FR_RANGES):
            reg = loc.get(fr, np.array([], int))
            if len(reg) < 3:
                continue
            m, *_ = _region_metrics(C, fwT[chain], reg, np.array([], int))
            rows.append({"system": pdbid, "cdr": fr, "region_type": "FR", **m,
                         "n_loop": int(len(reg)), "n_valid": int(valid.sum()), "pairing": how})
    return rows, pd.DataFrame(exc)


# --------------------------------------------------------------------------- #
# Geometry linkage (measured CDR3 bend-angle exploration)
# --------------------------------------------------------------------------- #
def _bend_std(angles_dir: Path):
    """Per-system std of the CDR3 bend angle (α & β), from the angle results."""
    out = {}
    for f in sorted((angles_dir / "per_system").glob("*_angles.csv")):
        df = pd.read_csv(f)
        if "alpha_cdr3_bend_deg" not in df.columns:
            continue
        v = df[df.get("valid", True)] if "valid" in df.columns else df
        out[str(df["system"].iloc[0])] = {
            "A_CDR3": float(v["alpha_cdr3_bend_deg"].std(ddof=0)),
            "B_CDR3": float(v["beta_cdr3_bend_deg"].std(ddof=0)),
        }
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
C_TOTAL, C_ANGLE, C_DEFORM = "#2c5282", "#dd6b20", "#a0aec0"


def _bar_plots(df, out_dir):
    """Two clear RMSF (Å) figures — no Å² anywhere.

    ``flex_components_bars.png`` : per-CDR grouped bars, total / angle / deform.
    ``flex_by_tcr_cdr3.png``     : per-TCR CDR3 with α and β adjacent.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # ---- (1) per-CDR grouped decomposition, in Å ----
    order = CDRS
    m = (df.groupby("cdr")
           .agg(total=("rmsf_total_A", "mean"), angle=("rmsf_angle_A", "mean"),
                deform=("rmsf_deform_A", "mean"))
           .reindex(order))
    x = np.arange(len(order)); w = 0.27
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    b1 = ax.bar(x - w, m["total"],  w, color=C_TOTAL,  label="total loop flexibility")
    b2 = ax.bar(x,      m["angle"],  w, color=C_ANGLE,  label="angle  (loop reorientation)")
    b3 = ax.bar(x + w, m["deform"], w, color=C_DEFORM, label="deformation  (internal shape)")
    for b in (b1, b2, b3):
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Cα RMSF (Å)")
    ax.set_title("CDR loop flexibility: reorientation vs internal deformation\n"
                 "total = framework-aligned RMSF · deform = loop-aligned RMSF · "
                 "angle = re-posed average loop (direct)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_dir / "flex_components_bars.png", dpi=140)
    plt.close(fig)

    # ---- (2) per-TCR CDR3, α and β adjacent, in Å ----
    c3 = df[df["cdr"].isin(["A_CDR3", "B_CDR3"])].copy()
    order_sys = (c3.groupby("system")["rmsf_total_A"].mean()
                   .sort_values(ascending=True).index.tolist())
    rows, y = [], 0.0
    for s in order_sys:
        for chain in ("A", "B"):                       # α then β → adjacent
            r = c3[(c3["system"] == s) & (c3["cdr"] == f"{chain}_CDR3")]
            if len(r):
                r = r.iloc[0]
                rows.append((y, f"{s}  {chain}", r["rmsf_angle_A"], r["rmsf_deform_A"], r["rmsf_total_A"]))
                y += 1.0
        y += 0.7                                        # gap between TCRs

    fig, ax = plt.subplots(figsize=(8.5, max(6, 0.30 * len(rows) + 1.4)))
    hh = 0.40
    for (yy, lab, ang, dfm, tot) in rows:
        ax.barh(yy + hh / 2, ang, height=hh, color=C_ANGLE)
        ax.barh(yy - hh / 2, dfm, height=hh, color=C_DEFORM)
        ax.plot([tot, tot], [yy - hh, yy + hh], color="k", lw=1.3)   # total marker
    ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels([r[1] for r in rows], fontsize=6)
    ax.set_xlabel("Cα RMSF (Å)")
    ax.set_title("CDR3 loop flexibility per TCR — α and β of each TCR adjacent\n"
                 "orange = angle (reorientation),  grey = deformation,  black tick = total")
    ax.legend(handles=[Patch(color=C_ANGLE, label="angle (reorientation)"),
                       Patch(color=C_DEFORM, label="deformation (internal)"),
                       Line2D([0], [0], color="k", lw=1.3, label="total RMSF")],
              loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(out_dir / "flex_by_tcr_cdr3.png", dpi=140)
    plt.close(fig)


def _angle_deform_scatter(df, out_path, xcol="rmsf_deform_A", ycol="rmsf_angle_A",
                          quantity="RMSF (typical)"):
    """One point per CDR3 loop in (deformation, angle) space.

    Position vs the dashed diagonal = the angle:deformation ratio; distance from
    the origin = √(angle² + deform²). ``quantity`` selects the typical RMSF or the
    maximum-excursion columns.
    """
    c3 = df[df["cdr"].isin(["A_CDR3", "B_CDR3"])].copy()
    maxr = np.ceil(np.sqrt((c3[[ycol, xcol]] ** 2).sum(1).max()) * 2 + 0.5) / 2
    th = np.linspace(0, np.pi / 2, 100)

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    for R in np.arange(0.5, maxr + 1e-6, 0.5):                    # iso-distance arcs
        ax.plot(R * np.cos(th), R * np.sin(th), color="0.86", lw=0.8, zorder=0)
        ax.text(R * np.cos(np.deg2rad(63)), R * np.sin(np.deg2rad(63)), f"{R:.1f}",
                color="0.6", fontsize=6, ha="center", va="center", zorder=0)
    ax.plot([0, maxr], [0, maxr], "k--", lw=1, alpha=0.5, zorder=1)
    ax.text(maxr * 0.66, maxr * 0.71, "angle = deform", fontsize=8, color="0.45",
            rotation=45, ha="center")
    ax.text(0.06 * maxr, 0.90 * maxr, "reorientation-\ndominated", fontsize=9, color=C_ANGLE)
    ax.text(0.60 * maxr, 0.07 * maxr, "deformation-\ndominated", fontsize=9, color="#5a6570")

    for chain, col in (("A", "#2b6cb0"), ("B", "#c53030")):
        s = c3[c3["cdr"] == f"{chain}_CDR3"]
        ax.scatter(s[xcol], s[ycol], c=col, s=48, alpha=0.85,
                   edgecolors="k", linewidths=0.4, label=f"{chain} CDR3", zorder=3)
        for _, r in s.iterrows():
            ax.annotate(r["system"], (r[xcol], r[ycol]), fontsize=5,
                        alpha=0.7, xytext=(3, 2), textcoords="offset points", zorder=4)

    ax.set_xlim(0, maxr); ax.set_ylim(0, maxr); ax.set_aspect("equal")
    ax.set_xlabel(f"deformation {quantity} (Å)  —  internal shape change")
    ax.set_ylabel(f"angle {quantity} (Å)  —  loop reorientation")
    ax.set_title(f"CDR3 reorientation vs deformation — {quantity}\n"
                 "distance from origin = √(angle² + deform²) (arcs, Å) · "
                 "above diagonal = angle-dominated")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def _angle_deform_spread(df, out_path):
    """CDR3 angle vs deformation showing the SPREAD, not just a point.

    Each loop is a crosshair: dot = median per-frame excursion, bars = the 10th–90th
    percentile range in each direction. So you see both where the loop typically
    sits and how widely its reorientation / deformation vary over the trajectory.
    """
    c3 = df[df["cdr"].isin(["A_CDR3", "B_CDR3"])].copy()
    maxr = np.ceil(np.sqrt((c3[["exc_angle_p90", "exc_deform_p90"]] ** 2).sum(1).max()) * 2 + 0.5) / 2
    th = np.linspace(0, np.pi / 2, 100)

    fig, ax = plt.subplots(figsize=(8.4, 8.4))
    for R in np.arange(0.5, maxr + 1e-6, 0.5):
        ax.plot(R * np.cos(th), R * np.sin(th), color="0.88", lw=0.8, zorder=0)
        ax.text(R * np.cos(np.deg2rad(63)), R * np.sin(np.deg2rad(63)), f"{R:.1f}",
                color="0.6", fontsize=6, ha="center", va="center", zorder=0)
    ax.plot([0, maxr], [0, maxr], "k--", lw=1, alpha=0.5, zorder=1)
    ax.text(maxr * 0.66, maxr * 0.71, "angle = deform", fontsize=8, color="0.45",
            rotation=45, ha="center")
    ax.text(0.05 * maxr, 0.90 * maxr, "reorientation-\ndominated", fontsize=9, color=C_ANGLE)
    ax.text(0.60 * maxr, 0.06 * maxr, "deformation-\ndominated", fontsize=9, color="#5a6570")

    for chain, col in (("A", "#2b6cb0"), ("B", "#c53030")):
        s = c3[c3["cdr"] == f"{chain}_CDR3"]
        xerr = np.vstack([s["exc_deform_p50"] - s["exc_deform_p10"],
                          s["exc_deform_p90"] - s["exc_deform_p50"]])
        yerr = np.vstack([s["exc_angle_p50"] - s["exc_angle_p10"],
                          s["exc_angle_p90"] - s["exc_angle_p50"]])
        ax.errorbar(s["exc_deform_p50"], s["exc_angle_p50"], xerr=xerr, yerr=yerr,
                    fmt="o", ms=4, color=col, ecolor=col, elinewidth=0.9, capsize=1.5,
                    alpha=0.55, label=f"{chain} CDR3", zorder=3)

    ax.set_xlim(0, maxr); ax.set_ylim(0, maxr); ax.set_aspect("equal")
    ax.set_xlabel("deformation excursion (Å)  —  internal shape change")
    ax.set_ylabel("angle excursion (Å)  —  loop reorientation")
    ax.set_title("CDR3 reorientation vs deformation — spread over the trajectory\n"
                 "dot = median per-frame excursion · bars = 10th–90th percentile")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def _density_montage(out_dir, cdr="CDR3"):
    """Per-TCR 2D density of (deformation, angle) per-frame excursion for a CDR,
    decomposed into α (blue) and β (red) as overlaid KDE contours.

    One panel per TCR, so you can compare each chain's deformation-vs-reorientation
    cloud. Reads the per-frame traces in ``per_system/<ID>_excursions.csv``.
    """
    from scipy.stats import gaussian_kde
    per = sorted((out_dir / "per_system").glob("*_excursions.csv"))
    if not per:
        return
    data = {p.name[:-len("_excursions.csv")]: pd.read_csv(p) for p in per}
    systems = sorted(data)
    chains = (("A", "#2b6cb0"), ("B", "#c53030"))

    allv = np.concatenate([np.concatenate([d[f"{c}_{cdr}_deform"].to_numpy(),
                                           d[f"{c}_{cdr}_angle"].to_numpy()])
                           for d in data.values() for c, _ in chains])
    hi = np.percentile(allv, 99.5) * 1.05
    gx, gy = np.mgrid[0:hi:80j, 0:hi:80j]
    grid = np.vstack([gx.ravel(), gy.ravel()])

    n = len(systems); ncol = 5; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, systems):
        d = data[s]
        for c, col in chains:
            x = d[f"{c}_{cdr}_deform"].to_numpy(); y = d[f"{c}_{cdr}_angle"].to_numpy()
            if len(x) < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
                continue
            z = gaussian_kde(np.vstack([x, y]))(grid).reshape(gx.shape)
            lv = z.max() * np.array([0.1, 0.35, 0.7])       # contour levels
            ax.contour(gx, gy, z, levels=lv, colors=col, linewidths=0.9, alpha=0.9)
        ax.plot([0, hi], [0, hi], "0.6", ls="--", lw=0.7)
        ax.set_title(s, fontsize=9)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
    for ax in axes[n:]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([0], [0], color="#2b6cb0", label="α CDR3"),
                            Line2D([0], [0], color="#c53030", label="β CDR3")],
                   fontsize=7, loc="upper right")
    fig.supxlabel("deformation excursion (Å)  →  internal shape change")
    fig.supylabel("angle excursion (Å)  →  loop reorientation")
    fig.suptitle(f"{cdr} deformation vs angle — per-TCR density, α (blue) vs β (red) "
                 "· dashed = angle=deform", y=0.998)
    fig.tight_layout()
    fig.savefig(out_dir / f"density_{cdr}_per_tcr.png", dpi=130)
    plt.close(fig)


def _exploration_plot(df, out_dir):
    """Typical (RMSF, average) vs extreme (max excursion) exploration of each loop.

    ``flex_exploration.png``        : per-CDR, RMSF bar → max, with per-TCR maxima.
    ``flex_exploration_by_tcr.png`` : per-TCR CDR3 dumbbell, RMSF → max, α/β adjacent.
    """
    from matplotlib.lines import Line2D

    # (1) per-CDR: typical vs maximum
    m = (df.groupby("cdr").agg(rmsf=("rmsf_total_A", "mean"),
                               mx=("max_total_A", "mean")).reindex(CDRS))
    x = np.arange(len(CDRS))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.bar(x, m["rmsf"], width=0.5, color="#2c5282", zorder=2,
           label="typical (RMSF · average over frames)")
    ax.vlines(x, m["rmsf"], m["mx"], color="#dd6b20", lw=2, zorder=3)
    ax.scatter(x, m["mx"], color="#dd6b20", s=45, zorder=4,
               label="max excursion (mean across TCRs)")
    for xi, cdr in zip(x, CDRS):
        vals = df.loc[df["cdr"] == cdr, "max_total_A"].to_numpy()
        ax.scatter(np.full_like(vals, xi) + 0.30, vals, s=9, color="#dd6b20",
                   alpha=0.4, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(CDRS)
    ax.set_ylabel("Cα deviation from the average conformation (Å)")
    ax.set_title("Loop exploration: typical fluctuation vs maximum excursion\n"
                 "bar = mean RMSF · stem → mean max · dots = each TCR's most extreme frame")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_dir / "flex_exploration.png", dpi=140)
    plt.close(fig)

    # (2) per-TCR CDR3 dumbbell: RMSF → max, α and β adjacent
    c3 = df[df["cdr"].isin(["A_CDR3", "B_CDR3"])].copy()
    order_sys = (c3.groupby("system")["max_total_A"].mean()
                   .sort_values(ascending=True).index.tolist())
    rows, y = [], 0.0
    for s in order_sys:
        for ch in ("A", "B"):
            r = c3[(c3["system"] == s) & (c3["cdr"] == f"{ch}_CDR3")]
            if len(r):
                r = r.iloc[0]
                rows.append((y, f"{s}  {ch}", r["rmsf_total_A"], r["max_total_A"]))
                y += 1.0
        y += 0.7
    fig, ax = plt.subplots(figsize=(8.5, max(6, 0.30 * len(rows) + 1.4)))
    for (yy, lab, rm, mx) in rows:
        ax.plot([rm, mx], [yy, yy], color="#cbd5e0", lw=2.2, zorder=1)
        ax.scatter([rm], [yy], color="#2c5282", s=28, zorder=2)
        ax.scatter([mx], [yy], color="#dd6b20", s=28, zorder=2)
    ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels([r[1] for r in rows], fontsize=6)
    ax.set_xlabel("CDR3 Cα deviation from the average conformation (Å)")
    ax.set_title("CDR3 exploration per TCR — α and β adjacent\n"
                 "typical RMSF (blue) → maximum excursion (orange)")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c5282",
                              markersize=8, label="typical (RMSF)"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor="#dd6b20",
                              markersize=8, label="max excursion")],
              loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(out_dir / "flex_exploration_by_tcr.png", dpi=140)
    plt.close(fig)


def _heatmap(df, out_path):
    regions = [f"{c}_{r}" for c in "AB" for r in REGION_SEQ]      # FR+CDR, N→C order
    cols = [r for r in regions if r in df["cdr"].unique()]
    piv = df.pivot_table(index="system", columns="cdr", values="rmsf_total_A").reindex(columns=cols)
    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 2, max(4, 0.3 * len(piv) + 1)))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=7)
    for i in range(len(piv)):
        for j in range(len(cols)):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="w", fontsize=5)
    # divider between α and β blocks
    ax.axvline(sum(c.startswith("A_") for c in cols) - 0.5, color="w", lw=1.5)
    fig.colorbar(im, ax=ax, label="total RMSF (Å)")
    ax.set_title("Per-region flexibility (Å) — framework + CDR, α | β")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _flex_profile(df, out_path):
    """Flexibility profile along the variable domain: framework (grey) vs CDR (red).

    Total Cα RMSF per region in N→C order (FR1·CDR1·FR2·CDR2·FR3·CDR3·FR4), one
    panel per chain, mean ± std across TCRs with each TCR overlaid — the classic
    'rigid scaffold, mobile loops' picture, now measured directly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(REGION_SEQ))
    for ax, chain, name in zip(axes, ("A", "B"), ("α", "β")):
        regs = [f"{chain}_{r}" for r in REGION_SEQ]
        means = [df.loc[df["cdr"] == r, "rmsf_total_A"].mean() for r in regs]
        stds = [df.loc[df["cdr"] == r, "rmsf_total_A"].std() for r in regs]
        colors = ["#a0aec0" if r.startswith("FR") else "#c53030" for r in REGION_SEQ]
        ax.bar(x, means, yerr=stds, color=colors, capsize=3, alpha=0.85, zorder=2)
        for xi, r in zip(x, regs):
            vals = df.loc[df["cdr"] == r, "rmsf_total_A"].to_numpy()
            ax.scatter(np.full_like(vals, xi), vals, s=8, color="k", alpha=0.25, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(REGION_SEQ, rotation=45, ha="right")
        ax.set_title(f"{name} chain"); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("total Cα RMSF (Å)")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color="#a0aec0", label="framework (FR)"),
                            Patch(color="#c53030", label="CDR loop")], fontsize=9)
    fig.suptitle("Flexibility profile along the variable domain — rigid framework vs mobile CDRs", y=1.0)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def _flex_vs_geometry(df, bend, out_png, out_csv):
    rows = []
    for _, r in df[df["cdr"].isin(["A_CDR3", "B_CDR3"])].iterrows():
        b = bend.get(r["system"], {}).get(r["cdr"])
        if b is not None:
            rows.append({"system": r["system"], "cdr": r["cdr"], "chain": r["cdr"][0],
                         "bend_std_deg": b, "rmsf_angle_A": r["rmsf_angle_A"],
                         "rmsf_total_A": r["rmsf_total_A"], "angle_frac": r["angle_frac"]})
    if not rows:
        return None
    g = pd.DataFrame(rows)
    g.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    for chain, col in (("A", "#2b6cb0"), ("B", "#c53030")):
        s = g[g["chain"] == chain]
        ax.scatter(s["bend_std_deg"], s["rmsf_angle_A"], c=col, s=40,
                   label=f"{chain} CDR3", alpha=0.8, edgecolors="k", linewidths=0.4)
        for _, r in s.iterrows():
            ax.annotate(r["system"], (r["bend_std_deg"], r["rmsf_angle_A"]),
                        fontsize=5, alpha=0.6)
    if len(g) >= 3:
        r = np.corrcoef(g["bend_std_deg"], g["rmsf_angle_A"])[0, 1]
        a, b0 = np.polyfit(g["bend_std_deg"], g["rmsf_angle_A"], 1)
        xs = np.linspace(g["bend_std_deg"].min(), g["bend_std_deg"].max(), 50)
        ax.plot(xs, a * xs + b0, "k--", lw=1, alpha=0.6)
        ax.set_title(f"CDR3 flexibility ANGLE component vs measured bend-angle exploration\n"
                     f"Pearson r = {r:.2f}  (n={len(g)})")
    ax.set_xlabel("CDR3 bend-angle std over MD (°)   [from angle_exploration]")
    ax.set_ylabel("CDR3 angle (reorientation) RMSF (Å)   [this analysis]")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return g


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*", default=None, help="subset of PDB ids")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    ap.add_argument("--angles-dir", default=str(DEFAULT_ANGLES),
                    help="dir with per_system/<ID>_angles.csv (CDR3 bend std)")
    ap.add_argument("--target-frames", type=int, default=3000, help="approx frames (sets stride)")
    ap.add_argument("--stride", type=int, default=None, help="explicit frame stride")
    ap.add_argument("--contact-cutoff", type=float, default=CONTACT_CUTOFF,
                    help="drop frames whose Vα–Vβ closest Cα exceeds this (Å)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of systems")
    ap.add_argument("--plots-only", action="store_true",
                    help="regenerate figures from flex_components.csv (no MD recompute)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        df = pd.read_csv(out_dir / "flex_components.csv")
        _bar_plots(df, out_dir)
        _angle_deform_scatter(df, out_dir / "flex_angle_vs_deform.png")
        _angle_deform_scatter(df, out_dir / "flex_angle_vs_deform_max.png",
                              xcol="max_deform_A", ycol="max_angle_A", quantity="max excursion")
        _angle_deform_spread(df, out_dir / "flex_angle_vs_deform_spread.png")
        _exploration_plot(df, out_dir)
        _density_montage(out_dir)
        _flex_profile(df, out_dir / "flex_profile.png")
        _heatmap(df, out_dir / "cdr_flexibility_heatmap.png")
        bend = _bend_std(Path(args.angles_dir))
        _flex_vs_geometry(df, bend, out_dir / "flex_vs_geometry.png",
                          out_dir / "flex_vs_geometry.csv")
        print(f"[plots-only] regenerated figures in {out_dir}")
        return

    systems = args.systems or discover_systems()
    if args.limit:
        systems = systems[: args.limit]
    print(f"[info] {len(systems)} system(s): {', '.join(systems)}")

    con = Consensus()
    per_dir = out_dir / "per_system"
    per_dir.mkdir(exist_ok=True)
    all_rows, done, errors = [], [], []
    err_log = out_dir / "errors.txt"
    for s in systems:
        print(f"\n=== {s} ===")
        try:
            rows, exc_df = process_system(s, con, args.target_frames, args.stride, args.contact_cutoff)
            all_rows.extend(rows)
            exc_df.to_csv(per_dir / f"{s}_excursions.csv", index=False)
            done.append(s)
            c3 = next((r for r in rows if r["cdr"] == "A_CDR3"), None)
            if c3:
                print(f"    {c3['n_valid']} valid frames; "
                      f"A_CDR3 total={c3['rmsf_total_A']:.2f}Å "
                      f"(angle {c3['rmsf_angle_A']:.2f} / deform {c3['rmsf_deform_A']:.2f})")
        except Exception as e:
            errors.append(s)
            with open(err_log, "a") as fh:
                fh.write(f"{s}: {e}\n")
            print(f"    [ERROR] {e} (logged)")

    if not all_rows:
        print("[fatal] nothing processed."); return
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "flex_components.csv", index=False)

    _bar_plots(df, out_dir)
    _angle_deform_scatter(df, out_dir / "flex_angle_vs_deform.png")
    _angle_deform_scatter(df, out_dir / "flex_angle_vs_deform_max.png",
                          xcol="max_deform_A", ycol="max_angle_A", quantity="max excursion")
    _angle_deform_spread(df, out_dir / "flex_angle_vs_deform_spread.png")
    _exploration_plot(df, out_dir)
    _density_montage(out_dir)
    _flex_profile(df, out_dir / "flex_profile.png")
    _heatmap(df, out_dir / "cdr_flexibility_heatmap.png")

    bend = _bend_std(Path(args.angles_dir))
    g = _flex_vs_geometry(df, bend, out_dir / "flex_vs_geometry.png",
                          out_dir / "flex_vs_geometry.csv")

    # clean the scratch dir
    tmp = out_dir / "_tmp"
    if tmp.exists():
        for f in tmp.glob("*"):
            f.unlink(missing_ok=True)
        tmp.rmdir()

    print(f"\n[done] {len(done)} ok, {len(errors)} failed. Outputs in {out_dir}")
    if g is not None and len(g) >= 3:
        r = np.corrcoef(g["bend_std_deg"], g["rmsf_angle_A"])[0, 1]
        print(f"       CDR3 angle-component vs bend-std Pearson r = {r:.2f} (n={len(g)})")


if __name__ == "__main__":
    main()
