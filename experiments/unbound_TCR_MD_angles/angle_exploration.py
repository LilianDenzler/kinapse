#!/usr/bin/env python
"""Explore the Vα/Vβ inter-domain docking geometry along unbound TCR MD.

For every TCR MD system in ``CORY_ORIOL_MERGED_MD`` this computes the packaged
kinapse 6-parameter α/β docking geometry per frame:

    BA        signed twist (torsion) between the Vα and Vβ principal axes  [deg]
    AC1, AC2  tilt of the Vα principal axes vs. the inter-centroid vector   [deg]
    BC1, BC2  tilt of the Vβ principal axes vs. the inter-centroid vector   [deg]
    dc        distance between the Vα and Vβ centroids                      [Å]

These six numbers are the TCR analogue of the antibody VH–VL "ABangle" and
describe the shape/orientation of the combined CDR binding platform. Watching
them fluctuate over an *unbound* trajectory shows the conformational-fit space
each receptor samples before it ever meets pMHC.

Method
------
Geometry is *identical* to ``kinapse.geometry.calc_geometry.process`` (biotite
``superimpose_structural_homologs`` onto the packaged consensus Vα/Vβ frames,
then the same angle algebra). The only change is speed: the topology is
ANARCI/IMGT-renumbered **once** via ``kinapse.structures.TCR`` and the two
per-frame superpositions run on in-memory ``AtomArray``s (no per-frame PDB
round-trip). This reproduces ``process()`` to < 1e-2° and runs ~30 ms/frame,
so full multi-µs trajectories are tractable.

Robustness
----------
* These merged trajectories carry **no periodic box**, and a subset of frames
  (typically a contiguous block appended from a second source) have the two
  domains split/displaced. Such frames are detected by the Vα–Vβ interface
  contact distance and **excluded** from statistics and plots (kept in the CSV
  with ``valid=False`` for transparency).
* ANARCI occasionally mistypes an α chain as δ (TRAV/TRDV ambiguity), which
  breaks the automatic α/β pairing. When the default pairing fails we retry with
  coerced chain types, disambiguating orientation by superposition fit quality.

Run (kinapse env; user-site is excluded automatically to avoid a broken h5py)::

    python angle_exploration.py                       # all systems, ~4000 frames each
    python angle_exploration.py --systems 1KGC 8YJ3   # a subset
    python angle_exploration.py --full                # every frame (slow)

Requires ``gemmi`` in the env (a hard dependency of anarcii used for numbering).
"""
from __future__ import annotations

# --- exclude the user site-packages: it shadows the env with an ABI-broken
#     h5py that crashes MDAnalysis. Re-exec once with it disabled. ------------
import os
import sys

if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import glob
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MDAnalysis as mda
import biotite.structure as bts
import biotite.structure.io as btsio
from tqdm import tqdm

from kinapse.geometry import DATA_PATH
from kinapse.geometry.calc_geometry import as_unit, angle_between, read_pseudo_points
from kinapse.structures.tcr import TCR
from kinapse.structures.numbering.tcr_pairing import pair_tcrs_by_interface

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = Path("/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD")
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results"

# A frame is discarded if the two variable domains are not in contact (their
# closest framework Cα approach exceeds this, in Å). Real paired domains sit at
# ~5 Å; broken/split frames are 40+ Å, so the cutoff is not sensitive.
CONTACT_CUTOFF = 15.0

PARAMS = ["BA", "AC1", "AC2", "BC1", "BC2", "dc"]
PARAM_LABEL = {
    "BA": "BA twist (°)",
    "AC1": "AC1 tilt (°)",
    "AC2": "AC2 tilt (°)",
    "BC1": "BC1 tilt (°)",
    "BC2": "BC2 tilt (°)",
    "dc": "dc centroid dist (Å)",
}

# CDR3 loop shape (added): signed bend (deviation-from-straight, 0 = the loop
# points straight away from its domain centroid) + apex protrusion above the
# 104->118 base chord. Same definition as kinapse.geometry.calc_geometry.
CDR3_NUM = [
    "alpha_cdr3_bend_deg", "alpha_cdr3_apex_height_A",
    "beta_cdr3_bend_deg", "beta_cdr3_apex_height_A",
]
CDR3_STR = ["alpha_cdr3_apex_resi", "beta_cdr3_apex_resi"]
PARAM_LABEL.update({
    "alpha_cdr3_bend_deg": "α CDR3 bend (°)",
    "alpha_cdr3_apex_height_A": "α CDR3 apex height (Å)",
    "beta_cdr3_bend_deg": "β CDR3 bend (°)",
    "beta_cdr3_apex_height_A": "β CDR3 apex height (Å)",
})


# --------------------------------------------------------------------------- #
# Consensus reference (loaded once)
# --------------------------------------------------------------------------- #
class Consensus:
    """Packaged consensus Vα/Vβ frames + pseudo-axes used by the geometry."""

    def __init__(self):
        cA = os.path.join(DATA_PATH, "chain_A/average_structure_with_pca.pdb")
        cB = os.path.join(DATA_PATH, "chain_B/average_structure_with_pca.pdb")
        self.A_res = _read_res(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt"))
        self.B_res = _read_res(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt"))

        sA = btsio.load_structure(cA, model=1)
        sB = btsio.load_structure(cB, model=1)
        self.static_A = _ca_subset(sA, "A", self.A_res)   # fixed Vα CA anchors
        self.static_B = _ca_subset(sB, "B", self.B_res)   # fixed Vβ CA anchors

        # Pseudo-axes: Vα read in the (fixed) consensus-A frame; Vβ carried along
        # with the second superposition, so keep its local coords.
        self.Apts = read_pseudo_points(cA, "A")
        Bloc = read_pseudo_points(cB, "B")
        self.Bloc = np.array([Bloc.C, Bloc.V1, Bloc.V2], float)


def _read_res(path):
    return [int(x) for x in open(path).read().split(",") if x.strip()]


def _ca_subset(struct, chain, res_list):
    m = (struct.atom_name == "CA") & (struct.chain_id == chain)
    sub = struct[m]
    return sub[np.isin(sub.res_id, res_list)]


def _apply(coords, transform):
    """Apply a biotite affine transform to an (N,3) coordinate array."""
    M = np.asarray(transform.as_matrix(), float)
    if M.shape == (1, 4, 4):
        M = M[0]
    return coords @ M[:3, :3].T + M[:3, 3]


# --------------------------------------------------------------------------- #
# Per-system template (built once from the renumbered topology)
# --------------------------------------------------------------------------- #
class SystemTemplate:
    """Maps MD atoms -> the two consensus-residue CA sets, in IMGT numbering.

    ANARCI numbering depends only on sequence, which is constant across a
    trajectory, so this is built once from a reference frame and reused for
    every frame. Vα is always modelled as chain "A", Vβ as chain "B" (so both
    αβ and γδ receptors map onto the packaged αβ consensus).
    """

    def __init__(self, universe, pair, con: Consensus):
        a_md, b_md = pair.alpha_chain_id, pair.beta_chain_id           # original MD chain ids
        a_lab, b_lab = pair.chain_map["alpha"], pair.chain_map["beta"]  # IMGT labels
        map_a = _imgt_map(universe, pair, a_md, a_lab)
        map_b = _imgt_map(universe, pair, b_md, b_lab)

        self.idxA, ridA, resnA = _pick(universe, a_md, map_a, con.A_res)
        self.idxB, ridB, resnB = _pick(universe, b_md, map_b, con.B_res)
        self.cdr3A_idx, self.cdr3A_rid = _pick_cdr3(universe, a_md, map_a)
        self.cdr3B_idx, self.cdr3B_rid = _pick_cdr3(universe, b_md, map_b)
        self.arrA = _template_arr(ridA, resnA, "A")
        self.arrB = _template_arr(ridB, resnB, "B")
        self.movedB = self.arrB.copy()
        self.alpha_type, self.beta_type = pair.alpha_type, pair.beta_type
        # md chain ids + IMGT maps, so callers can pick other regions (CDR1/CDR2)
        self.a_md, self.b_md = a_md, b_md
        self.map_a, self.map_b = map_a, map_b
        self._con = con

    def _load(self, positions):
        self.arrA.coord = positions[self.idxA].astype(np.float32)
        self.arrB.coord = positions[self.idxB].astype(np.float32)

    def geometry(self, positions) -> dict:
        con = self._con
        self._load(positions)
        interdom = float(cdist(self.arrA.coord, self.arrB.coord).min())
        # 1) fit input Vα onto consensus Vα; carry input Vβ along
        _, T1, _, _ = bts.superimpose_structural_homologs(
            fixed=con.static_A, mobile=self.arrA, max_iterations=1)
        inB_aligned = _apply(self.arrB.coord.astype(float), T1)
        # 2) fit consensus Vβ (with its pseudo-axes) onto the aligned input Vβ
        self.movedB.coord = inB_aligned.astype(np.float32)
        _, T2, _, _ = bts.superimpose_structural_homologs(
            fixed=self.movedB, mobile=con.static_B, max_iterations=1)
        Bp = _apply(con.Bloc, T2)
        g = _angles(con.Apts, Bp[0], Bp[1], Bp[2])
        g["interdom_min"] = interdom

        # CDR3 bend/apex in the consensus-A frame: T1 rigidly maps the whole input
        # there, so the loop CAs share the frame of the domain centroids used by
        # the 6-parameter geometry (con.Apts.C for Vα, Bp[0] for Vβ).
        ca = _cdr3_bend_from_picked(
            _apply(positions[self.cdr3A_idx].astype(float), T1), self.cdr3A_rid, con.Apts.C)
        cb = _cdr3_bend_from_picked(
            _apply(positions[self.cdr3B_idx].astype(float), T1), self.cdr3B_rid, Bp[0])
        g["alpha_cdr3_bend_deg"] = ca["bend_deg"]
        g["alpha_cdr3_apex_height_A"] = ca["apex_height_A"]
        g["alpha_cdr3_apex_resi"] = ca["apex_resi"]
        g["beta_cdr3_bend_deg"] = cb["bend_deg"]
        g["beta_cdr3_apex_height_A"] = cb["apex_height_A"]
        g["beta_cdr3_apex_resi"] = cb["apex_resi"]
        return g

    def fit_score(self, positions) -> float:
        """Summed Cα-fit RMSD of each chain onto its assigned consensus.

        Small only when Vα really matches consensus-A and Vβ matches
        consensus-B, so it disambiguates the α/β assignment when typing fails.
        """
        con = self._con
        self._load(positions)
        return _fit_rmsd(con.static_A, self.arrA) + _fit_rmsd(con.static_B, self.arrB)


def _fit_rmsd(static, mobile) -> float:
    fitted, _, fi, mi = bts.superimpose_structural_homologs(
        fixed=static, mobile=mobile, max_iterations=1)
    d = fitted.coord[mi] - static.coord[fi]
    return float(np.sqrt((d ** 2).sum(1).mean()))


def _imgt_map(u, pair, md_chain, imgt_label):
    """orig MD resid -> IMGT resid, by 1:1 residue-order zip on a chain."""
    md_resids = list(u.select_atoms(f"chainID {md_chain} and name CA").resids)
    chain = next(c for c in pair.full_structure.get_chains() if c.id == imgt_label)
    imgt_resids = [r.id[1] for r in chain.get_residues()]
    if len(md_resids) != len(imgt_resids):
        raise RuntimeError(
            f"residue count mismatch chain {md_chain}/{imgt_label}: "
            f"{len(md_resids)} vs {len(imgt_resids)}")
    return dict(zip(md_resids, imgt_resids))


def _pick(u, md_chain, res_map, res_list):
    """Global atom indices / IMGT resids / resnames of the consensus CA anchors."""
    ca = u.select_atoms(f"chainID {md_chain} and name CA")
    got = []
    for at in ca:
        imgt = res_map.get(int(at.resid))
        if imgt in res_list:
            got.append((res_list.index(imgt), at.index, imgt, at.resname))
    got.sort()  # consensus-residue order
    idx = np.array([g[1] for g in got])
    rid = np.array([g[2] for g in got], int)
    resn = [g[3] for g in got]
    if len(idx) < 4:
        raise RuntimeError(f"only {len(idx)} anchors on chain {md_chain}")
    return idx, rid, resn


def _template_arr(res_ids, res_names, chain):
    arr = bts.AtomArray(len(res_ids))
    arr.chain_id = np.array([chain] * len(res_ids))
    arr.atom_name = np.array(["CA"] * len(res_ids))
    arr.res_id = res_ids.astype(int)
    arr.res_name = np.array(res_names)
    arr.element = np.array(["C"] * len(res_ids))
    return arr


def _angles(Apts, Bc, Bv1, Bv2):
    """The exact angle algebra of kinapse.geometry.calc_geometry.process."""
    Cvec = as_unit(Bc - Apts.C)
    A1 = as_unit(Apts.V1 - Apts.C)
    A2 = as_unit(Apts.V2 - Apts.C)
    B1 = as_unit(Bv1 - Bc)
    B2 = as_unit(Bv2 - Bc)
    nx = np.cross(A1, Cvec)
    ny = np.cross(Cvec, nx)
    Lp = as_unit([0.0, np.dot(A1, nx), np.dot(A1, ny)])
    Hp = as_unit([0.0, np.dot(B1, nx), np.dot(B1, ny)])
    BA = angle_between(Lp, Hp)
    if np.cross(Lp, Hp)[0] < 0:
        BA = -BA
    return {
        "BA": BA,
        "BC1": angle_between(B1, -Cvec),
        "AC1": angle_between(A1, Cvec),
        "BC2": angle_between(B2, -Cvec),
        "AC2": angle_between(A2, Cvec),
        "dc": float(np.linalg.norm(Bc - Apts.C)),
    }


# --------------------------------------------------------------------------- #
# CDR3 loop bend / apex (in-memory port of calc_geometry's fixed metric)
# --------------------------------------------------------------------------- #
def _pick_cdr3(u, md_chain, res_map, lo=104, hi=118):
    """Global CA atom indices + IMGT integer resids for the CDR3 region (104..118).

    Ordered by IMGT number; insertion codes collapse onto their integer, which is
    harmless here — the apex is chosen geometrically, not by sequence position.
    """
    ca = u.select_atoms(f"chainID {md_chain} and name CA")
    got = []
    for at in ca:
        imgt = res_map.get(int(at.resid))
        if imgt is not None and lo <= imgt <= hi:
            got.append((imgt, at.index))
    got.sort()
    idx = np.array([g[1] for g in got], int)
    rid = np.array([g[0] for g in got], int)
    return idx, rid


def _nan_cdr3():
    return {"bend_deg": np.nan, "apex_height_A": np.nan, "apex_resi": None}


def _cdr3_bend_from_picked(coords, rid, centroid):
    """Signed CDR3 bend + apex height from picked CA coords, in the frame shared
    with ``centroid`` (the metric is rigid-motion invariant).

    Base chord = anchor 104 -> 118; work in the plane perpendicular to it. The
    apex (loop residue furthest from the nearer anchor) is expressed relative to
    the "into the domain" reference (base midpoint -> centroid):

        bend_deg = (180 - angle(base->centroid, base->apex))  # 0 = straight

    signed by a stable in-plane lateral axis. No 0/360 wrap; sign well-defined at
    the near-straight modal pose.
    """
    if rid.size == 0:
        return _nan_cdr3()
    a_sel = np.flatnonzero(rid == 104)
    b_sel = np.flatnonzero(rid == 118)
    loop_sel = np.flatnonzero((rid >= 105) & (rid <= 117))
    if a_sel.size != 1 or b_sel.size != 1 or loop_sel.size < 3:
        return _nan_cdr3()

    A = coords[a_sel[0]]
    B = coords[b_sel[0]]
    loop = coords[loop_sel]
    loop_rid = rid[loop_sel]

    uhat = as_unit(B - A)                       # base chord 104 -> 118
    M = 0.5 * (A + B)
    v = np.asarray(centroid, float) - M
    v_perp = v - np.dot(v, uhat) * uhat
    v_perp_norm = float(np.linalg.norm(v_perp))

    w = loop - M
    w_perp = w - np.outer(w @ uhat, uhat)       # apex vectors, off the chord
    height = np.linalg.norm(w_perp, axis=1)
    dA = np.linalg.norm(loop - A, axis=1)
    dB = np.linalg.norm(loop - B, axis=1)
    score = np.minimum(dA, dB)
    mid = (len(loop) - 1) / 2.0
    centrality = -np.abs(np.arange(len(loop)) - mid)
    apex_i = int(np.lexsort((centrality, height, score))[-1])   # max score, tie->height->centre

    wR = w_perp[apex_i]
    w_perp_norm = float(height[apex_i])
    if v_perp_norm > 1e-6 and w_perp_norm > 1e-6:
        vhat = v_perp / v_perp_norm
        what = wR / w_perp_norm
        bend = 180.0 - angle_between(vhat, what)          # 0 = straight, no wrap
        lat = np.cross(uhat, vhat)                        # stable in-plane lateral axis
        lat_norm = float(np.linalg.norm(lat))
        sign = -1.0 if (lat_norm > 1e-9 and np.dot(what, lat / lat_norm) < 0) else 1.0
        bend_signed = sign * bend
    else:
        bend_signed = np.nan
    return {"bend_deg": bend_signed,
            "apex_height_A": w_perp_norm,
            "apex_resi": int(loop_rid[apex_i])}


# --------------------------------------------------------------------------- #
# Pairing / α-β resolution (handles ANARCI mistyping)
# --------------------------------------------------------------------------- #
def _try_template(u, frame0, con, manual):
    try:
        tcr = TCR(input_pdb=frame0, legacy_anarci=True, manual_chain_types=manual)
        if tcr.pairs:
            return SystemTemplate(u, tcr.pairs[0], con)
    except Exception:
        pass
    return None


def _complement_types(types):
    """From detected chain types, force a valid pair using the confident chain.

    β (or α) is rarely confused, whereas α↔δ / β↔γ are. So a confident β fixes
    its partner as α, and a confident α fixes its partner as β (and likewise for
    the γδ locus). Returns a {chain_id: type} override, or None if undecidable.
    """
    chains = list(types)
    if len(chains) != 2:
        return None
    c1, c2 = chains
    t = types
    for a, b in ((c1, c2), (c2, c1)):
        if t[a] == "B":
            return {a: "B", b: "A"}      # confident β -> partner α
        if t[a] == "A":
            return {a: "A", b: "B"}      # confident α -> partner β
        if t[a] == "G":
            return {a: "G", b: "D"}
        if t[a] == "D":
            return {a: "D", b: "G"}
    return None


def resolve_template(u, frame0, con, ref_positions):
    """Build a SystemTemplate, coping with automatic-pairing failures."""
    t = _try_template(u, frame0, con, None)
    if t is not None:
        return t, "auto"

    # need the detected chain types to coerce
    _, _, _, types = pair_tcrs_by_interface(frame0, legacy_anarci=True)
    coerced = _complement_types(types)
    if coerced is not None:
        t = _try_template(u, frame0, con, coerced)
        if t is not None:
            return t, f"coerced {coerced}"

    # last resort: try both orders, keep the better superposition fit
    chains = list(types)
    if len(chains) == 2:
        scored = []
        for m in ({chains[0]: "A", chains[1]: "B"}, {chains[0]: "B", chains[1]: "A"}):
            t = _try_template(u, frame0, con, m)
            if t is not None:
                scored.append((t.fit_score(ref_positions), m, t))
        if scored:
            score, m, t = min(scored, key=lambda x: x[0])
            return t, f"fit-picked {m} (rmsd {score:.1f})"

    raise RuntimeError(f"could not resolve TCR pair (types={types})")


# --------------------------------------------------------------------------- #
# Per-system run
# --------------------------------------------------------------------------- #
def run_system(pdbid, con, out_dir, target_frames, stride_override, full, contact_cutoff):
    d = DATA_DIR / pdbid
    top, xtc = d / f"{pdbid}.pdb", d / f"{pdbid}.xtc"
    u = mda.Universe(top.as_posix(), xtc.as_posix())
    n_frames = len(u.trajectory)
    dt_ps = float(getattr(u.trajectory, "dt", 0.0) or 0.0)

    if full:
        stride = 1
    elif stride_override:
        stride = stride_override
    else:
        stride = max(1, n_frames // target_frames)

    # renumber once from frame 0
    u.trajectory[0]
    frame0 = out_dir / f"{pdbid}_frame0.pdb"
    u.atoms.write(frame0.as_posix())
    ref_positions = u.atoms.positions.astype(float)
    tmpl, how = resolve_template(u, frame0.as_posix(), con, ref_positions)
    frame0.unlink(missing_ok=True)
    if how != "auto":
        print(f"    [pairing] {how}")

    rows = []
    for ts in tqdm(u.trajectory[::stride],
                   total=int(np.ceil(n_frames / stride)),
                   desc=f"{pdbid}", leave=False):
        g = tmpl.geometry(u.atoms.positions.astype(float))
        g["frame"] = int(ts.frame)
        g["time_ns"] = (ts.frame * dt_ps) / 1000.0 if dt_ps else np.nan
        rows.append(g)

    df = pd.DataFrame(rows)
    df["valid"] = df["interdom_min"] < contact_cutoff
    df.insert(0, "system", pdbid)
    df = df[["system", "frame", "time_ns", *PARAMS, *CDR3_NUM, *CDR3_STR, "interdom_min", "valid"]]
    df.to_csv(out_dir / f"{pdbid}_angles.csv", index=False)

    valid = df[df["valid"]].copy()
    _plot_system(pdbid, df, valid, out_dir)

    meta = {
        "system": pdbid, "n_frames_total": n_frames, "n_frames_used": len(df),
        "n_valid": int(df["valid"].sum()), "n_dropped": int((~df["valid"]).sum()),
        "stride": stride, "dt_ps": dt_ps, "pairing": how,
        "alpha_type": tmpl.alpha_type, "beta_type": tmpl.beta_type,
    }
    return df, valid, meta


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _plot_system(pdbid, df, valid, out_dir):
    # time series with gaps where frames were dropped (invalid -> NaN)
    ts_params = PARAMS + CDR3_NUM
    t = df["time_ns"].to_numpy()
    masked = {p: np.where(df["valid"].to_numpy(), df[p].to_numpy(), np.nan) for p in ts_params}
    fig, axes = plt.subplots(len(ts_params), 1, figsize=(9, 1.55 * len(ts_params)), sharex=True)
    for ax, p in zip(axes, ts_params):
        ax.plot(t, masked[p], lw=0.5, color="#2b6cb0")
        ax.set_ylabel(PARAM_LABEL[p])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time (ns)")
    drop = int((~df["valid"]).sum())
    fig.suptitle(f"{pdbid} — Vα/Vβ docking geometry over unbound MD"
                 + (f"  ({drop} broken frames dropped)" if drop else ""), y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / f"{pdbid}_timeseries.png", dpi=140)
    plt.close(fig)

    if len(valid) < 5:
        return
    tv = valid["time_ns"].to_numpy()
    n = len(PARAMS)
    fig, axes = plt.subplots(n, n, figsize=(2.1 * n, 2.1 * n))
    sc = None
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(valid[PARAMS[i]], bins=40, color="#4a5568", alpha=0.8)
                ax.set_yticks([])
            elif i > j:
                sc = ax.scatter(valid[PARAMS[j]], valid[PARAMS[i]], c=tv, s=3,
                                cmap="viridis", alpha=0.5, linewidths=0)
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(PARAMS[j], fontsize=9)
            if j == 0 and i > 0:
                ax.set_ylabel(PARAMS[i], fontsize=9)
            if i != n - 1:
                ax.set_xticklabels([])
            if j != 0 or i == 0:
                ax.set_yticklabels([])
    if sc is not None:
        cax = fig.add_axes([0.62, 0.72, 0.25, 0.02])
        fig.colorbar(sc, cax=cax, orientation="horizontal", label="time (ns)")
    fig.suptitle(f"{pdbid} — explored conformational-fit space", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / f"{pdbid}_corner.png", dpi=130)
    plt.close(fig)


def _ridgelines(all_df, systems, out_path, params=None):
    import scipy.stats as st
    params = params if params is not None else PARAMS + CDR3_NUM
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 0.42 * len(systems) + 2.2))
    axes = np.atleast_1d(axes)
    ridge_h = 1.6
    for ax, p in zip(axes, params):
        allv = all_df[p].to_numpy()
        allv = allv[np.isfinite(allv)]
        if allv.size == 0:
            ax.set_title(PARAM_LABEL.get(p, p), fontsize=10); continue
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        pad = 0.05 * (hi - lo + 1e-9)
        grid = np.linspace(lo - pad, hi + pad, 256)
        for k, s in enumerate(systems):
            v = all_df.loc[all_df["system"] == s, p].to_numpy()
            v = v[np.isfinite(v)]
            if v.size < 5 or np.ptp(v) == 0:
                continue
            y = st.gaussian_kde(v)(grid)
            y = y / y.max() * ridge_h
            ax.fill_between(grid, k, k + y, color=plt.cm.turbo(k / max(1, len(systems) - 1)),
                            alpha=0.7, lw=0.5, edgecolor="black")
        ax.set_title(PARAM_LABEL.get(p, p), fontsize=10)
        ax.set_yticks(np.arange(len(systems)))
        ax.set_yticklabels(systems if ax is axes[0] else [], fontsize=7)
        ax.set_ylim(-0.5, len(systems) + ridge_h)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Per-parameter distributions across TCR systems "
                 "(docking geometry + CDR3 loop shape)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _spread_heatmap(summary_df, out_path, params=None):
    params = params if params is not None else PARAMS + CDR3_NUM
    cols = [f"{p}_std" for p in params if f"{p}_std" in summary_df.columns]
    used = [p for p in params if f"{p}_std" in summary_df.columns]
    piv = summary_df.set_index("system")[cols]
    piv.columns = used
    z = (piv - piv.mean()) / (piv.std(ddof=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(1.1 * len(used) + 2, 0.35 * len(piv) + 2))
    im = ax.imshow(z.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(used)))
    ax.set_xticklabels(used, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=7)
    for i in range(len(piv)):
        for j in range(len(used)):
            ax.text(j, i, f"{piv.iloc[i, j]:.1f}", ha="center", va="center",
                    color="white", fontsize=6)
    fig.colorbar(im, ax=ax, label="std (z-scored across systems)")
    ax.set_title("Exploration breadth — std per metric (annotated: raw std)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _cdr3_summary(all_df, systems, out_path):
    """Dedicated CDR3 overview: per-system bend/apex-height distributions + the
    α-vs-β bend joint density (pooled)."""
    import scipy.stats as st

    def _violin(ax, col, title):
        data, labels = [], []
        for s in systems:
            v = all_df.loc[all_df["system"] == s, col].to_numpy()
            v = v[np.isfinite(v)]
            if v.size >= 5:
                data.append(v); labels.append(s)
        if not data:
            ax.set_title(title + " (no data)"); return
        parts = ax.violinplot(data, vert=False, showmeans=True, widths=0.9)
        for k, b in enumerate(parts["bodies"]):
            b.set_facecolor(plt.cm.turbo(k / max(1, len(data) - 1))); b.set_alpha(0.7)
        ax.set_yticks(range(1, len(labels) + 1)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(PARAM_LABEL.get(col, col)); ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
        if "bend" in col:
            ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)  # 0 = straight

    fig = plt.figure(figsize=(16, 0.42 * len(systems) + 3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
    _violin(fig.add_subplot(gs[0, 0]), "alpha_cdr3_bend_deg", "α CDR3 bend (° from straight)")
    _violin(fig.add_subplot(gs[0, 1]), "beta_cdr3_bend_deg", "β CDR3 bend (° from straight)")

    axj = fig.add_subplot(gs[0, 2])
    xa = all_df["alpha_cdr3_bend_deg"].to_numpy()
    xb = all_df["beta_cdr3_bend_deg"].to_numpy()
    m = np.isfinite(xa) & np.isfinite(xb)
    if m.sum() > 10:
        axj.hexbin(xa[m], xb[m], gridsize=45, cmap="viridis", mincnt=1)
        r = np.corrcoef(xa[m], xb[m])[0, 1]
        axj.set_title(f"α vs β CDR3 bend  (pooled, r={r:.2f})", fontsize=11)
    axj.axhline(0, color="w", lw=0.6, ls="--", alpha=0.6)
    axj.axvline(0, color="w", lw=0.6, ls="--", alpha=0.6)
    axj.set_xlabel("α CDR3 bend (°)"); axj.set_ylabel("β CDR3 bend (°)")

    fig.suptitle("CDR3 loop shape across unbound TCR MD "
                 "(bend: 0 = points straight off the domain; +/- = lateral lean)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _pca_landscape(all_df, systems, out_path):
    from sklearn.decomposition import PCA
    X = all_df[PARAMS].to_numpy()
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pca = PCA(n_components=2).fit(Xz)
    Y = pca.transform(Xz)
    evr = pca.explained_variance_ratio_ * 100

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6.5))
    h, xe, ye = np.histogram2d(Y[:, 0], Y[:, 1], bins=80)
    with np.errstate(divide="ignore"):
        f = -np.log(h.T / h.max())
    f[np.isinf(f)] = np.nan
    im = ax0.imshow(f, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                    aspect="auto", cmap="viridis_r")
    fig.colorbar(im, ax=ax0, label="-ln(p)  (pooled)")
    ax0.set_title("Pooled conformational-fit landscape")
    ax0.set_xlabel(f"PC1 ({evr[0]:.0f}%)")
    ax0.set_ylabel(f"PC2 ({evr[1]:.0f}%)")

    for k, s in enumerate(systems):
        m = (all_df["system"] == s).to_numpy()
        c = plt.cm.turbo(k / max(1, len(systems) - 1))
        ax1.scatter(Y[m, 0], Y[m, 1], s=2, color=c, alpha=0.15, linewidths=0)
        cx, cy = Y[m, 0].mean(), Y[m, 1].mean()
        ax1.scatter([cx], [cy], s=40, color=c, edgecolors="black", linewidths=0.7, zorder=5)
        ax1.annotate(s, (cx, cy), fontsize=6, ha="center", va="center")
    ax1.set_title("Per-system occupancy (dots) + centroids")
    ax1.set_xlabel(f"PC1 ({evr[0]:.0f}%)")
    ax1.set_ylabel(f"PC2 ({evr[1]:.0f}%)")
    ax1.grid(alpha=0.2)

    load = "  ".join(f"{p}:({pca.components_[0, i]:+.2f},{pca.components_[1, i]:+.2f})"
                     for i, p in enumerate(PARAMS))
    fig.text(0.5, 0.005, "PC loadings (PC1,PC2)  " + load, ha="center", fontsize=7)
    fig.suptitle("Shared conformational-fit landscape (6-parameter geometry PCA)")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def discover_systems():
    out = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / f"{d.name}.pdb").exists() and (d / f"{d.name}.xtc").exists():
            out.append(d.name)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*", default=None,
                    help="subset of PDB ids (default: all discovered)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    ap.add_argument("--target-frames", type=int, default=4000,
                    help="approx frames to sample per system (sets stride)")
    ap.add_argument("--stride", type=int, default=None, help="explicit frame stride")
    ap.add_argument("--full", action="store_true", help="use every frame (stride=1)")
    ap.add_argument("--contact-cutoff", type=float, default=CONTACT_CUTOFF,
                    help="drop frames whose Vα–Vβ closest Cα exceeds this (Å)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of systems")
    ap.add_argument("--plots-only", action="store_true",
                    help="regenerate all figures from existing per_system CSVs (no MD recompute)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.plots_only:
        regenerate_plots(out_dir)
        return
    per_dir = out_dir / "per_system"
    sum_dir = out_dir / "summary"
    per_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    systems = args.systems or discover_systems()
    if args.limit:
        systems = systems[: args.limit]
    print(f"[info] {len(systems)} system(s): {', '.join(systems)}")

    con = Consensus()
    valid_dfs, summ, done, errors = [], [], [], []
    err_log = out_dir / "errors.txt"

    for s in systems:
        print(f"\n=== {s} ===")
        try:
            df, valid, meta = run_system(
                s, con, per_dir, args.target_frames, args.stride, args.full, args.contact_cutoff)
            for p in PARAMS + CDR3_NUM:
                meta[f"{p}_mean"] = float(valid[p].mean())
                meta[f"{p}_std"] = float(valid[p].std(ddof=0))
                # NaN-safe (CDR3 can be NaN on frames with a missing loop/anchor)
                meta[f"{p}_range"] = float(valid[p].max() - valid[p].min()) if len(valid) else np.nan
            valid_dfs.append(valid)
            summ.append(meta)
            done.append(s)
            print(f"    {meta['n_valid']} valid / {meta['n_dropped']} dropped "
                  f"(stride {meta['stride']}); BA std={meta['BA_std']:.1f}°, dc std={meta['dc_std']:.2f}Å")
        except Exception as e:
            errors.append(s)
            with open(err_log, "a") as fh:
                fh.write(f"{s}: {e}\n{traceback.format_exc()}\n")
            print(f"    [ERROR] {e} (logged to {err_log})")

    if not valid_dfs:
        print("[fatal] no systems processed successfully.")
        return

    all_valid = pd.concat(valid_dfs, ignore_index=True)
    all_valid.to_csv(sum_dir / "all_systems_valid_angles.csv", index=False)
    summary_df = pd.DataFrame(summ)
    summary_df.to_csv(sum_dir / "system_summary.csv", index=False)

    if len(done) >= 1:
        _ridgelines(all_valid, done, sum_dir / "param_distributions.png")
        _cdr3_summary(all_valid, done, sum_dir / "cdr3_summary.png")
    if len(done) >= 2:
        _spread_heatmap(summary_df, sum_dir / "exploration_spread.png")
        _pca_landscape(all_valid, done, sum_dir / "pca_landscape.png")

    print(f"\n[done] {len(done)} ok, {len(errors)} failed. Outputs in {out_dir}")
    print(f"       summary tables + figures in {sum_dir}")


def regenerate_plots(out_dir: Path):
    """Rebuild every figure from the per-system CSVs — no MD recompute.

    Used by ``--plots-only`` to refresh visualisations (e.g. after adding the
    CDR3 metrics) without re-running the trajectories.
    """
    per_dir, sum_dir = out_dir / "per_system", out_dir / "summary"
    sum_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(per_dir.glob("*_angles.csv"))
    print(f"[plots-only] {len(csvs)} per-system CSV(s) in {per_dir}")
    valid_dfs, summ, done = [], [], []
    for c in csvs:
        df = pd.read_csv(c)
        if "alpha_cdr3_bend_deg" not in df.columns:
            print(f"    [skip] {c.name}: no CDR3 columns (stale run)"); continue
        pdbid = str(df["system"].iloc[0])
        valid = df[df["valid"]].copy()
        _plot_system(pdbid, df, valid, per_dir)
        valid_dfs.append(valid)
        meta = {"system": pdbid}
        for p in PARAMS + CDR3_NUM:
            meta[f"{p}_std"] = float(valid[p].std(ddof=0))
        summ.append(meta); done.append(pdbid)
        print(f"    [ok] {pdbid}")
    if not valid_dfs:
        print("[plots-only] nothing to plot."); return
    all_valid = pd.concat(valid_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summ)
    if len(done) >= 1:
        _ridgelines(all_valid, done, sum_dir / "param_distributions.png")
        _cdr3_summary(all_valid, done, sum_dir / "cdr3_summary.png")
    if len(done) >= 2:
        _spread_heatmap(summary_df, sum_dir / "exploration_spread.png")
        _pca_landscape(all_valid, done, sum_dir / "pca_landscape.png")
    print(f"[plots-only] refreshed {len(done)} systems; figures in {sum_dir}")


if __name__ == "__main__":
    main()
