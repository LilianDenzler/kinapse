import os
import math
import warnings
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd

# Biopython for quick writing/reading
from Bio.PDB import PDBParser, PDBIO

# Biotite for structural alignment
import biotite.structure as bts
import biotite.structure.io as btsio

from . import DATA_PATH
from importlib.resources import files
from pathlib import Path

warnings.filterwarnings("ignore", ".*is discontinuous.*")
from pathlib import Path


# -------------------------
# Geometry helpers
# -------------------------
Points = namedtuple("Points", ["C", "V1", "V2"])  # endpoints (absolute coords)

def as_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def angle_between(v1, v2):
    v1 = as_unit(v1); v2 = as_unit(v2)
    return math.degrees(math.acos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

# -------------------------
# Read pseudoatoms (CEN/PC1/PC2) from PDB
# Expectation: they live as residues named 'CEN','PC1','PC2' on chain 'Z'
# (This matches your change-geometry script behavior.)
# -------------------------
def read_pseudo_points(pdb_path, chain_id_main):
    """
    Returns Points for the given chain from a consensus PDB that contains
    pseudoatoms on chain 'Z' with residue names: CEN, PC1, PC2.
    The main protein chain_id_main is used only to sanity-check presence;
    pseudoatoms are read from chain Z.
    """
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("consensus", pdb_path)

    # Sanity: ensure the main chain exists
    if chain_id_main not in [ch.id for ch in s[0]]:
        raise ValueError(f"Chain '{chain_id_main}' not found in {pdb_path}")

    # Find pseudoatoms in chain Z
    try:
        z = s[0]["Z"]
    except KeyError:
        raise ValueError(f"Chain 'Z' with pseudoatoms (CEN/PC1/PC2) not found in {pdb_path}")

    def _get_first_atom(res_name):
        for res in z:
            if res.get_resname() == res_name:
                for atom in res:
                    return atom.get_coord()
        raise ValueError(f"Pseudoatom '{res_name}' not found in chain Z of {pdb_path}")

    C  = _get_first_atom("CEN")
    V1 = _get_first_atom("PC1")
    V2 = _get_first_atom("PC2")
    return Points(C=np.array(C, float), V1=np.array(V1, float), V2=np.array(V2, float))

# -------------------------
# Biotite alignment
# -------------------------
def apply_affine_to_atomarray(atomarray, transform):
    arr = atomarray.copy()
    M = np.asarray(transform.as_matrix(), dtype=np.float64)
    if M.shape == (1, 4, 4):
        M = M[0]
    R = M[:3, :3]
    t = M[:3,  3]
    coords = np.asarray(arr.coord, dtype=np.float64)
    new_coords = (coords @ R.T) + t
    arr.coord = new_coords.astype(np.float32)
    return arr

def align_with_biotite(static_pdb_file: str, mobile_pdb_file: str, output_pdb_file: str, chain_name: str,static_consenus_res: list =None, mobile_consenus_res: list =None):
    """
    Align 'mobile' to 'static' using C-alpha atoms of a single protein chain.
    Saves the aligned full-atom mobile structure to output_pdb_file.
    Pseudoatoms (chain Z) ride along with the same transform if present.
    """
    static_structure = btsio.load_structure(static_pdb_file, model=1)
    mobile_structure = btsio.load_structure(mobile_pdb_file, model=1)

    static_mask = (static_structure.atom_name == "CA") & (static_structure.chain_id == chain_name)
    mobile_mask = (mobile_structure.atom_name == "CA") & (mobile_structure.chain_id == chain_name)

    static_ca = static_structure[static_mask]
    mobile_ca = mobile_structure[mobile_mask]
    if static_consenus_res:
        static_ca = static_ca[np.isin(static_ca.res_id, static_consenus_res)]
    if mobile_consenus_res:
        mobile_ca = mobile_ca[np.isin(mobile_ca.res_id, mobile_consenus_res)]
    if static_ca.array_length() < 4 or mobile_ca.array_length() < 4:
        raise ValueError(f"Not enough CA atoms to align on chain {chain_name}")

    _, transform, _, _ = bts.superimpose_structural_homologs(
        fixed=static_ca, mobile=mobile_ca, max_iterations=1
    )
    mobile_full_aligned = apply_affine_to_atomarray(mobile_structure, transform)
    btsio.save_structure(output_pdb_file, mobile_full_aligned)
    print(f"💾 Saved aligned structure to '{output_pdb_file}'")

# -------------------------
# CGO arrow helper (for PyMOL script text)
# -------------------------
def add_cgo_arrow(start, end, color, radius=0.3):
    return f"""[
        cgo.CYLINDER,{start[0]:.3f},{start[1]:.3f},{start[2]:.3f},
                     {end[0]:.3f},{end[1]:.3f},{end[2]:.3f},
                     {radius},
                     {color[0]},{color[1]},{color[2]},
                     {color[0]},{color[1]},{color[2]},
        cgo.CONE,{end[0]:.3f},{end[1]:.3f},{end[2]:.3f},
                 {start[0]:.3f},{start[1]:.3f},{start[2]:.3f},
                 {radius*1.5},0.0,
                 {color[0]},{color[1]},{color[2]},
                 {color[0]},{color[1]},{color[2]},1.0
    ]"""

# -------------------------
# CDR3 loop bend / apex metric (IMGT anchors 104 <-> 118)
# -------------------------
# The CDR3 is the stretch strictly between the conserved IMGT anchors 104
# (2nd-CYS) and 118 (J-PHE/TRP-GLY). Given the two anchor CA's A,B (the base
# "chord"), their midpoint M and the domain centroid C, we work in the plane
# perpendicular to the chord and report the apex loop's *deviation from pointing
# straight away from the domain body*:
#
#     bend_deg = 180 - angle(base->centroid, base->apex)     # 0 = straight
#
# signed by a stable in-plane lateral axis. This replaces the earlier
# ``(x - 180) % 360`` output, which placed a 0/360 wrap exactly at the modal
# (near-straight) loop pose and took its sign from a cross product that is
# degenerate there -- both fatal for per-frame MD statistics.
def _res_sort_key(res):
    _, resseq, icode = res.get_id()
    return (int(resseq), (icode or " ").strip())


def _get_ca_coord(res):
    if "CA" not in res:
        return None
    return np.asarray(res["CA"].get_coord(), dtype=float)


def _pymol_resi_string(res):
    _, resseq, icode = res.get_id()
    icode = (icode or " ").strip()
    return f"{int(resseq)}{icode}" if icode else f"{int(resseq)}"


def _collect_chain_residues(model0, chain_id):
    if chain_id not in model0:
        raise ValueError(f"Chain '{chain_id}' not found in structure.")
    residues = []
    for res in model0[chain_id]:
        if res.get_id()[0] != " ":          # skip hetero / water
            continue
        if _get_ca_coord(res) is None:
            continue
        residues.append(res)
    residues.sort(key=_res_sort_key)
    return residues


def _pick_res_by_resseq(residues, resseq):
    matches = [r for r in residues if int(r.get_id()[1]) == int(resseq)]
    if not matches:
        return None
    for r in matches:                        # prefer blank insertion code
        if (r.get_id()[2] or " ").strip() == "":
            return r
    return matches[0]


def compute_cdr3_bend_metric_from_anchors(input_pdb, chain_id, chain_centroid,
                                          anchor_pre_resseq=104, anchor_post_resseq=118):
    """Signed CDR3 bend + apex protrusion from the IMGT framework anchors.

    Returns a dict with ``bend_deg`` (signed deviation from straight; 0 means the
    loop points directly away from the domain centroid), ``bend_abs_deg`` (the
    raw ``angle(base->centroid, base->apex)``, ~180 for a straight loop),
    ``apex_height_A`` (apex CA distance from the base chord) and ``apex_resi``.
    All are rigid-motion invariant; only the loop CA's and ``chain_centroid``
    must share a coordinate frame.
    """
    parser = PDBParser(QUIET=True)
    model0 = parser.get_structure("tcr", str(input_pdb))[0]

    residues = _collect_chain_residues(model0, chain_id)
    if len(residues) < 10:
        raise ValueError(f"Too few residues with CA in chain {chain_id}.")

    pre = _pick_res_by_resseq(residues, anchor_pre_resseq)
    post = _pick_res_by_resseq(residues, anchor_post_resseq)
    if pre is None or post is None:
        raise ValueError(
            f"Missing anchor(s) {anchor_pre_resseq}/{anchor_post_resseq} on chain {chain_id}.")

    i_pre, i_post = residues.index(pre), residues.index(post)
    if i_post <= i_pre + 1:
        raise ValueError(f"No residues between anchors on chain {chain_id}.")

    cdr3_residues = residues[i_pre + 1:i_post]
    if len(cdr3_residues) < 3:
        raise ValueError(f"Too few CDR3 residues on chain {chain_id} (n={len(cdr3_residues)}).")

    A = _get_ca_coord(pre)
    B = _get_ca_coord(post)
    uhat = as_unit(B - A)                      # base chord, pre(104) -> post(118)
    M = 0.5 * (A + B)

    # Reference: base midpoint -> domain centroid, projected off the chord.
    C = np.asarray(chain_centroid, float)
    v = C - M
    v_perp = v - np.dot(v, uhat) * uhat
    v_perp_norm = float(np.linalg.norm(v_perp))

    cdr_coords = [(r, _get_ca_coord(r)) for r in cdr3_residues]
    cdr_coords = [(r, c) for r, c in cdr_coords if c is not None]
    if not cdr_coords:
        raise ValueError(f"No CDR3 CA coords on chain {chain_id}.")

    def _height_from_chord(coord):
        w = coord - M
        return float(np.linalg.norm(w - np.dot(w, uhat) * uhat))

    def _apex_score(coord):
        return min(float(np.linalg.norm(coord - A)), float(np.linalg.norm(coord - B)))

    # Deterministic apex: furthest from the nearer anchor, tie-break by chord
    # height then by sequence-centrality.
    mid_index = (len(cdr3_residues) - 1) / 2.0
    res_to_idx = {r: i for i, r in enumerate(cdr3_residues)}

    def _key(rc):
        r, coord = rc
        return (_apex_score(coord), _height_from_chord(coord), -abs(res_to_idx[r] - mid_index))

    r_apex, R = max(cdr_coords, key=_key)

    w = R - M
    w_perp = w - np.dot(w, uhat) * uhat
    w_perp_norm = float(np.linalg.norm(w_perp))

    # Bend = deviation from straight, signed by a stable in-plane lateral axis.
    if v_perp_norm > 1e-6 and w_perp_norm > 1e-6:
        vhat = v_perp / v_perp_norm
        what = w_perp / w_perp_norm
        bend_abs_deg = angle_between(vhat, what)         # ~180 for a straight loop
        bend_deg = 180.0 - bend_abs_deg                  # 0 = straight, no wrap
        lat = np.cross(uhat, vhat)                       # stable in-plane lateral axis
        lat_norm = float(np.linalg.norm(lat))
        sign = 1.0
        if lat_norm > 1e-9 and np.dot(what, lat / lat_norm) < 0:
            sign = -1.0
        bend_signed_deg = sign * bend_deg
    else:
        bend_abs_deg = bend_deg = bend_signed_deg = float("nan")

    return {
        "chain_id": chain_id,
        "anchor_pre_resi": _pymol_resi_string(pre),
        "anchor_post_resi": _pymol_resi_string(post),
        "apex_resi": _pymol_resi_string(r_apex),
        "A_anchor": A, "B_anchor": B, "M_mid": M, "C_chain": C, "R_apex": R,
        "bend_deg": bend_signed_deg,          # signed deviation from straight (0 = straight)
        "bend_abs_deg": bend_abs_deg,         # raw angle(centroid, apex); ~180 = straight
        "apex_height_A": _height_from_chord(R),
        "centroid_height_A": v_perp_norm,
        "apex_score_minDist": _apex_score(R),
    }


def _safe_cdr3(pdb_path, chain_id, centroid):
    """CDR3 metric that degrades to NaNs instead of aborting the geometry."""
    try:
        return compute_cdr3_bend_metric_from_anchors(
            input_pdb=pdb_path, chain_id=chain_id, chain_centroid=centroid)
    except Exception as e:  # missing loop / anchors / CAs on this frame
        warnings.warn(f"CDR3 metric failed for chain {chain_id}: {e}")
        return {"apex_resi": None, "bend_deg": float("nan"),
                "bend_abs_deg": float("nan"), "apex_height_A": float("nan")}


# -------------------------
# Core processing (NO geometry modification of the input)
# -------------------------
def process(input_pdb, consA_with_pca, consB_with_pca, out_dir, vis_folder=None, A_consenus_res=None, B_consenus_res=None):
    """
    1) Renumbered input is aligned to consensus A (chain A).
    2) Consensus B is aligned to the aligned input (chain B).
    3) Read pseudoatoms (CEN/PC1/PC2) from consensus A and aligned consensus B.
    4) Compute BA, BC1/2, AC1/2, dc from those points.
    5) Optionally write and run a PyMOL visualization script.
    """
    parser = PDBParser(QUIET=True)

    os.makedirs(out_dir, exist_ok=True)
    aligned_input_path = os.path.join(out_dir, "aligned_input.pdb")
    aligned_consB_path = os.path.join(out_dir, "aligned_consB.pdb")

    # Align input (mobile) to consensus A (static) on chain A
    align_with_biotite(
        static_pdb_file=consA_with_pca,
        mobile_pdb_file=input_pdb,
        output_pdb_file=aligned_input_path,
        chain_name="A",
        static_consenus_res=A_consenus_res,
        mobile_consenus_res=A_consenus_res
    )
    # Align consensus B (mobile) to aligned input (static) on chain B
    align_with_biotite(
        static_pdb_file=aligned_input_path,
        mobile_pdb_file=consB_with_pca,
        output_pdb_file=aligned_consB_path,
        chain_name="B",
        static_consenus_res=B_consenus_res,
        mobile_consenus_res=B_consenus_res
    )

    # Read Points from pseudoatoms
    Apts = read_pseudo_points(consA_with_pca, chain_id_main="A")
    Bpts = read_pseudo_points(aligned_consB_path, chain_id_main="B")

    # Compute geometry
    Cvec = as_unit(Bpts.C - Apts.C)
    A1 = as_unit(Apts.V1 - Apts.C)
    A2 = as_unit(Apts.V2 - Apts.C)
    B1 = as_unit(Bpts.V1 - Bpts.C)
    B2 = as_unit(Bpts.V2 - Bpts.C)

    # BA (signed dihedral-like torsion) using plane-projected method
    nx = np.cross(A1, Cvec)
    ny = np.cross(Cvec, nx)
    Lp = as_unit([0.0, np.dot(A1, nx), np.dot(A1, ny)])
    Hp = as_unit([0.0, np.dot(B1, nx), np.dot(B1, ny)])
    BA = angle_between(Lp, Hp)
    if np.cross(Lp, Hp)[0] < 0:
        BA = -BA

    BC1 = angle_between(B1, -Cvec)
    AC1 = angle_between(A1,  Cvec)
    BC2 = angle_between(B2, -Cvec)
    AC2 = angle_between(A2,  Cvec)
    dc  = float(np.linalg.norm(Bpts.C - Apts.C))

    # CDR3 loop bend/apex. Read from the consensus-A-aligned input so the loop
    # CA's and the domain centroids (Apts.C / Bpts.C) live in one frame; the
    # metric is rigid-motion invariant so the choice of frame does not matter.
    # Defensive: a bad loop must never abort the 6-parameter geometry.
    alpha_cdr3 = _safe_cdr3(aligned_input_path, "A", Apts.C)
    beta_cdr3  = _safe_cdr3(aligned_input_path, "B", Bpts.C)

    # Visualization outputs (optional)
    if vis_folder:
        vis_folder = Path(vis_folder)
        vis_folder.mkdir(exist_ok=True, parents=True)
        io = PDBIO()
        input_struct = parser.get_structure("input_aligned", aligned_input_path)
        consA = parser.get_structure("consA", consA_with_pca)
        consB = parser.get_structure("consB_aligned", aligned_consB_path)

        input_aligned_viz = os.path.join(vis_folder, f"{Path(input_pdb).stem}_aligned_input.pdb")
        consA_out_viz     = os.path.join(vis_folder, f"{Path(consA_with_pca).stem}.pdb")
        consB_out_viz     = os.path.join(vis_folder, f"{Path(consB_with_pca).stem}_aligned.pdb")
        io.set_structure(input_struct); io.save(input_aligned_viz)
        io.set_structure(consA);        io.save(consA_out_viz)
        io.set_structure(consB);        io.save(consB_out_viz)

        generate_pymol_script(
            input_aligned_viz=os.path.abspath(input_aligned_viz),
            consA_pdb=os.path.abspath(consA_out_viz),
            consB_pdb=os.path.abspath(consB_out_viz),
            Apts=Apts, Bpts=Bpts,
            vis_folder=str(vis_folder)
        )
        os.system(f"pymol -cq {os.path.join(str(vis_folder), 'vis.py')}")
    else:
        input_aligned_viz = os.path.abspath(aligned_input_path)

    return {
        "pdb_name": Path(input_pdb).stem,
        "BA": BA, "BC1": BC1, "AC1": AC1, "BC2": BC2, "AC2": AC2, "dc": dc,
        "alpha_cdr3_bend_deg": alpha_cdr3["bend_deg"],
        "alpha_cdr3_apex_height_A": alpha_cdr3["apex_height_A"],
        "alpha_cdr3_apex_resi": alpha_cdr3["apex_resi"],
        "beta_cdr3_bend_deg": beta_cdr3["bend_deg"],
        "beta_cdr3_apex_height_A": beta_cdr3["apex_height_A"],
        "beta_cdr3_apex_resi": beta_cdr3["apex_resi"],
        "input_aligned": input_aligned_viz
    }

# -------------------------
# PyMOL visualization (matches your previous style)
# -------------------------
def generate_pymol_script(input_aligned_viz, consA_pdb, consB_pdb, Apts, Bpts, vis_folder):
    pdb_name = Path(input_aligned_viz).stem
    scale = 1.0

    # Precompute lists for f-string insertion (avoid inline math in { ... } exprs)
    A_C = Apts.C.tolist()
    B_C = Bpts.C.tolist()
    a1_end = (Apts.C + scale * (Apts.V1 - Apts.C)).tolist()
    a2_end = (Apts.C + scale * (Apts.V2 - Apts.C)).tolist()
    b1_end = (Bpts.C + scale * (Bpts.V1 - Bpts.C)).tolist()
    b2_end = (Bpts.C + scale * (Bpts.V2 - Bpts.C)).tolist()

    png_path = os.path.join(vis_folder, f"{pdb_name}_final_vis.png")
    pse_path = os.path.join(vis_folder, f"{pdb_name}_final_vis.pse")
    vis_script = os.path.join(vis_folder, "vis.py")

    script = f"""
import numpy as np
from pymol import cmd, cgo

cmd.load("{input_aligned_viz}","input_{pdb_name}")
cmd.load("{consA_pdb}","consA_{pdb_name}")
cmd.load("{consB_pdb}","consB_{pdb_name}")

cmd.bg_color("white")
cmd.hide("everything","all")

# Input TCR (aligned) colors
cmd.show("cartoon","input_{pdb_name}")
cmd.color("marine","input_{pdb_name} and chain A")
cmd.color("teal","input_{pdb_name} and chain B")

# Consensus overlays
cmd.show("cartoon","consA_{pdb_name} or consB_{pdb_name}")
cmd.color("gray70","consA_{pdb_name}")
cmd.color("gray70","consB_{pdb_name}")
cmd.set("cartoon_transparency", 0.5, "consA_{pdb_name} or consB_{pdb_name}")

# Pseudoatoms for centroids & scaled PC endpoints
cmd.pseudoatom("centroid_A_{pdb_name}", pos={A_C}, color="red")
cmd.pseudoatom("centroid_B_{pdb_name}", pos={B_C}, color="orange")
cmd.pseudoatom("PCA_A1_{pdb_name}", pos={a1_end}, color="white")
cmd.pseudoatom("PCA_A2_{pdb_name}", pos={a2_end}, color="white")
cmd.pseudoatom("PCA_B1_{pdb_name}", pos={b1_end}, color="white")
cmd.pseudoatom("PCA_B2_{pdb_name}", pos={b2_end}, color="white")
cmd.show("spheres","centroid_A_{pdb_name} or centroid_B_{pdb_name} or PCA_A1_{pdb_name} or PCA_A2_{pdb_name} or PCA_B1_{pdb_name} or PCA_B2_{pdb_name}")
cmd.set("sphere_scale", 0.5, "centroid_A_{pdb_name} or centroid_B_{pdb_name} or PCA_A1_{pdb_name} or PCA_A2_{pdb_name} or PCA_B1_{pdb_name} or PCA_B2_{pdb_name}")

# CGO arrows: use precomputed endpoints
cmd.load_cgo({add_cgo_arrow(A_C, a1_end, (0.2, 0.5, 1.0))}, "PC1_A_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(A_C, a2_end, (0.1, 0.8, 0.1))}, "PC2_A_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(B_C, b1_end, (1.0, 0.5, 0.2))}, "PC1_B_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(B_C, b2_end, (0.8, 0.8, 0.1))}, "PC2_B_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(A_C, B_C, (0.5, 0.0, 0.5))}, "dc_vec_{pdb_name}")

# Measurements (wizard-equivalent)
cmd.distance("dc_len_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}")
cmd.angle("BC1_ang_{pdb_name}", "PCA_B1_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}")
cmd.angle("BC2_ang_{pdb_name}", "PCA_B2_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}")
cmd.angle("AC1_ang_{pdb_name}", "PCA_A1_{pdb_name}", "centroid_A_{pdb_name}", "centroid_B_{pdb_name}")
cmd.angle("AC2_ang_{pdb_name}", "PCA_A2_{pdb_name}", "centroid_A_{pdb_name}", "centroid_B_{pdb_name}")
cmd.dihedral("BA_dih_{pdb_name}", "PCA_B1_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}", "PCA_A1_{pdb_name}")

# Ensure measurement objects are visible
cmd.enable("dc_len_{pdb_name}")
cmd.enable("BC1_ang_{pdb_name}")
cmd.enable("BC2_ang_{pdb_name}")
cmd.enable("AC1_ang_{pdb_name}")
cmd.enable("AC2_ang_{pdb_name}")
cmd.enable("BA_dih_{pdb_name}")

# Global style
cmd.set("dash_width", 3.0)
cmd.set("dash_gap", 0.2)
cmd.set("dash_round_ends", 0)
cmd.set("label_size", 18)
cmd.set("label_color", "black")
cmd.set("label_distance_digits", 2)
cmd.set("label_angle_digits", 1)

# Chain labels on INPUT TCR only
def _centroid(selection):
    arr = cmd.get_coords(selection, state=1)
    if arr is None or len(arr) == 0:
        return None
    import numpy as _np
    return _np.mean(arr, axis=0)

alpha_sel = "input_{pdb_name} and chain A and polymer"
beta_sel  = "input_{pdb_name} and chain B and polymer"
alpha_pos = _centroid(alpha_sel)
beta_pos  = _centroid(beta_sel)
if alpha_pos is not None:
    cmd.pseudoatom("label_alpha_chain_{pdb_name}", pos=alpha_pos.tolist(), label="TCR α")
if beta_pos is not None:
    cmd.pseudoatom("label_beta_chain_{pdb_name}", pos=beta_pos.tolist(), label="TCR β")
cmd.hide("everything", "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")
cmd.show("labels", "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")
cmd.set("label_size", 20, "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")
cmd.set("label_color", "black", "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")
cmd.set("label_outline_color", "white", "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")

cmd.orient()
cmd.zoom("all", 1.2)
cmd.png(r"{png_path}", dpi=300, ray=1)
cmd.save(r"{pse_path}")
cmd.quit()
"""
    with open(vis_script, "w") as f:
        f.write(script)
    os.system(f"pymol -cq {vis_script}")
    print(f"✅ PyMOL script written: {vis_script}")

# -------------------------
# Public API (similar to your previous run() signature)
# -------------------------
def run(input_pdb, out_path, vis=True):

    consA_pca_path = os.path.join(DATA_PATH, "chain_A/average_structure_with_pca.pdb")
    consB_pca_path = os.path.join(DATA_PATH, "chain_B/average_structure_with_pca.pdb")
     #read file with consensus alignment residues as list of integers
    with open(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    A_consenus_res = [int(x) for x in content.split(",") if x.strip()]
    with open(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    B_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    pdb_name = Path(input_pdb).stem
    out_dir = Path(out_path); out_dir.mkdir(exist_ok=True)
    tmp_out = out_dir / pdb_name; tmp_out.mkdir(exist_ok=True)
    vis_folder = tmp_out / "vis"
    if vis:
        vis_folder.mkdir(exist_ok=True)


    # write_renumbered_fv lives in get_anchor_coords (was never imported here in
    # the original TCR_Metrics — a latent NameError; imported lazily to avoid a cycle).
    from .get_anchor_coords import write_renumbered_fv
    out_path, fv_input=write_renumbered_fv(tmp_out, input_pdb)

    # Process (align + compute angles + visualize)
    result = process(
        input_pdb=fv_input,
        consA_with_pca=consA_pca_path,
        consB_with_pca=consB_pca_path,
        out_dir=str(tmp_out),
        vis_folder=str(vis_folder) if vis else None,
        A_consenus_res=A_consenus_res,
        B_consenus_res=B_consenus_res
    )

    # Save CSV
    df = pd.DataFrame([result])[[
        "pdb_name", "BA", "BC1", "AC1", "BC2", "AC2", "dc",
        "alpha_cdr3_bend_deg", "alpha_cdr3_apex_height_A", "alpha_cdr3_apex_resi",
        "beta_cdr3_bend_deg", "beta_cdr3_apex_height_A", "beta_cdr3_apex_resi",
    ]]
    df.to_csv(tmp_out / "angles_results.csv", index=False)
    print(f"📄 Saved: {tmp_out/'angles_results.csv'}")
    if vis:
        print(f"🖼️  Figures/PSE in: {vis_folder}")
    return df

# -------------------------
# Example / CLI
# -------------------------
def main():
    # CLI
    import argparse
    parser = argparse.ArgumentParser(description="Calculate TCR geometry from pseudoatom-defined axes (no geometry modification).")
    parser.add_argument("--input_pdb", type=str, help="Path to input PDB.")
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--vis", action="store_true", help="PyMOL visualization.")
    args = parser.parse_args()
    vis_val= True if args.vis else False
    if args.input_pdb:
        run(args.input_pdb, args.out_path, vis=vis_val)


if __name__ == "__main__":
    main()