from kinapse.geometry.calc_geometry import *
import MDAnalysis as mda
import tempfile
from tqdm import tqdm
import sys, os
from contextlib import redirect_stdout
import warnings
from . import DATA_PATH
from pathlib import Path

warnings.filterwarnings("ignore", message=".*formalcharges.*")

def run(input_traj, input_top, stride=1, start=0, stop=None):
    """Per-frame TCR α/β docking geometry over a trajectory.

    ``stride``/``start``/``stop`` slice the trajectory (``u.trajectory[start:stop:stride]``)
    so large ensembles can be subsampled — essential for multi-10k-frame runs. The
    topology must be IMGT-numbered (chains A=α, B=β) since ``process`` selects the
    consensus-alignment residues and the CDR3 anchors (104/118) by IMGT resseq.
    """
    consA_pca_path = os.path.join(DATA_PATH, "chain_A/average_structure_with_pca.pdb")
    consB_pca_path = os.path.join(DATA_PATH, "chain_B/average_structure_with_pca.pdb")
     #read file with consensus alignment residues as list of integers
    with open(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    A_consenus_res = [int(x) for x in content.split(",") if x.strip()]
    with open(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    B_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    u = mda.Universe(input_top, input_traj)
    # Prepare arrays for results; list-of-dicts is fine but arrays are faster
    frames = []
    times  = []
    BA_arr  = []
    BC1_arr = []
    AC1_arr = []
    BC2_arr = []
    AC2_arr = []
    dc_arr  = []
    a_cdr3_bend = []
    a_cdr3_apexh = []
    a_cdr3_apexr = []
    b_cdr3_bend = []
    b_cdr3_apexh = []
    b_cdr3_apexr = []

    # Stream frames (optionally strided/sliced)
    sliced = u.trajectory[start:stop:stride]
    for ts in tqdm(sliced, total=len(sliced), desc="Processing frames"):
        # Write out current frame to PDB
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
            with tempfile.TemporaryDirectory() as td:
                tmp_pdb = Path(td) / "frame.pdb"
                u.atoms.write(tmp_pdb.as_posix())  # writes PDB

                # Process (align + compute angles + visualize)
                result_frame = process(
                    input_pdb=tmp_pdb,
                    consA_with_pca=consA_pca_path,
                    consB_with_pca=consB_pca_path,
                    out_dir=str(td),
                    vis_folder=None,
                    A_consenus_res=A_consenus_res,
                    B_consenus_res=B_consenus_res
                )
            BA, BC1, AC1, BC2, AC2, dc = (
                result_frame["BA"],
                result_frame["BC1"],
                result_frame["AC1"],
                result_frame["BC2"],
                result_frame["AC2"],
                result_frame["dc"],
            )
            frames.append(ts.frame)
            times.append(getattr(ts, "time", np.nan))  # ps if present
            BA_arr.append(BA); BC1_arr.append(BC1); AC1_arr.append(AC1)
            BC2_arr.append(BC2); AC2_arr.append(AC2); dc_arr.append(dc)
            a_cdr3_bend.append(result_frame["alpha_cdr3_bend_deg"])
            a_cdr3_apexh.append(result_frame["alpha_cdr3_apex_height_A"])
            a_cdr3_apexr.append(result_frame["alpha_cdr3_apex_resi"])
            b_cdr3_bend.append(result_frame["beta_cdr3_bend_deg"])
            b_cdr3_apexh.append(result_frame["beta_cdr3_apex_height_A"])
            b_cdr3_apexr.append(result_frame["beta_cdr3_apex_resi"])
    df = pd.DataFrame({
        "frame": frames,
        "time_ps": times,
        "BA": BA_arr,
        "BC1": BC1_arr,
        "AC1": AC1_arr,
        "BC2": BC2_arr,
        "AC2": AC2_arr,
        "dc": dc_arr,
        "alpha_cdr3_bend_deg": a_cdr3_bend,
        "alpha_cdr3_apex_height_A": a_cdr3_apexh,
        "alpha_cdr3_apex_resi": a_cdr3_apexr,
        "beta_cdr3_bend_deg": b_cdr3_bend,
        "beta_cdr3_apex_height_A": b_cdr3_apexh,
        "beta_cdr3_apex_resi": b_cdr3_apexr,
    })
    return df


