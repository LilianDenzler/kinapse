import os
import subprocess
from pathlib import Path
import shutil
import sys

# Add parent directories to path to import path_config
from kinapse.config import get_paths

# Add TCR_Metrics to path

from kinapse.structures.tcr import TCR
from kinapse.structures.io import write_pdb
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import json
import numpy as np
import time

# Initialize path configuration
paths = get_paths()

def save_noise_params(noise_params, final_out_subdir):
    json_path = os.path.join(final_out_subdir, "noise_params.json")
    with open(json_path, "w") as f:
        json.dump(noise_params, f, indent=4)
    print(f"✅ Saved noise params to {json_path}")
    return json_path


def pdb_to_fasta(pdb_path, fasta_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_path)

    seq_records = []

    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.get_id()[0] == " "]
            sequence = "".join(seq1(res.get_resname()) for res in residues)
            record = SeqRecord(
                Seq(sequence),
                id=f"{structure.id}_{chain.id}",
                description=f"Chain {chain.id} from {pdb_path}"
            )
            seq_records.append(record)

    SeqIO.write(seq_records, fasta_path, "fasta")
    print(f"✅ Wrote {len(seq_records)} chains to {fasta_path}")


def compute_embedding(linked_pdb, final_out, pdb_name, use_mmseqs2_gpu=False):
    """Compute the OpenFold/Evoformer conditioning embedding (``<pdb_name>.pkl``) *in the
    pipeline*, by running the configured OpenFold wrapper on the linked structure, then
    normalising the produced ``.pkl`` to ``<final_out>/<pdb_name>.pkl``.

    The wrapper is ``generation.openfold_wrapper`` in the config (set it in ``kinapse.yaml``);
    it needs the OpenFold checkout + AlphaFold DBs + a GPU. Returns the embedding path."""
    target_pkl = os.path.join(final_out, f"{pdb_name}.pkl")
    wrapper = paths.get_pipeline_script("openfold_wrapper")
    if not wrapper or not Path(wrapper).exists():
        raise FileNotFoundError(
            "Cannot compute embeddings: the OpenFold wrapper is not configured/found "
            f"(generation.openfold_wrapper = {wrapper!r}). Point it at your "
            "openfold_wrapper_for_evoformer.py via kinapse.yaml, or pass a precomputed pkl_dir.")
    fasta_dir = os.path.join(final_out, "embed_fasta")
    os.makedirs(fasta_dir, exist_ok=True)
    cmd = [sys.executable, wrapper,
           "--pdb_path", str(linked_pdb),
           "--fasta_dir", fasta_dir,
           "--output_dir", str(final_out)]
    if use_mmseqs2_gpu:
        cmd.append("--usemmseq2_gpu")
    print("🧬 Computing Evoformer embedding:\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    # normalise the wrapper's output to <pdb_name>.pkl
    if not Path(target_pkl).exists():
        found = sorted(Path(final_out).rglob("*.pkl"))
        if not found:
            raise FileNotFoundError(
                f"OpenFold wrapper ran but produced no .pkl under {final_out} — check where "
                "your wrapper writes the Evoformer representation.")
        print(f"   using {found[0]} → {target_pkl}")
        shutil.copyfile(found[0], target_pkl)
    print(f"✅ Embedding ready: {target_pkl}")
    return target_pkl


def run_prep(final_out, input_pdb, pdb_name, linker_sequence="GGGGS" * 3,
             regions_masked_true=None, pkl_dir=None, compute_embeddings=True,
             use_mmseqs2_gpu=False):
    tcr=TCR(input_pdb)
    pv = tcr.pairs[0]
    linked_out_file=os.path.join(final_out, f"{pdb_name}_linked.pdb")
    print(linked_out_file)
    linked_structure=pv.linked_structure(linker_sequence)
    if regions_masked_true:
        binary_res_mask=pv.linked_resmask( regions_masked_true)
    else:
        binary_res_mask=None
    print(f"Writing linked structure to {linked_structure}")
    write_pdb(linked_out_file, linked_structure)

    # FASTA of the linked (single-chain) structure is always needed for inference
    fasta_path = os.path.join(final_out, f"{pdb_name}.fasta")
    pdb_to_fasta(linked_out_file, fasta_path)

    # --- resolve the conditioning embedding <pdb_name>.pkl ---
    target_pkl = os.path.join(final_out, f"{pdb_name}.pkl")
    src_pkl = os.path.join(pkl_dir, f"{pdb_name}.pkl") if pkl_dir else None
    if Path(target_pkl).exists():
        print(f"✅ Embedding already present: {target_pkl}")
    elif src_pkl and Path(src_pkl).exists():
        print(f"Copying precomputed embedding from {pkl_dir}")
        shutil.copyfile(src_pkl, target_pkl)
    elif compute_embeddings:
        compute_embedding(linked_out_file, final_out, pdb_name, use_mmseqs2_gpu=use_mmseqs2_gpu)
    else:
        raise FileNotFoundError(
            f"Embedding {pdb_name}.pkl not found (looked in {final_out} and pkl_dir={pkl_dir}). "
            "Pass compute_embeddings=True to generate it, or provide a precomputed pkl_dir.")
    #run inference
    if Path(os.path.join(final_out, f"{pdb_name}.init_state.npz")).exists():
        print(f"Init state file already exists. Skipping init state extraction step.")
    else:
        init_state_script = paths.get_pipeline_script('get_init_state')
        subprocess.run(["python", init_state_script,
                    linked_out_file,
                    "--out_path", os.path.join(final_out, f"{pdb_name}.init_state.npz")])
        print("Init state extraction finished.")
    return Path(final_out),binary_res_mask


def run_vanilla_no_init_state(final_out, pdb_name, n_samples=100):
    final_out_subdir=os.path.join(final_out, "dig_vanilla")
    os.makedirs(final_out_subdir, exist_ok=True)
    time_start=time.time()
    inference_script = paths.get_pipeline_script('run_inference')
    checkpoint = paths.get_checkpoint('main_model')
    subprocess.run([sys.executable, inference_script,
                    "-c", checkpoint,
                    "-i", os.path.join(final_out,  f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out,  f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", final_out_subdir,
                    #NO INIT STATE ARGUMENT HERE
                    "-n", str(n_samples),
                    #NO NOISE MASK ARGUMENT HERE
                    #NO NOISE PARAMS ARGUMENT HERE
                    "--use-gpu",
                    "--use-tqdm"])
    time_end=time.time()
    print(f"Total inference time: {time_end - time_start} seconds")
    total_time_per_sample=(time_end - time_start)/n_samples
    #write to file
    with open(os.path.join(final_out_subdir, "time_per_sample.txt"), "w") as f:
        f.write(f"Total inference time: {time_end - time_start} seconds\n")
        f.write(f"Time per sample: {total_time_per_sample} seconds\n")
    return Path(os.path.join(final_out, "dig_vanilla"))

def run_with_init_state(final_out, pdb_name, n_samples=100, noise_params=None):
    final_out_subdir=os.path.join(final_out, "dig_with_init"+f"_trA{str(noise_params['tr_a'])}_rotA{str(noise_params['rot_a'])}_trB{str(noise_params['tr_b'])}_rotB{str(noise_params['rot_b'])}/")
    if not Path(final_out_subdir).exists():
        os.makedirs(final_out_subdir)
    if any(Path(final_out_subdir).glob("*_199.pdb")):
        print(f"✅ Inference outputs already exist in {final_out_subdir}. Skipping inference step.")
        return Path(final_out_subdir)
    #write noise params to txt file
    noise_param_json=save_noise_params(noise_params, final_out_subdir)
    time_start=time.time()
    inference_addnoise_script = paths.get_pipeline_script('run_inference_addnoise')
    checkpoint = paths.get_checkpoint('main_model')
    subprocess.run([sys.executable, inference_addnoise_script,
                    "-c", checkpoint,
                    "-i", os.path.join(final_out,  f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out,  f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", final_out_subdir,
                    "--init-state", os.path.join(final_out, f"{pdb_name}.init_state.npz"),
                    "-n", str(n_samples),
                    "--use-gpu",
                    "--use-tqdm",
                    #NO NOISE MASK ARGUMENT HERE
                    "--noise-params-json", noise_param_json],cwd="/tmp")
    time_end=time.time()
    print(f"Total inference time: {time_end - time_start} seconds")
    total_time_per_sample=(time_end - time_start)/n_samples
    #write to file
    with open(os.path.join(final_out_subdir, "time_per_sample.txt"), "w") as f:
        f.write(f"Total inference time: {time_end - time_start} seconds\n")
        f.write(f"Time per sample: {total_time_per_sample} seconds\n")
    return Path(final_out_subdir)


def run_with_init_state_cdr_mask(final_out, pdb_name, n_samples=100, noise_params=None, regions_masked_true=None, binary_res_mask=None):
    final_out_subdir=os.path.join(final_out, "dig_with_init_cdr_mask"+f"_trA{str(noise_params['tr_a'])}_rotA{str(noise_params['rot_a'])}_trB{str(noise_params['tr_b'])}_rotB{str(noise_params['rot_b'])}/")
    if not Path(final_out_subdir).exists():
        os.makedirs(final_out_subdir)
    #write noise params to txt file
    noise_param_json=save_noise_params(noise_params, final_out_subdir)
    binary_res_mask_path=os.path.join(final_out_subdir, f"binary_res_mask.npy")
    np.save(binary_res_mask_path, binary_res_mask)
    print(f"✅ Saved binary residue mask to {binary_res_mask_path}")
    #save regions masked true to json
    regions_masked_true_json_path=os.path.join(final_out_subdir, "regions_masked_true.json")
    with open(regions_masked_true_json_path, "w") as f:
        json.dump(regions_masked_true, f, indent=4)
    print(f"✅ Saved regions masked true to {regions_masked_true_json_path}")
    time_start=time.time()
    inference_addnoise_script = paths.get_pipeline_script('run_inference_addnoise')
    checkpoint = paths.get_checkpoint('main_model')   # your newest model (KINAPSE_CHECKPOINT_MAIN_MODEL)
    subprocess.run([sys.executable, inference_addnoise_script,
                    "-c", checkpoint,
                    "-i", os.path.join(final_out,  f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out,  f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", final_out_subdir,
                    "--init-state", os.path.join(final_out, f"{pdb_name}.init_state.npz"),
                    "-n", str(n_samples),
                    "--use-gpu",
                    "--use-tqdm",
                    "--binary_res_mask_path", binary_res_mask_path,
                    "--noise-params-json", noise_param_json],cwd="/tmp")
    time_end=time.time()
    print(f"Total inference time: {time_end - time_start} seconds")
    total_time_per_sample=(time_end - time_start)/n_samples
    #write to file
    with open(os.path.join(final_out_subdir, "time_per_sample.txt"), "w") as f:
        f.write(f"Total inference time: {time_end - time_start} seconds\n")
        f.write(f"Time per sample: {total_time_per_sample} seconds\n")
    return Path(final_out_subdir)


def run_one(pdb_path, output_dir_all, pkl_dir=None, n_samples=200, dig_mode="init",
            noise_params=None, regions_masked_true=("A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"),
            compute_embeddings=True, use_mmseqs2_gpu=False):
    """Generate a DiG ensemble for a **single** TCR PDB (prep → embedding → inference).

    Uses the checkpoint from ``KINAPSE_CHECKPOINT_MAIN_MODEL`` (or ``generation.main_model``).
    With ``compute_embeddings=True`` (default) the Evoformer embedding is computed in-pipeline
    if not already present / not in ``pkl_dir``. Returns the frame directory."""
    if noise_params is None:
        noise_params = {"tr_a": 0.2, "rot_a": 0.2, "tr_b": 0.1, "rot_b": 0.1}
    regions_masked_true = list(regions_masked_true)
    pdb_name = os.path.basename(str(pdb_path)).replace(".pdb", "")
    output_dir = os.path.join(output_dir_all, pdb_name)
    os.makedirs(output_dir, exist_ok=True)
    preprocess_out, binary_res_mask = run_prep(
        output_dir, str(pdb_path), pdb_name, linker_sequence="GGGGS" * 3,
        regions_masked_true=regions_masked_true, pkl_dir=pkl_dir,
        compute_embeddings=compute_embeddings, use_mmseqs2_gpu=use_mmseqs2_gpu)
    if dig_mode == "vanilla_no_init":
        return run_vanilla_no_init_state(preprocess_out, pdb_name, n_samples=n_samples)
    if dig_mode == "init":
        return run_with_init_state(preprocess_out, pdb_name, n_samples=n_samples, noise_params=noise_params)
    if dig_mode == "init_cdr_mask":
        return run_with_init_state_cdr_mask(preprocess_out, pdb_name, n_samples=n_samples,
                                            noise_params=noise_params, regions_masked_true=regions_masked_true,
                                            binary_res_mask=binary_res_mask)
    raise ValueError(f"unknown dig_mode {dig_mode!r} "
                     "(vanilla_no_init | init | init_cdr_mask)")


def runall(output_dir_all, all_pdb_folder, pkl_dir=None, n_samples=200, dig_mode="vanilla_no_init",
           noise_params=None, regions_masked_true=("A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"),
           compute_embeddings=True, use_mmseqs2_gpu=False):
    """Batch DiG generation over every ``*.pdb`` in ``all_pdb_folder`` (see :func:`run_one`)."""
    os.makedirs(output_dir_all, exist_ok=True)
    for pdb_path in sorted(Path(all_pdb_folder).glob("*.pdb")):
        run_one(pdb_path, output_dir_all, pkl_dir=pkl_dir, n_samples=n_samples, dig_mode=dig_mode,
                noise_params=noise_params, regions_masked_true=regions_masked_true,
                compute_embeddings=compute_embeddings, use_mmseqs2_gpu=use_mmseqs2_gpu)


def main(argv=None):
    """CLI: run DiG on a single TCR PDB (``--pdb``) or a folder (``--pdb-dir``)."""
    import argparse
    ap = argparse.ArgumentParser(description="Run the DiG diffusion sampler on a TCR to generate a conformational ensemble.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb", help="a single TCR PDB")
    src.add_argument("--pdb-dir", help="a folder of TCR PDBs (batch)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--mode", default="init", choices=("vanilla_no_init", "init", "init_cdr_mask"),
                    help="DiG sampling mode (default: init)")
    ap.add_argument("--n", type=int, default=200, help="samples to draw (default 200)")
    ap.add_argument("--checkpoint", help="path to your DiG model — sets KINAPSE_CHECKPOINT_MAIN_MODEL")
    ap.add_argument("--pkl-dir", help="dir of precomputed Evoformer embeddings (else computed in-pipeline)")
    ap.add_argument("--no-compute-embeddings", action="store_true",
                    help="do NOT compute embeddings in-pipeline (require a precomputed pkl)")
    ap.add_argument("--mmseqs2-gpu", action="store_true", help="use MMseqs2-GPU for the MSA during embedding")
    ap.add_argument("--tr-a", type=float, default=0.2); ap.add_argument("--rot-a", type=float, default=0.2)
    ap.add_argument("--tr-b", type=float, default=0.1); ap.add_argument("--rot-b", type=float, default=0.1)
    a = ap.parse_args(argv)

    if a.checkpoint:
        os.environ["KINAPSE_CHECKPOINT_MAIN_MODEL"] = a.checkpoint
    ckpt = paths.get_checkpoint("main_model")
    print(f"DiG checkpoint (main_model): {ckpt}")
    noise = {"tr_a": a.tr_a, "rot_a": a.rot_a, "tr_b": a.tr_b, "rot_b": a.rot_b}
    common = dict(pkl_dir=a.pkl_dir, n_samples=a.n, dig_mode=a.mode, noise_params=noise,
                  compute_embeddings=not a.no_compute_embeddings, use_mmseqs2_gpu=a.mmseqs2_gpu)
    if a.pdb:
        run_one(a.pdb, a.out, **common)
    else:
        runall(a.out, a.pdb_dir, **common)
    print(f"✅ done — ensembles under {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
