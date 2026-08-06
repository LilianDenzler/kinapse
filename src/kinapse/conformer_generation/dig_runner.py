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

# --- DiG sampling modes ----------------------------------------------------
# Five ways to drive the sampler; every mode reuses the SAME prep + embedding +
# init_state (the expensive shared part), differing only in the inference script
# and how (if at all) the per-residue noise is masked/weighted:
#
#   vanilla        run_inference.py                          — sample from the prior, no init_state
#   no_mask        run_inference_addnoise.py                 — init_state + full noise (every residue)
#   binary_mask    run_inference_addnoise.py  -b <bool>      — hard on/off noise on the CDR residues
#   cdr_mask       ..._custom_cdrs.py         -r <value>     — value mask over the CDRs (amplitude path)
#   weighted_mask  ..._guided_physics.py      -r <float>     — graded per-region amplitude (no pocket → pure weighted noise)
ALL_MODES = ("vanilla", "no_mask", "binary_mask", "cdr_mask", "weighted_mask")

# Back-compat: the older mode names still accepted as input.
_MODE_ALIASES = {"vanilla_no_init": "vanilla", "init": "no_mask", "init_cdr_mask": "binary_mask"}

# Output sub-directory (prefix) per mode; the init_state modes append the noise-param suffix.
MODE_DIRS = {
    "vanilla": "dig_vanilla",
    "no_mask": "dig_no_mask",
    "binary_mask": "dig_binary_mask",
    "cdr_mask": "dig_cdr_mask",
    "weighted_mask": "dig_weighted_mask",
}

# weighted_mask default per-region noise amplitude (tune per experiment). CDR3 loops are the most
# flexible/variable, so they get full amplitude and CDR1/CDR2 half — a graded starting point.
DEFAULT_REGION_WEIGHTS = {
    "A_CDR1": 0.5, "A_CDR2": 0.5, "A_CDR3": 1.0,
    "B_CDR1": 0.5, "B_CDR2": 0.5, "B_CDR3": 1.0,
}


def canonical_mode(mode):
    """Map a user-supplied mode (incl. legacy aliases and ``all``) to a canonical name/list."""
    if mode == "all":
        return list(ALL_MODES)
    return _MODE_ALIASES.get(mode, mode)


def resolve_modes(dig_mode):
    """Normalise a mode spec to an ordered, de-duplicated list of canonical mode names.

    Accepts any of: ``"all"``; a single mode name or legacy alias; a comma/space-separated
    string (e.g. ``"cdr_mask,weighted_mask"``); or a list/tuple of names. Unknown modes raise
    ``ValueError``. This is what lets a driver run just a chosen subset of the five modes."""
    if isinstance(dig_mode, (list, tuple)):
        items = list(dig_mode)
    elif isinstance(dig_mode, str):
        s = dig_mode.strip()
        if s == "all":
            return list(ALL_MODES)
        items = [m.strip() for m in s.replace(" ", ",").split(",") if m.strip()]
    else:
        raise TypeError(f"dig_mode must be a str or list of modes, got {type(dig_mode).__name__}")
    out = []
    for m in items:
        if m == "all":
            for x in ALL_MODES:
                if x not in out:
                    out.append(x)
            continue
        cm = _MODE_ALIASES.get(m, m)
        if cm not in ALL_MODES:
            raise ValueError(f"unknown dig_mode {m!r} (choose from {', '.join(ALL_MODES)} | all)")
        if cm not in out:
            out.append(cm)
    if not out:
        raise ValueError("no valid modes given")
    return out


def _dig_prefix():
    """Launcher for the external DiG scripts (get_init_state / run_inference*): a configured
    env (generation.python, or generation.env_run + generation.conda_env) else this interpreter."""
    py = paths.get("generation", "python", default="")
    if py:
        return [py]
    env = paths.get("generation", "conda_env", default="")
    if env:
        import shlex
        return shlex.split(paths.get("generation", "env_run", default="") or "conda run -n") + [env, "python3"]
    return [sys.executable]

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
    pipeline* via :func:`kinapse.sequence_embedding.evoformer.compute_to` — a lightweight,
    config-driven orchestrator (kinapse shells ``conda run -n <env> …``; torch/OpenFold live
    in that env). Configure paths in ``kinapse.yaml`` under ``evoformer:`` (needs a GPU +
    AlphaFold DBs). Returns ``<final_out>/<pdb_name>.pkl``."""
    from kinapse.sequence_embedding.evoformer import compute_to
    target_pkl = os.path.join(final_out, f"{pdb_name}.pkl")
    compute_to(str(linked_pdb), str(final_out), target_pkl, use_mmseqs2_gpu=use_mmseqs2_gpu)
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
    # Combined CDR mask (binary/cdr modes) + per-region masks (weighted mode) over the linked chain.
    if regions_masked_true:
        binary_res_mask = np.asarray(pv.linked_resmask(regions_masked_true), dtype=bool)
        region_masks = {r: np.asarray(pv.linked_resmask([r]), dtype=bool) for r in regions_masked_true}
    else:
        binary_res_mask = None
        region_masks = {}
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
    # --- init_state extraction (needed by every mode except vanilla) ---
    init_state_out = os.path.join(final_out, f"{pdb_name}.init_state.npz")
    if Path(init_state_out).exists():
        print(f"Init state file already exists. Skipping init state extraction step.")
    else:
        init_state_script = paths.get_pipeline_script('get_init_state')
        # get_init_state only needs numpy + mdtraj + scipy (no torch/OpenFold), so run it in the
        # SAME interpreter as kinapse — which already imported mdtraj via TCR() above — NOT the
        # openfold_env (which lacks mdtraj). cwd=/tmp avoids getcwd errors from a stale launch dir.
        r = subprocess.run([sys.executable, init_state_script, linked_out_file,
                            "--out_path", init_state_out], cwd="/tmp")
        if r.returncode != 0 or not Path(init_state_out).exists():
            raise RuntimeError(
                f"init_state extraction failed (exit {r.returncode}); {init_state_out} not written. "
                f"Ran: {sys.executable} {init_state_script} {linked_out_file}")
        print("Init state extraction finished.")
    return Path(final_out), binary_res_mask, region_masks


def _write_time(subdir, dt, n_samples):
    per = dt / max(1, n_samples)
    with open(os.path.join(subdir, "time_per_sample.txt"), "w") as f:
        f.write(f"Total inference time: {dt} seconds\n")
        f.write(f"Time per sample: {per} seconds\n")


def _noise_subdir(final_out, prefix, noise_params):
    """Init_state modes write to ``<prefix>_trA…_rotA…_trB…_rotB…/`` (noise params in the name)."""
    suffix = f"_trA{noise_params['tr_a']}_rotA{noise_params['rot_a']}_trB{noise_params['tr_b']}_rotB{noise_params['rot_b']}"
    subdir = os.path.join(final_out, prefix + suffix)
    os.makedirs(subdir, exist_ok=True)
    return subdir


def _dump_regions(subdir, regions):
    with open(os.path.join(subdir, "regions_masked_true.json"), "w") as f:
        json.dump(list(regions or []), f, indent=4)


def _build_weighted_mask(region_masks, region_weights):
    """Per-residue float noise amplitude: each region's residues set to its weight (others 0)."""
    if not region_masks:
        raise ValueError("weighted_mask needs per-region masks; pass regions_masked_true so they get built.")
    length = len(next(iter(region_masks.values())))
    weighted = np.zeros(length, dtype=np.float32)
    for region, mask in region_masks.items():
        weighted[np.asarray(mask, dtype=bool)] = float(region_weights.get(region, 0.0))
    return weighted


def _run_addnoise(final_out, pdb_name, subdir, script_key, n_samples, noise_params,
                  mask_path=None, mask_flag=None):
    """Shared launcher for every init_state mode: resolves the script + checkpoint, wires the
    (optional) residue mask, runs the sampler in /tmp (so the SO(3) tables cache there), records
    timing. Idempotent — skips if the 200th frame (``*_199.pdb``) already exists."""
    if any(Path(subdir).glob("*_199.pdb")):
        print(f"✅ Inference outputs already exist in {subdir}. Skipping inference step.")
        return Path(subdir)
    script = paths.get_pipeline_script(script_key)
    if not script or not Path(script).exists():
        raise FileNotFoundError(
            f"DiG inference script for '{script_key}' not found: {script!r}. "
            f"Set generation.{script_key} in kinapse.yaml to the script path.")
    noise_json = save_noise_params(noise_params, subdir)
    checkpoint = paths.get_checkpoint('main_model')   # your newest model (KINAPSE_CHECKPOINT_MAIN_MODEL)
    cmd = _dig_prefix() + [script,
                    "-c", checkpoint,
                    "-i", os.path.join(final_out, f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out, f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", os.path.join(subdir, ""),   # trailing sep: run_inference.py string-concats the prefix
                    "--init-state", os.path.join(final_out, f"{pdb_name}.init_state.npz"),
                    "-n", str(n_samples),
                    "--use-gpu",
                    "--use-tqdm",
                    "--noise-params-json", noise_json]
    if mask_path and mask_flag:
        cmd += [mask_flag, mask_path]
    t0 = time.time()
    subprocess.run(cmd, cwd="/tmp")
    dt = time.time() - t0
    print(f"Total inference time: {dt} seconds")
    _write_time(subdir, dt, n_samples)
    return Path(subdir)


def run_vanilla_no_init_state(final_out, pdb_name, n_samples=100):
    """Mode ``vanilla`` — sample straight from the DiG prior (no init_state, no mask)."""
    subdir = os.path.join(final_out, MODE_DIRS["vanilla"])
    os.makedirs(subdir, exist_ok=True)
    if any(Path(subdir).glob("*_199.pdb")):
        print(f"✅ Inference outputs already exist in {subdir}. Skipping inference step.")
        return Path(subdir)
    script = paths.get_pipeline_script('run_inference')
    checkpoint = paths.get_checkpoint('main_model')
    t0 = time.time()
    subprocess.run(_dig_prefix() + [script,
                    "-c", checkpoint,
                    "-i", os.path.join(final_out, f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out, f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", os.path.join(subdir, ""),   # trailing sep: run_inference.py string-concats the prefix
                    # no init-state, no mask, no noise-params for the vanilla prior
                    "-n", str(n_samples),
                    "--use-gpu",
                    "--use-tqdm"], cwd="/tmp")
    dt = time.time() - t0
    print(f"Total inference time: {dt} seconds")
    _write_time(subdir, dt, n_samples)
    return Path(subdir)


def run_no_mask(final_out, pdb_name, n_samples=100, noise_params=None):
    """Mode ``no_mask`` — init_state + noise on every residue (run_inference_addnoise, no mask)."""
    subdir = _noise_subdir(final_out, MODE_DIRS["no_mask"], noise_params)
    return _run_addnoise(final_out, pdb_name, subdir, "run_inference_addnoise", n_samples, noise_params)


def run_binary_mask(final_out, pdb_name, n_samples=100, noise_params=None,
                    regions_masked_true=None, binary_res_mask=None):
    """Mode ``binary_mask`` — hard on/off noise on the CDR residues (run_inference_addnoise -b)."""
    subdir = _noise_subdir(final_out, MODE_DIRS["binary_mask"], noise_params)
    mask_path = os.path.join(subdir, "binary_res_mask.npy")
    np.save(mask_path, np.asarray(binary_res_mask, dtype=bool))
    print(f"✅ Saved binary residue mask to {mask_path}")
    _dump_regions(subdir, regions_masked_true)
    return _run_addnoise(final_out, pdb_name, subdir, "run_inference_addnoise", n_samples, noise_params,
                         mask_path=mask_path, mask_flag="--binary_res_mask_path")


def run_cdr_mask(final_out, pdb_name, n_samples=100, noise_params=None,
                 regions_masked_true=None, binary_res_mask=None):
    """Mode ``cdr_mask`` — value mask over the CDRs via the custom_cdrs script (-r, amplitude path)."""
    subdir = _noise_subdir(final_out, MODE_DIRS["cdr_mask"], noise_params)
    mask_path = os.path.join(subdir, "res_value_mask.npy")
    np.save(mask_path, np.asarray(binary_res_mask, dtype=np.float32))   # 1.0 on CDRs, 0.0 elsewhere
    print(f"✅ Saved CDR value mask to {mask_path}")
    _dump_regions(subdir, regions_masked_true)
    return _run_addnoise(final_out, pdb_name, subdir, "run_inference_addnoise_custom_cdrs", n_samples, noise_params,
                         mask_path=mask_path, mask_flag="-r")


def run_weighted_mask(final_out, pdb_name, n_samples=100, noise_params=None,
                      region_masks=None, region_weights=None):
    """Mode ``weighted_mask`` — graded per-region noise amplitude via the guided_physics script
    (``-r`` float mask, no pocket → pure weighted noise). Weights default to
    :data:`DEFAULT_REGION_WEIGHTS` (CDR3 full, CDR1/CDR2 half); override via ``region_weights``."""
    region_weights = dict(region_weights or DEFAULT_REGION_WEIGHTS)
    subdir = _noise_subdir(final_out, MODE_DIRS["weighted_mask"], noise_params)
    weighted = _build_weighted_mask(region_masks, region_weights)
    mask_path = os.path.join(subdir, "weighted_res_mask.npy")
    np.save(mask_path, weighted)
    with open(os.path.join(subdir, "region_weights.json"), "w") as f:
        json.dump(region_weights, f, indent=4)
    print(f"✅ Saved weighted residue mask to {mask_path}")
    return _run_addnoise(final_out, pdb_name, subdir, "run_inference_addnoise_guided_physics", n_samples, noise_params,
                         mask_path=mask_path, mask_flag="-r")


# Back-compat aliases (older internal/experiment code calls these names).
run_with_init_state = run_no_mask
run_with_init_state_cdr_mask = run_binary_mask


def _dispatch_mode(mode, final_out, pdb_name, n_samples, noise_params,
                   regions_masked_true, binary_res_mask, region_masks, region_weights):
    if mode == "vanilla":
        return run_vanilla_no_init_state(final_out, pdb_name, n_samples=n_samples)
    if mode == "no_mask":
        return run_no_mask(final_out, pdb_name, n_samples=n_samples, noise_params=noise_params)
    if mode == "binary_mask":
        return run_binary_mask(final_out, pdb_name, n_samples=n_samples, noise_params=noise_params,
                               regions_masked_true=regions_masked_true, binary_res_mask=binary_res_mask)
    if mode == "cdr_mask":
        return run_cdr_mask(final_out, pdb_name, n_samples=n_samples, noise_params=noise_params,
                            regions_masked_true=regions_masked_true, binary_res_mask=binary_res_mask)
    if mode == "weighted_mask":
        return run_weighted_mask(final_out, pdb_name, n_samples=n_samples, noise_params=noise_params,
                                 region_masks=region_masks, region_weights=region_weights)
    raise ValueError(f"unknown dig_mode {mode!r} (choose from {', '.join(ALL_MODES)} | all)")


def run_one(pdb_path, output_dir_all, pkl_dir=None, n_samples=200, dig_mode="init",
            noise_params=None, regions_masked_true=("A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"),
            compute_embeddings=True, use_mmseqs2_gpu=False, region_weights=None):
    """Generate a DiG ensemble for a **single** TCR PDB (prep → embedding → inference).

    ``dig_mode`` selects one or more of ``vanilla | no_mask | binary_mask | cdr_mask |
    weighted_mask`` (legacy ``vanilla_no_init | init | init_cdr_mask`` still accepted). Pass a
    single name, ``"all"``, a comma-separated subset (e.g. ``"cdr_mask,weighted_mask"``), or a
    list — the prep + embedding + init_state are computed **once** and shared across the modes.

    Uses the checkpoint from ``KINAPSE_CHECKPOINT_MAIN_MODEL`` (or ``generation.main_model``).
    With ``compute_embeddings=True`` (default) the Evoformer embedding is computed in-pipeline
    if not already present / not in ``pkl_dir``. Returns the frame directory ``Path`` for a single
    mode, or a ``{mode: Path}`` dict when several modes are requested."""
    if noise_params is None:
        noise_params = {"tr_a": 0.2, "rot_a": 0.2, "tr_b": 0.1, "rot_b": 0.1}
    regions_masked_true = list(regions_masked_true)
    pdb_name = os.path.basename(str(pdb_path)).replace(".pdb", "")
    output_dir = os.path.join(output_dir_all, pdb_name)
    os.makedirs(output_dir, exist_ok=True)
    preprocess_out, binary_res_mask, region_masks = run_prep(
        output_dir, str(pdb_path), pdb_name, linker_sequence="GGGGS" * 3,
        regions_masked_true=regions_masked_true, pkl_dir=pkl_dir,
        compute_embeddings=compute_embeddings, use_mmseqs2_gpu=use_mmseqs2_gpu)

    modes = resolve_modes(dig_mode)
    results = {}
    for mode in modes:
        results[mode] = _dispatch_mode(
            mode, preprocess_out, pdb_name, n_samples, noise_params,
            regions_masked_true, binary_res_mask, region_masks, region_weights)
    return results[modes[0]] if len(modes) == 1 else results


def runall(output_dir_all, all_pdb_folder, pkl_dir=None, n_samples=200, dig_mode="vanilla_no_init",
           noise_params=None, regions_masked_true=("A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"),
           compute_embeddings=True, use_mmseqs2_gpu=False, region_weights=None):
    """Batch DiG generation over every ``*.pdb`` in ``all_pdb_folder`` (see :func:`run_one`).
    ``dig_mode="all"`` runs every sampling mode per TCR."""
    os.makedirs(output_dir_all, exist_ok=True)
    for pdb_path in sorted(Path(all_pdb_folder).glob("*.pdb")):
        run_one(pdb_path, output_dir_all, pkl_dir=pkl_dir, n_samples=n_samples, dig_mode=dig_mode,
                noise_params=noise_params, regions_masked_true=regions_masked_true,
                compute_embeddings=compute_embeddings, use_mmseqs2_gpu=use_mmseqs2_gpu,
                region_weights=region_weights)


def main(argv=None):
    """CLI: run DiG on a single TCR PDB (``--pdb``) or a folder (``--pdb-dir``)."""
    import argparse
    ap = argparse.ArgumentParser(description="Run the DiG diffusion sampler on a TCR to generate a conformational ensemble.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb", help="a single TCR PDB")
    src.add_argument("--pdb-dir", help="a folder of TCR PDBs (batch)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--mode", default="no_mask",
                    help="sampling mode(s): any of {%s}, comma-separated for a subset (e.g. "
                         "cdr_mask,weighted_mask), or 'all' (default: no_mask). Legacy names also "
                         "accepted. Multiple modes share one prep/embedding." % ", ".join(ALL_MODES))
    ap.add_argument("--n", type=int, default=200, help="samples to draw (default 200)")
    ap.add_argument("--checkpoint", help="path to your DiG model — sets KINAPSE_CHECKPOINT_MAIN_MODEL")
    ap.add_argument("--pkl-dir", help="dir of precomputed Evoformer embeddings (else computed in-pipeline)")
    ap.add_argument("--no-compute-embeddings", action="store_true",
                    help="do NOT compute embeddings in-pipeline (require a precomputed pkl)")
    ap.add_argument("--mmseqs2-gpu", action="store_true", help="use MMseqs2-GPU for the MSA during embedding")
    ap.add_argument("--region-weights", default=None,
                    help="weighted_mask: JSON of {region: amplitude} (default CDR3=1.0, CDR1/2=0.5)")
    ap.add_argument("--tr-a", type=float, default=0.2); ap.add_argument("--rot-a", type=float, default=0.2)
    ap.add_argument("--tr-b", type=float, default=0.1); ap.add_argument("--rot-b", type=float, default=0.1)
    a = ap.parse_args(argv)

    try:
        modes = resolve_modes(a.mode)
    except ValueError as e:
        ap.error(str(e))
    if a.checkpoint:
        os.environ["KINAPSE_CHECKPOINT_MAIN_MODEL"] = a.checkpoint
    ckpt = paths.get_checkpoint("main_model")
    print(f"DiG checkpoint (main_model): {ckpt}")
    print(f"modes: {', '.join(modes)}")
    region_weights = json.loads(a.region_weights) if a.region_weights else None
    noise = {"tr_a": a.tr_a, "rot_a": a.rot_a, "tr_b": a.tr_b, "rot_b": a.rot_b}
    common = dict(pkl_dir=a.pkl_dir, n_samples=a.n, dig_mode=a.mode, noise_params=noise,
                  compute_embeddings=not a.no_compute_embeddings, use_mmseqs2_gpu=a.mmseqs2_gpu,
                  region_weights=region_weights)
    if a.pdb:
        run_one(a.pdb, a.out, **common)
    else:
        runall(a.out, a.pdb_dir, **common)
    print(f"✅ done — ensembles under {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
