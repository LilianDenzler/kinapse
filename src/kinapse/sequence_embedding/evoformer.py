"""OpenFold / Evoformer representation — the DiG conditioning embedding.

**Lightweight by design:** kinapse never imports torch or OpenFold. This module only
*orchestrates* — it builds and runs the OpenFold command inside its own conda env
(``conda run -n <env> python run_pretrained_openfold_shortened.py …``), so the heavy
stack lives entirely in that env. Every path is configurable (no ``/workspaces`` or
``/mnt/bob`` hardcoding): values come from the ``evoformer`` section of
:mod:`kinapse.config` (override in ``kinapse.yaml``).

The run script writes ``<name>_output_dict.pkl`` under ``output_dir`` (that's the
representation the DiG sampler conditions on).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from kinapse.config import get_paths

# evoformer config keys -> (default relative to project_root, description). Real values
# belong in kinapse.yaml; the defaults are generic placeholders so kinapse stays shareable.
_REQUIRED = ("openfold_dir", "mmcif_dir", "uniref90", "mgnify", "pdb70", "uniclust30", "bfd")


def _cfg():
    ev = get_paths().get("evoformer") or {}
    if not ev:
        raise RuntimeError(
            "No `evoformer` config found. Add an `evoformer:` section to your kinapse.yaml "
            "(openfold_dir, conda_env, mmcif_dir + AlphaFold DB paths) and point kinapse at it "
            "with KINAPSE_CONFIG. See docs/generation.md.")
    return ev


def run(pdb_path, output_dir, fasta_dir: Optional[str] = None,
        use_mmseqs2_gpu: bool = False, precom_alignments_dir: Optional[str] = None) -> str:
    """Compute the Evoformer representation for one (linked) structure.

    Makes a FASTA from ``pdb_path`` then runs OpenFold in its conda env; returns
    ``output_dir`` (OpenFold writes ``<name>_output_dict.pkl`` there). Raises a clear
    error if the ``evoformer`` config / required paths are missing.
    """
    from kinapse.sequence_embedding.fasta import pdb_to_fasta

    ev = _cfg()
    if ev.get("backend", "openfold") == "evoformer2":
        return _run_evoformer2(pdb_path, output_dir, ev)

    missing = [k for k in _REQUIRED if not ev.get(k)]
    if missing:
        raise RuntimeError(f"evoformer config is missing keys: {', '.join(missing)} "
                           "(set them in kinapse.yaml under `evoformer:`).")
    openfold_dir = ev["openfold_dir"]
    run_script = ev.get("run_script", "run_pretrained_openfold_shortened.py")
    script_path = run_script if os.path.isabs(run_script) else os.path.join(openfold_dir, run_script)
    if not Path(script_path).exists():
        raise FileNotFoundError(f"OpenFold run script not found: {script_path} "
                                "(check evoformer.openfold_dir / run_script).")

    output_dir = str(output_dir)
    fasta_dir = fasta_dir or os.path.join(output_dir, "embed_fasta")
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    pdb_to_fasta(pdb_path, os.path.join(fasta_dir, "seq.fasta"))

    if use_mmseqs2_gpu and not precom_alignments_dir:
        # optional MMseqs2-GPU MSA precompute (also config-driven); see msa.py
        from kinapse.sequence_embedding.msa import run_pipeline_with_precomputed_alignments
        precom_alignments_dir = run_pipeline_with_precomputed_alignments(
            fasta_path=os.path.join(fasta_dir, "seq.fasta"),
            align_dir=os.path.join(output_dir, "alignments"),
            output_dir=output_dir)

    # How to launch the openfold env. `python` (a direct interpreter path) wins if set;
    # otherwise `env_run` + `conda_env` (default `conda run -n`; micromamba/mamba users set
    # env_run: "micromamba run -n"). env_run activates the env so MSA binaries are on PATH.
    import shlex
    conda_env = ev.get("conda_env", "openfold_env")
    python_bin = ev.get("python")
    if python_bin:
        prefix = [python_bin, "-u"]
    else:
        prefix = shlex.split(ev.get("env_run", "conda run -n")) + [conda_env, "python3", "-u"]
    cmd = prefix + [script_path,
           fasta_dir, ev["mmcif_dir"],
           "--output_dir", output_dir,
           "--config_preset", ev.get("config_preset", "model_1_ptm"),
           "--uniref90_database_path", ev["uniref90"],
           "--mgnify_database_path", ev["mgnify"],
           "--pdb70_database_path", ev["pdb70"],
           "--uniclust30_database_path", ev["uniclust30"],
           "--bfd_database_path", ev["bfd"],
           "--model_device", ev.get("model_device", "cuda:0"),
           "--save_outputs"]
    if precom_alignments_dir:
        cmd += ["--use_precomputed_alignments", str(precom_alignments_dir)]

    print("🧬 OpenFold (Evoformer) — running in env %r, cwd %s:\n   %s"
          % (conda_env, openfold_dir, " ".join(cmd)))
    subprocess.run(cmd, check=True, cwd=openfold_dir)
    return output_dir


def _run_evoformer2(pdb_path, output_dir, ev) -> str:
    """DB-free backend: ColabFold remote MSA + alphaflow AF2 Evoformer representation.
    Shells our runner in the env (alphaflow + openfold + torch); no local AlphaFold DBs."""
    import shlex
    for k in ("weights", "alphaflow_dir", "evoformer_dir"):
        if not ev.get(k):
            raise RuntimeError(f"evoformer.backend=evoformer2 needs `{k}` in kinapse.yaml "
                               "(AF2 weights .npz, alphaflow source dir, evoformer_representation dir).")
    runner = str(Path(__file__).with_name("_evoformer2_runner.py"))
    conda_env = ev.get("conda_env", "openfold_env")
    python_bin = ev.get("python")
    prefix = ([python_bin] if python_bin
              else shlex.split(ev.get("env_run", "conda run -n")) + [conda_env, "python3"])
    cmd = prefix + ["-u", runner,
                    "--pdb_file", str(pdb_path),
                    "--outdir", str(output_dir),
                    "--weights_path", ev["weights"],
                    "--msa_dir", os.path.join(str(output_dir), "msa"),
                    "--alphaflow_dir", ev["alphaflow_dir"],
                    "--evoformer_dir", ev["evoformer_dir"],
                    "--openfold_dir", ev.get("openfold_dir", "")]
    print("🧬 Evoformer (alphaflow, ColabFold MSA — no local DBs):\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return str(output_dir)


def find_embedding_pkl(output_dir) -> Optional[str]:
    """Locate the representation .pkl OpenFold wrote (prefers ``*_output_dict.pkl``)."""
    out = Path(output_dir)
    for pat in ("*_output_dict.pkl", "*.pkl"):
        hits = sorted(out.rglob(pat))
        if hits:
            return str(hits[0])
    return None


def compute_to(pdb_path, output_dir, pkl_out, use_mmseqs2_gpu: bool = False) -> str:
    """Run OpenFold and copy the resulting representation to ``pkl_out``. Returns ``pkl_out``."""
    run(pdb_path, output_dir, use_mmseqs2_gpu=use_mmseqs2_gpu)
    src = find_embedding_pkl(output_dir)
    if src is None:
        raise FileNotFoundError(
            f"OpenFold ran but wrote no .pkl under {output_dir} — check where "
            "run_pretrained_openfold_shortened.py saves its output_dict.")
    if os.path.abspath(src) != os.path.abspath(pkl_out):
        shutil.copyfile(src, pkl_out)
    return pkl_out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute the OpenFold/Evoformer embedding for a PDB.")
    ap.add_argument("--pdb_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fasta_dir", default=None)
    ap.add_argument("--usemmseq2_gpu", action="store_true")
    a = ap.parse_args()
    run(a.pdb_path, a.output_dir, fasta_dir=a.fasta_dir, use_mmseqs2_gpu=a.usemmseq2_gpu)
