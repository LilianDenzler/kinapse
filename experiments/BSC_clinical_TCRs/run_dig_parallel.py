#!/usr/bin/env python3
"""Parallel DiG driver for the BSC clinical TCRs — one TCR per GPU, across all GPUs.

Where the per-TCR time goes: prep (ANARCII numbering + MODELLER linking, CPU, seconds) →
embedding (ColabFold MSA over the network + OpenFold forward, GPU, ~1-2 min) → DiG inference
(200 diffusion samples, GPU, the dominant cost). Running one TCR saturates ~one GPU, so the
biggest speedup is processing GPUS-many TCRs at once, each pinned to its own device.

Each worker is pinned to one GPU via CUDA_VISIBLE_DEVICES (inherited by the openfold_env
subprocesses that do the actual GPU work). The embedding .pkl is checkpointed, so re-runs skip
TCRs already embedded, and finished TCRs are skipped entirely.

Usage:  ANARCI_CPU=1 python run_dig_parallel.py                          # all GPUs, all sampling modes
        DIG_MODE=cdr_mask,weighted_mask python run_dig_parallel.py       # just a subset
        DIG_MODE=vanilla python run_dig_parallel.py                      # a single mode
        N_GPUS=2 WORKERS_PER_GPU=1 python run_dig_parallel.py

DIG_MODE (default "all") picks the sampling mode(s): "all", a single mode, or a comma-separated
subset of {vanilla, no_mask, binary_mask, cdr_mask, weighted_mask}. All requested modes for a TCR
share its one prep + embedding + init_state.
"""
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value
from pathlib import Path

TCR_DIR = "/mnt/larry/lilian/DATA/alex_barcelona_data/tcrs"
OUT_DIR = "/mnt/larry/lilian/DATA/alex_barcelona_data/tcrs_dig_out"
N_SAMPLES = 200
# Which sampling mode(s) to run per TCR, sharing the (expensive) prep + embedding + init_state.
# "all" = every mode; or a comma-separated subset, e.g. DIG_MODE=cdr_mask,weighted_mask;
# or a single mode, e.g. DIG_MODE=vanilla. Modes: vanilla, no_mask, binary_mask, cdr_mask, weighted_mask.
DIG_MODE = os.environ.get("DIG_MODE", "all")

# tuning: 1 worker/GPU is safest (GPU-saturating). Bump WORKERS_PER_GPU to overlap CPU/network
# prep with GPU inference — but watch GPU memory.
def _num_gpus():
    if os.environ.get("N_GPUS"):
        return int(os.environ["N_GPUS"])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return max(1, sum(1 for l in out.stdout.splitlines() if l.strip().startswith("GPU ")))
    except Exception:
        return 1

N_GPUS = _num_gpus()
WORKERS_PER_GPU = int(os.environ.get("WORKERS_PER_GPU", "1"))

_counter = None


def _init(counter, n_gpus):
    global _counter
    with counter.get_lock():
        gpu = counter.value % n_gpus
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)   # this worker (+ its subprocesses) sees only this GPU
    os.environ.setdefault("ANARCI_CPU", "1")


def _expected_modes():
    from kinapse.conformer_generation import resolve_modes
    return resolve_modes(DIG_MODE)


def _done(out_tcr):
    """True only if EVERY requested sampling mode already has frames (idempotent skip)."""
    from kinapse.conformer_generation import MODE_DIRS
    for mode in _expected_modes():
        if not any(Path(out_tcr).glob(f"{MODE_DIRS[mode]}*/*_199.pdb")):
            return False
    return True


def process_one(tcr_file):
    name = tcr_file[:-4]
    out_tcr = os.path.join(OUT_DIR, name)
    os.makedirs(out_tcr, exist_ok=True)
    if _done(out_tcr):
        return (tcr_file, "skip (done)", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # import inside the worker (after CUDA_VISIBLE_DEVICES is set)
    from kinapse.conformer_generation import run_one
    from kinapse.structures import TCR
    from kinapse.structures.io import write_pdb
    try:
        tcr = TCR(input_pdb=os.path.join(TCR_DIR, tcr_file), legacy_anarci=False)
        vs = tcr.pairs[0].variable_structure
        # name the variable-only PDB after the TCR so run_one writes to OUT_DIR/<name>/ (frames
        # <name>_*.pdb) — matching _done's glob and the single-TCR CLI layout.
        var_pdb = os.path.join(out_tcr, f"{name}.pdb")
        write_pdb(var_pdb, vs)
        run_one(var_pdb, OUT_DIR, n_samples=N_SAMPLES, dig_mode=DIG_MODE)
        return (tcr_file, "OK", os.environ.get("CUDA_VISIBLE_DEVICES"))
    except Exception as e:  # noqa: BLE001 - one TCR failing must not kill the batch
        return (tcr_file, f"FAIL: {type(e).__name__}: {str(e)[:200]}", os.environ.get("CUDA_VISIBLE_DEVICES"))


def main():
    files = sorted(f for f in os.listdir(TCR_DIR) if f.endswith(".pdb"))
    workers = N_GPUS * WORKERS_PER_GPU
    print(f"{len(files)} TCRs | {N_GPUS} GPUs x {WORKERS_PER_GPU} = {workers} workers")
    counter = Value("i", 0)
    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(counter, N_GPUS)) as ex:
        for tcr_file, status, gpu in ex.map(process_one, files):
            print(f"[gpu {gpu}] {status:40s} {tcr_file}")
            ok += status == "OK"; fail += status.startswith("FAIL"); skip += status.startswith("skip")
    print(f"\ndone — ok={ok} skipped={skip} failed={fail}")


if __name__ == "__main__":
    main()
