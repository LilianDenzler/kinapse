#!/usr/bin/env python3
"""Isolated tFold-TCR runner — executed by an *external* interpreter that has
TencentAI4S/tfold installed (never imported into kinapse's environment).

Protocol: reads one JSON object on stdin::

    {"jobs": [{"name": str, "chains": [{"id": "B"|"A"|"M"|"N"|"P",
                                        "sequence": str}], "out_pdb": str}, ...],
     "model_version": "Complex"|"TCR"|"pMHC",   # default "Complex"
     "device": "cuda"|"cuda:0"|"cpu"|null,       # default: cuda if available
     "chunk_size": int|null, "seed": int, "skip_existing": bool}

and writes one JSON object on stdout::

    {"status": "ok", "results": [{"name", "out_pdb", "status": "ok"|"skipped"|"error",
                                  "iptm": float|null, "ptm": float|null, "error": str|null}]}

Chain ids follow tFold-TCR: B = TCR beta, A = TCR alpha, M = MHC heavy,
N = beta-2-microglobulin / MHC-II beta (optional), P = peptide.

Depends only on tfold (+ torch) — nothing from kinapse. Model weights download
themselves from Zenodo on first use into ``torch.hub``'s checkpoints dir.

stdout is reserved for the JSON result; tfold/torch print a lot, so all library
output is redirected to stderr for the duration of the run.
"""
import json
import os
import re
import sys

# Running this script by path puts its own directory on sys.path[0]; drop it so
# our sibling ``tfold.py`` (kinapse's driver) can never shadow the real tfold pkg.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

# Optional: a cloned-but-not-installed tfold repo can be made importable this way.
_REPO = os.environ.get("KINAPSE_TFOLD_REPO")
if _REPO and os.path.isdir(_REPO):
    sys.path.insert(0, _REPO)

_REAL_STDOUT = sys.stdout


def _emit(obj):
    json.dump(obj, _REAL_STDOUT, default=str)
    _REAL_STDOUT.flush()


_REMARK = {
    "iptm": re.compile(r"Predicted ipTM score:\s*([0-9.]+)", re.I),
    "ptm": re.compile(r"Predicted pTM score:\s*([0-9.]+)", re.I),
}


def _parse_scores(pdb_path):
    """Pull ipTM / pTM out of the REMARK 250 header tfold writes."""
    scores = {"iptm": None, "ptm": None}
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith("REMARK"):
                    if line.startswith(("ATOM", "HETATM")):
                        break
                    continue
                for key, rx in _REMARK.items():
                    m = rx.search(line)
                    if m:
                        scores[key] = float(m.group(1))
    except OSError:
        pass
    return scores


def _build_predictor(model_version, device):
    import torch
    from tfold.deploy import TCRPredictor, TCRpMHCPredictor, PeptideMHCPredictor
    from tfold.model.pretrain import (
        esm_ppi_650m_tcr, tfold_tcr_trunk, tfold_tcr_pmhc_trunk, tfold_pmhc_trunk,
    )
    mv = (model_version or "Complex").lower()
    if mv.startswith("tcr") and mv != "tcr-pmhc":
        pred = TCRPredictor.restore_from_module(
            ppi_path=esm_ppi_650m_tcr(), trunk_path=tfold_tcr_trunk())
    elif mv.startswith("pmhc"):
        pred = PeptideMHCPredictor.restore_from_module(
            ppi_path=esm_ppi_650m_tcr(), trunk_path=tfold_pmhc_trunk())
    else:  # Complex = TCR-pMHC
        pred = TCRpMHCPredictor(
            ppi_path=esm_ppi_650m_tcr(), trunk_path=tfold_tcr_pmhc_trunk())
    pred.to(device)
    return pred


def main():
    args = json.loads(sys.stdin.read() or "{}")
    jobs = args.get("jobs") or []
    model_version = args.get("model_version", "Complex")
    chunk_size = args.get("chunk_size")
    seed = int(args.get("seed", 42))
    skip_existing = args.get("skip_existing", True)

    # keep stdout clean from here on
    sys.stdout = sys.stderr

    if not jobs:
        _emit({"status": "error", "error": "no jobs provided"})
        return

    try:
        import torch
        from tfold.utils import setup
    except Exception as e:  # noqa: BLE001
        _emit({"status": "error", "error": f"import tfold/torch failed: {e}"})
        return

    dev = args.get("device")
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    try:
        setup(True, seed=seed)
    except Exception:
        pass

    # Load the model once and reuse it across all jobs.
    try:
        predictor = _build_predictor(model_version, device)
    except Exception as e:  # noqa: BLE001
        _emit({"status": "error", "error": f"model load failed: {type(e).__name__}: {e}"})
        return

    results = []
    for job in jobs:
        name = job.get("name") or os.path.splitext(os.path.basename(job.get("out_pdb", "job")))[0]
        out_pdb = job["out_pdb"]
        rec = {"name": name, "out_pdb": out_pdb, "status": "error", "iptm": None,
               "ptm": None, "error": None}
        if skip_existing and os.path.exists(out_pdb):
            rec.update(status="skipped", **_parse_scores(out_pdb))
            results.append(rec)
            continue
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)
            predictor.infer_pdb(job["chains"], filename=out_pdb, chunk_size=chunk_size)
            rec.update(status="ok", **_parse_scores(out_pdb))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        results.append(rec)

    _emit({"status": "ok", "results": results})


if __name__ == "__main__":
    main()
