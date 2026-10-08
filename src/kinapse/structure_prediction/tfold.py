"""Run **tFold-TCR** (TencentAI4S/tfold) as a completely separate external model.

tFold-TCR predicts a full TCR-pMHC (or TCR-only, or pMHC-only) complex structure
directly from sequence, using an ESM-PPI language-model featuriser + a folding
trunk — no MSA. It pulls heavy deps (torch, deepspeed) and multi-GB weights, so
kinapse **never imports it**; it runs in its own environment behind a JSON
contract, exactly like the STCRpy runner.

Point kinapse at that environment with ``KINAPSE_TFOLD_PYTHON`` (recommended);
otherwise the active interpreter is used if ``tfold`` is importable there; else a
clear install hint is raised. If tfold is cloned but not ``pip install``-ed, also
set ``KINAPSE_TFOLD_REPO`` to the repo root. Registered as the ``tfold_tcr``
runner in the ``structure_prediction`` tier (:mod:`kinapse.structure_prediction`).

Chain ids (tFold-TCR convention): ``B`` = TCR beta, ``A`` = TCR alpha,
``M`` = MHC heavy, ``N`` = beta-2-microglobulin / MHC-II beta (optional),
``P`` = peptide.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HINT = (
    "tFold-TCR not found. It needs its own environment (heavy deps: torch, deepspeed;\n"
    "multi-GB weights auto-download from Zenodo):\n"
    "  git clone https://github.com/TencentAI4S/tfold.git\n"
    "  conda env create -n tfold-tcr -f tfold/environment.yaml && conda activate tfold-tcr\n"
    "  pip install ./tfold\n"
    "Then point kinapse at that env:\n"
    "  export KINAPSE_TFOLD_PYTHON=/path/to/tfold-tcr/bin/python\n"
    "  # if cloned but not pip-installed, also: export KINAPSE_TFOLD_REPO=/path/to/tfold\n"
    "(or install tfold into the active interpreter)."
)
_RUNNER = str(Path(__file__).with_name("_tfold_runner.py"))


def _python() -> Optional[List[str]]:
    """Interpreter to run tFold with: env var -> active interpreter -> None."""
    env_py = os.environ.get("KINAPSE_TFOLD_PYTHON")
    if env_py:
        return [env_py]
    if importlib.util.find_spec("tfold") is not None:
        return [sys.executable]
    return None


def available() -> bool:
    """True if a tFold-TCR environment can be located."""
    return _python() is not None


def predict_many(jobs: List[Dict[str, Any]], *, model_version: str = "Complex",
                 device: Optional[str] = None, chunk_size: Optional[int] = None,
                 seed: int = 42, skip_existing: bool = True,
                 timeout: Optional[float] = None) -> Dict[str, Any]:
    """Predict a batch of complexes in one external process (model loaded once).

    Each job is ``{"name": str, "chains": [{"id", "sequence"}], "out_pdb": str}``.
    Returns ``{"status", "results": [{"name", "out_pdb", "status", "iptm", "ptm",
    "error"}]}``. Never raises on a per-job failure — that job's ``status`` is
    ``"error"`` and the batch continues; raises :class:`ImportError` only if no
    tFold environment can be found at all.
    """
    py = _python()
    if py is None:
        raise ImportError(_HINT)
    if timeout is None:  # generous: model load (~1-2 min) + per-complex inference
        timeout = 600.0 + 180.0 * max(1, len(jobs))
    payload = {"jobs": jobs, "model_version": model_version, "device": device,
               "chunk_size": chunk_size, "seed": seed, "skip_existing": skip_existing}
    env = dict(os.environ)
    try:
        proc = subprocess.run(py + [_RUNNER], input=json.dumps(payload),
                              capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"tFold exceeded {timeout:.0f}s", "results": []}
    if proc.returncode != 0:
        return {"status": "error", "error": (proc.stderr or "").strip()[-800:], "results": []}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error",
                "error": f"tfold runner did not emit JSON: {proc.stdout[:300]!r}",
                "stderr": (proc.stderr or "").strip()[-800:], "results": []}


def predict(chains: List[Dict[str, str]], out_pdb: str, *, name: Optional[str] = None,
            **kwargs) -> Dict[str, Any]:
    """Predict a single complex. Returns that job's result dict (with ``status``)."""
    job = {"name": name or Path(out_pdb).stem, "chains": chains, "out_pdb": str(out_pdb)}
    res = predict_many([job], **kwargs)
    if res.get("results"):
        out = res["results"][0]
        out.setdefault("status", res.get("status", "error"))
        if res.get("status") not in ("ok", None) and out.get("status") == "error" and not out.get("error"):
            out["error"] = res.get("error")
        return out
    return {"status": res.get("status", "error"), "error": res.get("error"),
            "out_pdb": str(out_pdb), "name": job["name"]}


def run_tfold(inputs: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
    """Registry adapter: accept a single ``{chains, out_pdb}`` or a ``{jobs: [...]}``
    batch (plus optional model_version/device/chunk_size/seed/skip_existing)."""
    opts = {k: inputs[k] for k in
            ("model_version", "device", "chunk_size", "seed", "skip_existing")
            if k in inputs}
    if timeout is not None:
        opts["timeout"] = timeout
    if "jobs" in inputs:
        return predict_many(inputs["jobs"], **opts)
    if "chains" in inputs and "out_pdb" in inputs:
        return predict(inputs["chains"], inputs["out_pdb"], name=inputs.get("name"), **opts)
    return {"status": "error", "error": "run_tfold needs either 'jobs' or 'chains'+'out_pdb'"}


__all__ = ["available", "predict", "predict_many", "run_tfold"]
