"""kinapse.runners — the one engine for pluggable external models.

Every "runner" tier (`sequence_embedding`, `structure_prediction`,
`conformer_generation`, `binding_prediction`, `docking`, `scoring`,
`structure_analysis`) registers its external models here as :class:`RunnerSpec`s
and runs them through one uniform API. This is the generalised form of the pattern ifscore uses for scorers:
declare a spec, run it behind a JSON contract, keep it isolated. The scoring tier
delegates to ifscore directly; other tiers use this engine.

Adding your own model needs no kinapse fork — see ``docs/ADDING_A_MODEL.md``:
1. write a :class:`RunnerSpec` (in a tier's ``registry.py`` or your own package),
2. register it in-tree via :func:`register`, or from your package via a
   ``kinapse.runners`` entry point.
It then appears in :func:`specs` / ``kinapse <tier> --list`` and runs via :func:`run`.

Backends:
- ``native``  — an in-process ``func(inputs: dict) -> dict``. No isolation; for
  light, dependency-free runners (and for tests).
- ``uvenv``   — a JSON-in/JSON-out script executed in its own throwaway env via
  `uv` (`uv run --with <req> python <runner>`), so heavy/conflicting deps never
  touch kinapse. Mirrors ifscore's isolation.
- ``container`` — reserved (declare now, wire later).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

Status = str  # "planned" | "scaffold" | "stable"
Backend = str  # "native" | "uvenv" | "container"


@dataclass(frozen=True)
class RunnerSpec:
    """Declaration of one external model in one tier."""
    name: str
    tier: str
    backend: Backend = "native"
    status: Status = "planned"
    description: str = ""
    outputs: Tuple[str, ...] = ()          # fields/metrics it returns
    requirements: Tuple[str, ...] = ()     # pip requirements for the uvenv backend
    runner: Optional[str] = None           # path to a JSON-in/JSON-out script (uvenv)
    func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None  # native backend
    python: Optional[str] = None           # pin an interpreter for uvenv, e.g. "3.10"
    needs_native_ref: bool = False         # requires a native/reference structure (e.g. DockQ)
    homepage: str = ""
    license: str = "open"
    tags: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.tier}:{self.name}"


# (tier, name) -> RunnerSpec
_REGISTRY: Dict[Tuple[str, str], RunnerSpec] = {}
_DISCOVERED = False


def register(spec: RunnerSpec, *, replace: bool = False) -> RunnerSpec:
    """Register a runner. Set ``replace=True`` to override an existing one."""
    k = (spec.tier, spec.name)
    if k in _REGISTRY and not replace:
        raise ValueError(f"runner {spec.key!r} already registered (use replace=True)")
    _REGISTRY[k] = spec
    return spec


def _discover() -> None:
    """Load third-party runners advertised via the ``kinapse.runners`` entry-point
    group. Each entry point is a zero-arg callable that calls :func:`register`."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group = eps.select(group="kinapse.runners") if hasattr(eps, "select") else eps.get("kinapse.runners", [])
        for ep in group:
            try:
                ep.load()()
            except Exception:  # a broken plugin must not break discovery
                continue
    except Exception:
        pass


def _load_builtin(tier: str) -> None:
    """Import a tier's bundled ``registry.py`` so its ``register()`` calls run.

    Only a *missing* registry is ignored (not every module is a runner tier); a
    real error inside an existing ``registry.py`` is allowed to surface."""
    import importlib
    import importlib.util
    name = f"kinapse.{tier}.registry"
    try:
        if importlib.util.find_spec(name) is None:
            return
    except (ImportError, ValueError):
        return
    importlib.import_module(name)


def specs(tier: Optional[str] = None) -> List[RunnerSpec]:
    """List registered runners, optionally filtered to one tier."""
    _discover()
    if tier:
        _load_builtin(tier)
    return sorted(
        (s for s in _REGISTRY.values() if tier is None or s.tier == tier),
        key=lambda s: s.key,
    )


def get(tier: str, name: str) -> RunnerSpec:
    _discover()
    _load_builtin(tier)
    try:
        return _REGISTRY[(tier, name)]
    except KeyError:
        known = ", ".join(sorted(s.name for s in _REGISTRY.values() if s.tier == tier)) or "(none)"
        raise KeyError(f"unknown {tier} runner {name!r}; known: {known}") from None


def run(tier: str, name: str, inputs: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0) -> Dict[str, Any]:
    """Run a registered model. Returns a dict (always includes a ``status`` key).

    A failing runner returns ``{"status": "error", "error": ...}`` rather than
    raising, so a batch never dies on one bad model (the ifscore property)."""
    inputs = dict(inputs or {})
    spec = get(tier, name)
    if spec.backend == "native":
        if spec.func is None:
            return {"status": "unavailable", "error": f"{spec.key}: native runner has no func"}
        try:
            out = spec.func(inputs)
            out.setdefault("status", "ok")
            return out
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    if spec.backend == "uvenv":
        return _run_uvenv(spec, inputs, timeout)
    return {"status": "unavailable", "error": f"backend {spec.backend!r} not wired yet ({spec.key})"}


def _run_uvenv(spec: RunnerSpec, inputs: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    if spec.runner is None:
        return {"status": "unavailable", "error": f"{spec.key}: uvenv runner has no script"}
    uv = shutil.which("uv")
    if not uv:
        return {"status": "unavailable",
                "error": "uv not found; install from https://github.com/astral-sh/uv to run isolated backends"}
    cmd = [uv, "run"]
    if spec.python:
        cmd += ["--python", spec.python]
    for req in spec.requirements:
        cmd += ["--with", req]
    cmd += [spec.runner]
    try:
        proc = subprocess.run(cmd, input=json.dumps(inputs), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"{spec.key} exceeded {timeout}s"}
    if proc.returncode != 0:
        return {"status": "error", "error": (proc.stderr or "").strip()[-500:]}
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "error": f"runner did not emit JSON: {proc.stdout[:200]!r}"}
    out.setdefault("status", "ok")
    return out


__all__ = ["RunnerSpec", "register", "get", "specs", "run"]
