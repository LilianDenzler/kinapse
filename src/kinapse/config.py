#!/usr/bin/env python3
"""Centralised configuration & path resolution for kinapse.

Replaces the scattered hardcoded ``/mnt/larry`` / ``/mnt/dave`` paths that were
sprinkled through the original ``TCR_Metrics`` pipelines. Values resolve in this
order (highest priority first):

1. Environment variables (``KINAPSE_*``, with legacy ``TCR_DYNAMICS_*`` fallback)
2. A YAML config file (see :func:`PathConfig`)
3. Built-in defaults

Ported from the parent project's ``path_config.py`` and extended with helpers
that locate the data resources shipped *inside* the installed package
(consensus structures, SO(3) diffusion tables).

Quick start::

    from kinapse.config import get_paths, consensus_output_dir
    paths = get_paths()
    out = paths.get_output_dir("benchmarks")
    ref = consensus_output_dir()          # packaged geometry reference data
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:  # PyYAML is a core dependency but guard anyway so import never hard-fails
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# Env-var pairs: (preferred KINAPSE name, legacy TCR_DYNAMICS name)
_ENV = {
    "root": ("KINAPSE_ROOT", "TCR_DYNAMICS_ROOT"),
    "output": ("KINAPSE_OUTPUT", "TCR_DYNAMICS_OUTPUT"),
    "datasets": ("KINAPSE_DATASETS", "TCR_DYNAMICS_DATASETS"),
    "config": ("KINAPSE_CONFIG", "TCR_DYNAMICS_CONFIG"),
}


def _env(key: str) -> Optional[str]:
    """Return the first set env var for a logical key (preferred then legacy)."""
    for name in _ENV.get(key, ()):  # type: ignore[arg-type]
        val = os.environ.get(name)
        if val:
            return val
    return None


class PathConfig:
    """Environment-aware path configuration.

    Args:
        config_file: explicit path to a YAML config. If ``None``, standard
            locations are searched (see :meth:`_load_config`).
    """

    def __init__(self, config_file: Optional[str] = None):
        self.project_root = self._get_project_root()
        self.config = self._load_config(config_file)

    # -- discovery ---------------------------------------------------------
    def _get_project_root(self) -> Path:
        env_root = _env("root")
        if env_root:
            return Path(env_root)
        # Default: the current working directory (where you launch a run).
        return Path.cwd()

    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        if config_file is None:
            config_file = _env("config")
        if config_file is None:
            search_paths = [
                self.project_root / "kinapse.yaml",
                self.project_root / "config.yaml",
                Path.home() / ".kinapse" / "config.yaml",
                Path("/etc/kinapse/config.yaml"),
            ]
            for path in search_paths:
                if path.exists():
                    config_file = str(path)
                    break
        if config_file and Path(config_file).exists() and yaml is not None:
            with open(config_file, "r") as f:
                return self._resolve_vars(yaml.safe_load(f) or {})
        return self._default_config()

    def _resolve_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ``${project_root}`` and ``${ENV_VAR}`` references in strings."""
        import re

        def resolve(value):
            if isinstance(value, str):
                value = value.replace("${project_root}", str(self.project_root))
                for m in re.finditer(r"\$\{(\w+)\}", value):
                    value = value.replace(m.group(0), os.environ.get(m.group(1), ""))
                return value
            if isinstance(value, dict):
                return {k: resolve(v) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve(v) for v in value]
            return value

        return resolve(config)

    def _default_config(self) -> Dict[str, Any]:
        root = self.project_root
        return {
            "project_root": str(root),
            "data": {
                "datasets": str(root / "data" / "datasets"),
                "output": str(root / "data" / "output"),
                "trajectories": str(root / "data" / "trajectories"),
            },
            "generation": {
                # External DiG inference scripts / checkpoints. Point these at
                # your checkout via config.yaml or KINAPSE_* env vars.
                "run_inference": str(root / "protein" / "run_inference.py"),
                "run_inference_addnoise": str(root / "protein" / "run_inference_addnoise.py"),
                "get_init_state": str(root / "protein" / "full_pipeline" / "get_init_state.py"),
                "openfold_wrapper": str(root / "protein" / "full_pipeline"
                                        / "evoformer_representation"
                                        / "openfold_wrapper_for_evoformer.py"),
                # `checkpoint` is the legacy default; `main_model` is the checkpoint the
                # DiG runner actually loads — override it with KINAPSE_CHECKPOINT_MAIN_MODEL
                # (env) or generation.main_model (yaml) to run your newest model.
                "checkpoint": str(root / "protein" / "checkpoints" / "checkpoint-520k.pth"),
                "main_model": str(root / "protein" / "checkpoints" / "checkpoint-520k.pth"),
            },
        }

    # -- access ------------------------------------------------------------
    def get(self, *keys, default=None) -> Any:
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def get_output_dir(self, subdir: Optional[str] = None) -> str:
        base = _env("output") or self.get("data", "output")
        return os.path.join(base, subdir) if subdir else base

    def get_dataset_dir(self) -> str:
        return _env("datasets") or self.get("data", "datasets")

    def get_pipeline_script(self, name: str) -> Optional[str]:
        # Back-compat: original code read scripts under a 'pipeline' key.
        return self.get("generation", name) or self.get("pipeline", name)

    def get_checkpoint(self, name: str = "checkpoint") -> Optional[str]:
        env_var = f"KINAPSE_CHECKPOINT_{name.upper()}"
        return os.environ.get(env_var) or self.get("generation", name)

    @staticmethod
    def ensure_dir(path: str) -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        return path


# Global lazy singleton -----------------------------------------------------
_global_config: Optional[PathConfig] = None


def get_paths() -> PathConfig:
    """Return a process-wide :class:`PathConfig` singleton."""
    global _global_config
    if _global_config is None:
        _global_config = PathConfig()
    return _global_config


def get_project_root() -> str:
    return str(get_paths().project_root)


def get_output_dir(subdir: Optional[str] = None) -> str:
    return get_paths().get_output_dir(subdir)


def get_pipeline_script(name: str) -> Optional[str]:
    return get_paths().get_pipeline_script(name)


# Packaged data resources ---------------------------------------------------
def package_data_dir() -> Path:
    """Absolute path to the installed ``kinapse/data`` directory."""
    from importlib.resources import files
    return Path(str(files("kinapse.data")))


def so3_dir() -> Path:
    """Directory holding the SO(3) diffusion lookup tables used by generation."""
    from importlib.resources import files
    return Path(str(files("kinapse.data").joinpath("so3")))


def consensus_output_dir() -> Path:
    """Directory holding the TCR geometry consensus reference structures."""
    from importlib.resources import files
    return Path(str(files("kinapse.geometry.data").joinpath("consensus_output")))
