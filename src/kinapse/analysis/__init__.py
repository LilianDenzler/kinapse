"""Deprecated import path: ``kinapse.analysis`` was renamed to ``kinapse.dynamics_analysis``.

Kept as a thin alias so existing imports keep working. Prefer the new name.
"""
import importlib as _il
import sys as _sys
_target = _il.import_module("kinapse.dynamics_analysis")
__path__ = _target.__path__  # deep submodule imports resolve to the real package
_sys.modules[__name__] = _target
