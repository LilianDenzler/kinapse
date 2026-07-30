"""Deprecated import path: ``kinapse.embedding`` was renamed to ``kinapse.sequence_embedding``.

Kept as a thin alias so existing imports keep working. Prefer the new name.
"""
import importlib as _il
import sys as _sys
_target = _il.import_module("kinapse.sequence_embedding")
__path__ = _target.__path__  # deep submodule imports resolve to the real package
_sys.modules[__name__] = _target
