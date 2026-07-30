"""Dependency-light smoke tests.

These deliberately avoid importing heavy scientific deps (mdtraj, torch, anarci,
...). They check that the package structure, lazy loading and packaged data are
wired correctly — so they pass even in a minimal environment.
"""
import importlib

import pytest


def test_version_and_regions():
    import kinapse
    assert isinstance(kinapse.__version__, str)
    # IMGT region constants are always available (no heavy deps).
    assert "A_CDR3" in kinapse.CDR_FR_RANGES
    assert kinapse.VARIABLE_RANGE == (1, 128)


@pytest.mark.parametrize("mod", [
    "kinapse.structures",
    "kinapse.sequence_embedding",
    "kinapse.geometry",
    "kinapse.dynamics_analysis",
    "kinapse.conformer_generation",
    "kinapse.pipelines",
    "kinapse.config",
    "kinapse.structures.pmhc",
])
def test_subpackages_import_light(mod):
    # Importing a sub-package must not require its optional heavy deps.
    importlib.import_module(mod)


def test_lazy_maps_are_consistent():
    # Every lazily-exported name should resolve to a real submodule attribute
    # target string (we don't import the heavy target here, just the map).
    import kinapse.dynamics_analysis as a
    import kinapse.structures as s
    for name in a.__all__:
        assert name in a._LAZY or name == "load_tcr"
    assert "TCR" in s._LAZY and s._LAZY["TCR"] == (".tcr", "TCR")


def test_pmhc_scaffold_is_light():
    from kinapse.structures.pmhc import PMHC, TCRpMHC, load_pmhc
    assert callable(load_pmhc)
    assert PMHC.__doc__ and TCRpMHC.__doc__


def test_config_paths():
    from kinapse import config
    paths = config.get_paths()
    assert paths.get_output_dir()  # non-empty default
    # Packaged data resources resolve and exist.
    assert config.so3_dir().joinpath("so3_cdf_vals2.npy").exists()
    assert config.consensus_output_dir().exists()
