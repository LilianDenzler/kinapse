"""Smoke tests for kinapse.scoring (the ifscore integration).

Dependency-light: they do NOT require ifscore to be installed. They check that
the module imports, exposes its API, and gives a clear, actionable error when
ifscore is missing.
"""
import importlib.util

import pytest


def test_scoring_module_imports_without_ifscore():
    import kinapse.scoring as sc
    for name in ("score", "score_batch", "score_tcr_pmhc",
                 "infer_tcr_pmhc_chains", "available_scorers"):
        assert callable(getattr(sc, name)), name


def test_score_raises_clear_error_when_ifscore_absent():
    if importlib.util.find_spec("ifscore") is not None:
        pytest.skip("ifscore is installed; the missing-dependency path is not exercised")
    import kinapse.scoring as sc
    with pytest.raises(ImportError) as ei:
        sc.score("model.pdb", rec=["A"], lig=["B"])
    msg = str(ei.value).lower()
    assert "ifscore" in msg and "pip install" in msg


def test_chain_list_normalisation_helper():
    from kinapse.scoring import _as_list
    assert _as_list("D,E") == ["D", "E"]
    assert _as_list(["A", "B"]) == ["A", "B"]
    assert _as_list(None) is None
    assert _as_list("A, ,C") == ["A", "C"]
