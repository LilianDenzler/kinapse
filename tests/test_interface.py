"""Tests for structures.interface (native interface characterization)."""
from pathlib import Path

import pytest

PDB = Path(__file__).resolve().parent.parent / "examples" / "data" / "example_tcr.pdb"


def test_interface_api_imports():
    from kinapse.structures import analyze_interface, InterfaceResult
    assert callable(analyze_interface) and isinstance(InterfaceResult, type)


def test_optional_backend_raises_with_hint():
    from kinapse.structures.interface import analyze_interface
    with pytest.raises(NotImplementedError) as ei:
        analyze_interface(None, ["A"], ["B"], method="pisa")
    assert "pisa" in str(ei.value).lower()


@pytest.mark.skipif(not PDB.exists(), reason="example PDB not present")
def test_native_interface_on_example():
    from Bio.PDB import PDBParser
    from kinapse.structures import analyze_interface

    model = PDBParser(QUIET=True).get_structure("x", str(PDB))[0]
    res = analyze_interface(model, ["A"], ["B"], cutoff=5.0)
    assert res.method == "native"
    assert res.n_atom_contacts > 0
    assert res.n_interface_residues > 0
    assert "A" in res.interface_residues and "B" in res.interface_residues
    assert res.bsa is None or res.bsa > 0


def test_stcrpy_backend_needs_stcrpy_env():
    import importlib.util
    import os
    if importlib.util.find_spec("stcrpy") is not None or os.environ.get("KINAPSE_STCRPY_PYTHON"):
        pytest.skip("stcrpy available; the missing-env path is not exercised")
    from kinapse.structures.interface import analyze_interface
    with pytest.raises(ImportError) as ei:
        analyze_interface(None, ["A"], ["B"], method="stcrpy", pdb_path=str(PDB))
    msg = str(ei.value).lower()
    assert "stcrpy" in msg and "kinapse_stcrpy_python" in msg
