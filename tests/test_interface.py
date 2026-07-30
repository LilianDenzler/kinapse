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
