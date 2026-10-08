"""Tests for CD8 co-receptor grafting (`kinapse.structures.cd8`).

The registry/config checks are dependency-light; the actual grafting test needs
Biopython and is skipped where it is unavailable.
"""
import pytest


# --- light: registry + packaged data ---------------------------------------

def test_cd8_templates_dir_resolves():
    from kinapse import config
    d = config.cd8_templates_dir()
    assert d.joinpath("templates.yaml").exists()
    assert d.joinpath("1AKJ.pdb").exists()


def test_load_templates_light():
    # Parsing the manifest needs only PyYAML (a core dep), not Biopython.
    from kinapse.structures.cd8 import load_templates
    tmpls = load_templates()
    assert tmpls, "at least one CD8 template must be bundled"
    default = [t for t in tmpls if t.default]
    assert default, "one template must be flagged default"
    t = default[0]
    assert t.mhc_class == "I"                 # CD8 binds class I
    assert t.cd8_chains and t.mhc_chain
    from pathlib import Path
    assert Path(t.ref_pdb).exists()


def test_cd8_lazy_exports():
    import kinapse.structures as s
    for name in ("add_cd8", "select_cd8_template", "CD8Template", "CD8Result"):
        assert name in s._LAZY


# --- heavier: end-to-end graft (needs Biopython) ---------------------------

@pytest.fixture
def pmhc_only_pdb(tmp_path):
    """1AKJ with its CD8 (chains D,E) stripped → a bare class-I pMHC target."""
    pytest.importorskip("Bio")
    from kinapse import config
    from kinapse.structures.io import load_pdb, write_pdb
    from kinapse.structures.ops import copy_subset

    ref = config.cd8_templates_dir() / "1AKJ.pdb"
    struct = load_pdb(str(ref), "src")
    pmhc = copy_subset(struct, lambda cid, res, atom: cid in ("A", "B", "C"))
    out = tmp_path / "pmhc_only.pdb"
    write_pdb(str(out), pmhc)
    return out


def test_add_cd8_self_consistency(pmhc_only_pdb, tmp_path):
    # Re-adding CD8 to the very structure it came from must reproduce it exactly:
    # ~0 Å fit, 100% MHC identity, default chain ids S/T, no chains lost.
    from kinapse.structures import cd8
    from kinapse.structures.io import load_pdb

    out = tmp_path / "with_cd8.pdb"
    res = cd8.add_cd8(str(pmhc_only_pdb), str(out))

    assert res.template == "1AKJ"
    assert res.target_mhc_chain == "A"
    assert res.rmsd < 1e-3
    assert res.mhc_identity > 0.99
    assert list(res.cd8_chains) == ["S", "T"]

    chains = {c.id for c in next(load_pdb(str(out), "m").get_models())}
    assert {"A", "B", "C", "S", "T"} <= chains


def test_add_cd8_numbering_agnostic(pmhc_only_pdb, tmp_path):
    # Break resseq alignment; sequence-alignment-based Cα matching must still fit.
    from kinapse.structures import cd8
    from kinapse.structures.io import load_pdb, write_pdb

    struct = load_pdb(str(pmhc_only_pdb), "t")
    for r in list(next(struct.get_models())["A"]):
        het, num, ic = r.id
        r.id = (het, num + 5000, ic)
    shifted = tmp_path / "shifted.pdb"
    write_pdb(str(shifted), struct)

    res = cd8.add_cd8(str(shifted), str(tmp_path / "out.pdb"))
    assert res.rmsd < 1e-3 and res.n_matched > 200


def test_add_cd8_chain_id_collision(pmhc_only_pdb, tmp_path):
    # Occupy the default CD8 ids S/T — the graft must bump to free ids.
    from kinapse.structures import cd8
    from kinapse.structures.io import load_pdb, write_pdb

    struct = load_pdb(str(pmhc_only_pdb), "t")
    model = next(struct.get_models())
    model["B"].id = "S"
    model["C"].id = "T"
    collide = tmp_path / "collide.pdb"
    write_pdb(str(collide), struct)

    res = cd8.add_cd8(str(collide), str(tmp_path / "out.pdb"))
    assert "S" not in res.cd8_chains and "T" not in res.cd8_chains
    chains = {c.id for c in next(load_pdb(str(tmp_path / "out.pdb"), "m").get_models())}
    assert set(res.cd8_chains) <= chains


def test_add_cd8_rejects_non_class_i(tmp_path):
    # A TCR-only PDB has no MHC heavy chain → a clear, actionable error.
    import os
    pytest.importorskip("Bio")
    from kinapse.structures import cd8

    tcr = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "example_tcr.pdb")
    if not os.path.exists(tcr):
        pytest.skip("example_tcr.pdb not available")
    with pytest.raises(Exception) as exc:
        cd8.add_cd8(tcr, str(tmp_path / "out.pdb"))
    assert "MHC" in str(exc.value) or "class I" in str(exc.value)
