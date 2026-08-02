"""Tests for the scorer-benchmark analysis logic (synthetic data; no ifscore/real PDBs)."""
import numpy as np
import pandas as pd


def _synthetic():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):                      # positives: model correlated with GT, high BSA
        cid = f"c{i}"
        gt = rng.normal(10, 1.0)
        rows.append({"role": "gt", "id": cid, "group": cid, "label": np.nan, "geo__bsa": gt})
        rows.append({"role": "model_pos", "id": cid, "group": cid, "label": 1,
                     "geo__bsa": gt + rng.normal(0, 0.4)})
    for i in range(20):                      # negatives: lower BSA -> separable
        nid = f"n{i}"
        rows.append({"role": "model_neg", "id": nid, "group": nid, "label": 0,
                     "geo__bsa": rng.normal(6, 1.0)})
    return pd.DataFrame(rows)


def test_discover(tmp_path):
    from kinapse.benchmarks.scorer_benchmark import discover
    gt, md, ng = tmp_path / "gt", tmp_path / "md", tmp_path / "ng"
    for d in (gt, md, ng):
        d.mkdir()
    (gt / "a.pdb").write_text("")
    (gt / "b.pdb").write_text("")
    (md / "a.pdb").write_text("")            # only 'a' has a model -> only 'a' is a positive
    (ng / "x_y_neg.pdb").write_text("")
    pos, neg = discover(gt, md, ng)
    assert [p[0] for p in pos] == ["a"]
    assert [n[0] for n in neg] == ["x_y_neg"]


def test_split_is_leakage_aware():
    from kinapse.benchmarks import split_by_group
    df = split_by_group(_synthetic(), test_frac=0.3, seed=0)
    assert set(df["split"]) <= {"train", "test"}
    for _, sub in df.groupby("group"):
        assert sub["split"].nunique() == 1
    for lab in (0, 1):
        for sp in ("train", "test"):
            assert ((df.label == lab) & (df.split == sp)).any()


def test_agreement_and_discrimination():
    from kinapse.benchmarks import agreement_analysis, discrimination_analysis, split_by_group
    df = split_by_group(_synthetic(), test_frac=0.3, seed=0)

    ag = agreement_analysis(df)
    assert (ag.metric == "geo__bsa").any()
    assert ag.loc[ag.metric == "geo__bsa", "pearson"].iloc[0] > 0.5

    dc = discrimination_analysis(df)
    assert (dc.metric == "geo__bsa").any()
    assert dc.loc[dc.metric == "geo__bsa", "auroc_abs"].iloc[0] > 0.85


import pytest
from pathlib import Path as _Path
_PDB = _Path(__file__).resolve().parent.parent / "examples" / "data" / "example_tcr.pdb"


def test_assign_tier_thresholds():
    from kinapse.benchmarks import assign_tier
    assert assign_tier(1.5, 0.9) == "HQ"
    assert assign_tier(1.5) == "HQ"            # dockq unknown -> iRMSD only
    assert assign_tier(3.0, 0.6) == "MQ"
    assert assign_tier(4.9, 0.3) == "AQ"
    assert assign_tier(7.0, 0.9) == "LQ"
    assert assign_tier(None) is None


def test_tiers_summary_counts():
    from kinapse.benchmarks import tiers_summary
    df = pd.DataFrame([
        {"role": "model_pos", "tier": "HQ"}, {"role": "model_pos", "tier": "MQ"},
        {"role": "model_pos", "tier": "HQ"}, {"role": "model_neg", "tier": None}])
    ts = tiers_summary(df)
    assert int(ts.loc[ts.tier == "HQ", "n"].iloc[0]) == 2
    assert int(ts.loc[ts.tier == "MQ", "n"].iloc[0]) == 1


@pytest.mark.skipif(not _PDB.exists(), reason="example PDB missing")
def test_structural_self_agreement_is_hq():
    import os
    os.environ.setdefault("ANARCI_CPU", "1")
    from kinapse.benchmarks import assign_tier, structural_agreement
    s = structural_agreement(str(_PDB), str(_PDB), legacy_anarci=False)
    assert s.get("struct__cdr_irmsd") is not None
    assert s["struct__cdr_irmsd"] < 0.05            # identical structure -> ~0 Å
    assert assign_tier(s["struct__cdr_irmsd"]) == "HQ"


# ---- alternative loader (STCRpy) plumbing -----------------------------------
def _load_stcrpy_runner():
    """Import the external-only runner without leaking its sys.path mutation."""
    import importlib.util, sys
    saved = list(sys.path)
    path = _Path(__file__).resolve().parent.parent / "src/kinapse/structure_analysis/_stcrpy_runner.py"
    spec = importlib.util.spec_from_file_location("_stcrpy_runner_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def test_stcrpy_fragment_id_parsing():
    r = _load_stcrpy_runner()
    assert r._parse_region("cdra1") == ("A_CDR1", "A")
    assert r._parse_region("cdrb3") == ("B_CDR3", "B")
    assert r._parse_region("fwa2") == ("A_FR2", "A")
    assert r._parse_region("fwb4") == ("B_FR4", "B")
    assert r._parse_region("cdrd1") == ("A_CDR1", "A")   # delta -> alpha side
    assert r._parse_region("cdrg2") == ("B_CDR2", "B")   # gamma -> beta side
    assert r._parse_region("junk") is None
    assert r._parse_region("") is None


def test_run_scorer_benchmark_rejects_bad_loader(tmp_path):
    from kinapse.benchmarks import run_scorer_benchmark
    with pytest.raises(ValueError):
        run_scorer_benchmark("a", "b", "c", out_dir=str(tmp_path), loader="nope")


def test_stcrpy_loader_requires_env(tmp_path, monkeypatch):
    monkeypatch.delenv("KINAPSE_STCRPY_PYTHON", raising=False)
    from kinapse.structure_analysis import stcrpy
    if stcrpy.available():
        pytest.skip("stcrpy is importable in this env")
    from kinapse.benchmarks import run_scorer_benchmark
    with pytest.raises(RuntimeError):                     # clear install hint before numbering
        run_scorer_benchmark("a", "b", "c", out_dir=str(tmp_path), loader="stcrpy")
    with pytest.raises(ImportError):
        stcrpy.stcrpy_chains("nonexistent.pdb")
