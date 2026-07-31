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
