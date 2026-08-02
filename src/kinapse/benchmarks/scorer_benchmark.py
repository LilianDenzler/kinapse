"""Benchmark all interface scorers on ground-truth vs modelled vs negative complexes.

Given three directories of TCR-pMHC complex PDBs:
  * ``gt_dir``    — ground-truth (real) complexes, e.g. crystal structures minimised.
  * ``model_dir`` — modelled versions of those complexes (filenames match ``gt_dir``).
  * ``neg_dir``   — negative (artificial, non-existent) modelled complexes.

it loads each complex to derive correct receptor/ligand chains (TCR = receptor,
pMHC = ligand, via :func:`kinapse.scoring.infer_tcr_pmhc_chains`), scores everything
with :mod:`kinapse.scoring` (ifscore), makes a leakage-aware (by complex id)
train/test split, and answers, per scorer:

  1. **Agreement** — do the modelled complexes score like their ground truth?
     (Pearson/Spearman of GT-value vs model-value across complexes; mean |Δ|.)
  2. **Discrimination** — do modelled real (positive) complexes score differently
     from modelled negatives? (AUROC / AUPRC / Cohen's d on the TEST split.)

Reference-based scorers (e.g. DockQ) only apply to model-vs-GT; reference-free
scorers (geometry_scoring, prodigy, energy) drive the pos-vs-neg discrimination.
Metric roles are inferred from the data (a metric is "reference-free" if it has
values on the negatives). A scorer that is not provisioned yields NaN and is
skipped, never breaking the run (ifscore's design).

Run as a script::

    python -m kinapse.benchmarks.scorer_benchmark \
        --gt   /mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_openmm_minimised \
        --model /mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_tfold_openmm_minimised \
        --neg  /mnt/larry/lilian/DATA/TCR3d_datasets/negative_TCR_complexes_tfold/openmm_minimised \
        --out  scorer_benchmark_out --scorers all -j 16
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# metric columns from ifscore look like "<scorer>__<metric>"; these suffixes are not metrics
_NON_METRIC_SUFFIX = ("__status", "__runtime_s", "__error")


# ----------------------------------------------------------------- discovery
def _stem(p) -> str:
    return Path(p).stem


def discover(gt_dir, model_dir, neg_dir, limit: Optional[int] = None):
    """Return (positives, negatives).

    positives: list of (id, gt_path, model_path) for stems present in both dirs.
    negatives: list of (id, neg_path).
    """
    gt = {_stem(p): p for p in sorted(Path(gt_dir).glob("*.pdb"))}
    mod = {_stem(p): p for p in sorted(Path(model_dir).glob("*.pdb"))}
    shared = sorted(set(gt) & set(mod))
    positives = [(s, str(gt[s]), str(mod[s])) for s in shared]
    negatives = [(_stem(p), str(p)) for p in sorted(Path(neg_dir).glob("*.pdb"))]
    if limit:
        positives = positives[:limit]
        negatives = negatives[:limit]
    return positives, negatives


# ----------------------------------------------------------------- chains
def _resolve_one(path: str, legacy_anarci: bool, loader: str) -> Tuple[str, list]:
    """Worker: (receptor, ligand) chains for one PDB via the chosen loader.
    Returns (path, [rec, lig]) or (path, ['__error__', msg]). Module-level so it
    is picklable for the process pool (spawn)."""
    try:
        if loader == "stcrpy":
            from kinapse.structure_analysis.stcrpy import stcrpy_chains
            rec, lig = stcrpy_chains(path)
        else:
            from kinapse.scoring import infer_tcr_pmhc_chains
            rec, lig = infer_tcr_pmhc_chains(path, legacy_anarci=legacy_anarci)
        return path, [rec, lig]
    except Exception as e:  # noqa: BLE001
        return path, ["__error__", str(e)]


def resolve_chains(paths: List[str], legacy_anarci: bool = False,
                   cache_path: Optional[str] = None, n_jobs: int = 1,
                   loader: str = "native") -> Dict[str, Tuple[List[str], List[str]]]:
    """Map each PDB path -> (receptor_chains, ligand_chains). Cached to JSON; a
    file that fails to parse/number is skipped. ``loader`` selects the backend:
    ``native`` (kinapse loader) or ``stcrpy`` (external OPIG STCRpy env). With
    ``n_jobs > 1`` the numbering runs in a process pool (numbering is the bottleneck)."""
    cache: Dict[str, list] = {}
    cpath = Path(cache_path) if cache_path else None
    if cpath and cpath.exists():
        cache = json.loads(cpath.read_text())

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        def tqdm(x, **k):
            return x

    # retry cached errors; dedupe so the pool doesn't score the same file twice
    todo = [p for p in dict.fromkeys(paths)
            if p not in cache
            or (isinstance(cache.get(p), list) and cache[p][:1] == ['__error__'])]

    def _flush(force=False):
        if cpath and (force or len(cache) % 25 == 0):
            cpath.write_text(json.dumps(cache))

    if n_jobs and n_jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")   # avoid fork+threaded-lib (ANARCI/BLAS) deadlocks
        done = 0
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            futs = [ex.submit(_resolve_one, p, legacy_anarci, loader) for p in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="chains", unit="pdb"):
                p, rec = fut.result()
                cache[p] = rec
                done += 1
                if done % 25 == 0:
                    _flush(force=True)
    else:
        for p in tqdm(todo, desc="chains", unit="pdb"):
            _, rec = _resolve_one(p, legacy_anarci, loader)
            cache[p] = rec
            _flush()
    _flush(force=True)

    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for p in paths:
        v = cache.get(p)
        if v and v[0] != "__error__":
            out[p] = (v[0], v[1])
    return out


# ----------------------------------------------------------------- manifest + scoring
def build_manifest(positives, negatives, chains) -> pd.DataFrame:
    """One row per scored structure: role/id + ifscore manifest columns.

    role ∈ {gt, model_pos, model_neg}. model_pos rows carry native=matching GT so
    reference-based scorers (DockQ) run against the ground truth.
    """
    rows = []
    for cid, gt_path, model_path in positives:
        if gt_path in chains:
            rec, lig = chains[gt_path]
            rows.append(dict(role="gt", id=cid, group=cid, label=np.nan,
                             model=gt_path, native="",
                             receptor_chains=",".join(rec), ligand_chains=",".join(lig)))
        if model_path in chains:
            rec, lig = chains[model_path]
            native = gt_path if gt_path in chains else ""
            rows.append(dict(role="model_pos", id=cid, group=cid, label=1,
                             model=model_path, native=native,
                             receptor_chains=",".join(rec), ligand_chains=",".join(lig)))
    for cid, neg_path in negatives:
        if neg_path in chains:
            rec, lig = chains[neg_path]
            rows.append(dict(role="model_neg", id=cid, group=cid, label=0,
                             model=neg_path, native="",
                             receptor_chains=",".join(rec), ligand_chains=",".join(lig)))
    return pd.DataFrame(rows)


def run_scoring(manifest: pd.DataFrame, scorers="all", n_jobs: int = 8,
                timeout: float = 900.0, out_dir: Optional[str] = None) -> pd.DataFrame:
    """Score every manifest row with ifscore (one batch) and merge back the
    role/id/label/group metadata (joined on the model path)."""
    from kinapse import scoring

    if manifest.empty:
        raise RuntimeError("no scorable complexes — chain resolution failed for all inputs "
                           "(are these TCR-pMHC complexes? see chains_cache.json).")
    cols = ["model", "native", "receptor_chains", "ligand_chains"]
    man_path = Path(out_dir or ".") / "_ifscore_manifest.csv"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    manifest[cols].to_csv(man_path, index=False)

    scored = scoring.score_batch(manifest=str(man_path), scorers=scorers,
                                 n_jobs=n_jobs, timeout=timeout)
    meta = manifest[["model", "role", "id", "group", "label"]].assign(
        _key=manifest["model"].astype(str))
    scored = scored.assign(_key=scored["model"].astype(str))
    # keep only scorer outputs ("<scorer>__<metric>") from ifscore — everything else
    # (id/model/native/chains) already lives in `meta` and would collide on merge.
    metric_cols = [c for c in scored.columns if "__" in c]
    merged = meta.merge(scored[["_key"] + metric_cols], on="_key", how="left").drop(columns="_key")
    return merged


# ----------------------------------------------------------------- split
def split_by_group(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Assign each row a 'split' (train/test), grouping by complex id so a
    complex's gt+model never straddle the split; stratified by label."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    # each group's label (positives -> 1, negatives -> 0; gt rows carry NaN but
    # share their positive's group, so the whole group moves together)
    grp_label = df.groupby("group")["label"].apply(
        lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    test_groups = set()
    for lab in (0, 1):
        groups = sorted(grp_label[grp_label == lab].index)
        rng.shuffle(groups)
        n_test = max(1, int(round(len(groups) * test_frac)))
        test_groups |= set(groups[:n_test])
    df["split"] = np.where(df["group"].isin(test_groups), "test", "train")
    return df


# ----------------------------------------------------------------- analyses
def _metric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if "__" in c and not c.endswith(_NON_METRIC_SUFFIX)]


def agreement_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per metric: correlation of GT value vs modelled-positive value across complexes."""
    from scipy.stats import pearsonr, spearmanr

    gt = df[df.role == "gt"].set_index("id")
    mp = df[df.role == "model_pos"].set_index("id")
    ids = gt.index.intersection(mp.index)
    rows = []
    for col in _metric_columns(df):
        if col not in gt.columns or col not in mp.columns:
            continue
        # .astype(float): pandas keeps bool/int columns as-is, but scipy's pearsonr
        # rejects non-inexact dtypes (bool) — force float.
        x = pd.to_numeric(gt.loc[ids, col], errors="coerce").astype(float)
        y = pd.to_numeric(mp.loc[ids, col], errors="coerce").astype(float)
        m = x.notna() & y.notna()
        n = int(m.sum())
        if n < 3:
            continue
        if x[m].nunique() > 1 and y[m].nunique() > 1:   # correlation undefined for constant input
            pear = float(pearsonr(x[m], y[m])[0])
            spear = float(spearmanr(x[m], y[m])[0])
        else:
            pear = spear = float("nan")
        rows.append(dict(metric=col, n=n, pearson=round(pear, 3) if not math.isnan(pear) else np.nan,
                         spearman=round(spear, 3) if not math.isnan(spear) else np.nan,
                         mean_abs_diff=round(float((y[m] - x[m]).abs().mean()), 4),
                         gt_mean=round(float(x[m].mean()), 4),
                         model_mean=round(float(y[m].mean()), 4)))
    return pd.DataFrame(rows).sort_values("pearson", ascending=False, ignore_index=True) if rows else pd.DataFrame()


def discrimination_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per reference-free metric: separate modelled positives (1) from negatives (0)
    on the TEST split (AUROC/AUPRC/Cohen's d); threshold fit on TRAIN → test accuracy."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    def cohens_d(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return np.nan
        sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
        return (a.mean() - b.mean()) / sp if sp > 0 else np.nan

    labelled = df[df.label.isin([0, 1])]
    rows = []
    for col in _metric_columns(df):
        sub = labelled[["label", "split", col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce").astype(float)  # bool -> float for sklearn/arith
        sub = sub.dropna()
        # reference-free only: needs both classes present (negatives have this metric)
        if not {0, 1}.issubset(set(sub.label.unique())):
            continue
        tr, te = sub[sub.split == "train"], sub[sub.split == "test"]
        if te.label.nunique() < 2 or len(te) < 6:
            continue
        auc = roc_auc_score(te.label, te[col])
        auc_abs = max(auc, 1 - auc)                      # direction-agnostic power
        ap = average_precision_score(te.label, te[col] if auc >= 0.5 else -te[col])
        d = cohens_d(te.loc[te.label == 1, col], te.loc[te.label == 0, col])
        # Youden-J threshold on train, accuracy on test (direction from AUC)
        acc = np.nan
        if len(tr) >= 6 and tr.label.nunique() == 2:
            sign = 1.0 if roc_auc_score(tr.label, tr[col]) >= 0.5 else -1.0
            vals = sign * tr[col].values
            thr_grid = np.unique(vals)
            best_j, best_thr = -1, thr_grid[0]
            for t in thr_grid:
                pred = (vals >= t).astype(int)
                tp = ((pred == 1) & (tr.label.values == 1)).sum()
                fn = ((pred == 0) & (tr.label.values == 1)).sum()
                tn = ((pred == 0) & (tr.label.values == 0)).sum()
                fp = ((pred == 1) & (tr.label.values == 0)).sum()
                tpr = tp / (tp + fn) if (tp + fn) else 0
                fpr = fp / (fp + tn) if (fp + tn) else 0
                if tpr - fpr > best_j:
                    best_j, best_thr = tpr - fpr, t
            te_pred = ((sign * te[col].values) >= best_thr).astype(int)
            acc = float((te_pred == te.label.values).mean())
        rows.append(dict(metric=col, auroc=round(float(auc), 3), auroc_abs=round(float(auc_abs), 3),
                         auprc=round(float(ap), 3), cohens_d=round(float(d), 3) if not np.isnan(d) else np.nan,
                         test_accuracy=round(acc, 3) if not np.isnan(acc) else np.nan,
                         n_train=int(len(tr)), n_test=int(len(te))))
    return pd.DataFrame(rows).sort_values("auroc_abs", ascending=False, ignore_index=True) if rows else pd.DataFrame()


# ----------------------------------------------------------------- orchestration
def add_structural_agreement(scored, positives, legacy_anarci: bool = False,
                             loader: str = "native"):
    """Add model-vs-GT Cα RMSDs (per-CDR, local, Cα iRMSD over the 6 CDRs) and an
    HQ/MQ/AQ/LQ tier to the modelled-positive rows. ``loader`` picks the backend
    used to identify the CDRs: ``native`` (kinapse) or ``stcrpy`` (external)."""
    from .structural import assign_tier, structural_agreement
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        def tqdm(x, **k):
            return x

    dq = {}
    if "dockq__dockq" in scored.columns:
        mp = scored[scored.role == "model_pos"]
        dq = dict(zip(mp["id"], pd.to_numeric(mp["dockq__dockq"], errors="coerce")))

    rows = []
    for cid, gt_path, model_path in tqdm(positives, desc="structural", unit="pair"):
        try:
            s = structural_agreement(model_path, gt_path, legacy_anarci=legacy_anarci,
                                     loader=loader)
        except Exception:  # noqa: BLE001
            s = {}
        if not s:
            continue
        s = dict(s)
        s["id"] = cid
        s["tier"] = assign_tier(s.get("struct__cdr_irmsd"), dq.get(cid))
        rows.append(s)
    if not rows:
        return scored

    sdf = pd.DataFrame(rows)
    struct_cols = [c for c in sdf.columns if c != "id"]
    scored = scored.merge(sdf, on="id", how="left")
    scored.loc[scored.role != "model_pos", struct_cols] = np.nan   # model-vs-GT only
    return scored


def tiers_summary(scored) -> pd.DataFrame:
    """HQ/MQ/AQ/LQ distribution among modelled positives (the validated pairs)."""
    if "tier" not in scored.columns:
        return pd.DataFrame()
    mp = scored[scored.role == "model_pos"]
    total = int(len(mp))
    vc = mp["tier"].value_counts(dropna=True)
    rows = [dict(tier=x, n=int(vc.get(x, 0)),
                 fraction=round(int(vc.get(x, 0)) / total, 3) if total else 0.0)
            for x in ("HQ", "MQ", "AQ", "LQ")]
    n_un = int(mp["tier"].isna().sum())
    if n_un:
        rows.append(dict(tier="untiered", n=n_un, fraction=round(n_un / total, 3) if total else 0.0))
    return pd.DataFrame(rows)


def run_scorer_benchmark(gt_dir, model_dir, neg_dir, out_dir="scorer_benchmark_out",
                         scorers="all", test_frac: float = 0.3, seed: int = 0,
                         legacy_anarci: bool = False, limit: Optional[int] = None,
                         n_jobs: int = 8, timeout: float = 900.0,
                         structural: bool = True, loader: str = "native") -> Dict[str, pd.DataFrame]:
    if loader not in ("native", "stcrpy"):
        raise ValueError(f"loader must be 'native' or 'stcrpy', got {loader!r}")
    if loader == "stcrpy":
        from kinapse.structure_analysis import stcrpy as _stcrpy
        if not _stcrpy.available():
            raise RuntimeError(_stcrpy._HINT)   # clear install hint before we start numbering
        print("loader: STCRpy (external) — chain ID + CDR annotation")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    positives, negatives = discover(gt_dir, model_dir, neg_dir, limit=limit)
    print(f"discovered {len(positives)} matched positives, {len(negatives)} negatives")

    all_paths = [p for _, g, m in positives for p in (g, m)] + [p for _, p in negatives]
    chains = resolve_chains(all_paths, legacy_anarci=legacy_anarci, n_jobs=n_jobs, loader=loader,
                            cache_path=str(out / "chains_cache.json"))
    print(f"resolved chains for {len(chains)}/{len(set(all_paths))} structures")

    manifest = build_manifest(positives, negatives, chains)
    scored = run_scoring(manifest, scorers=scorers, n_jobs=n_jobs, timeout=timeout, out_dir=str(out))
    scored = split_by_group(scored, test_frac=test_frac, seed=seed)

    tiers = pd.DataFrame()
    if structural:
        scored = add_structural_agreement(scored, positives, legacy_anarci=legacy_anarci,
                                          loader=loader)
        tiers = tiers_summary(scored)
        tiers.to_csv(out / "tiers.csv", index=False)
        sc = [c for c in scored.columns if c.startswith("struct__")]
        sc += ["tier"] if "tier" in scored.columns else []
        if sc:
            scored[scored.role == "model_pos"][["id"] + sc].to_csv(out / "structural.csv", index=False)
    scored.to_csv(out / "scores.csv", index=False)

    agree = agreement_analysis(scored)
    disc = discrimination_analysis(scored)
    agree.to_csv(out / "agreement.csv", index=False)
    disc.to_csv(out / "discrimination.csv", index=False)

    print("\n=== AGREEMENT (GT vs modelled positive) ===")
    print(agree.to_string(index=False) if len(agree) else "  (no reference-free metrics with data)")
    print("\n=== DISCRIMINATION (modelled positive vs negative, TEST split) ===")
    print(disc.to_string(index=False) if len(disc) else "  (no reference-free metrics with data)")
    result = {"scores": scored, "agreement": agree, "discrimination": disc}
    if structural:
        print("\n=== STRUCTURAL TIERS (modelled positive vs GT: Cα iRMSD over 6 CDRs + DockQ) ===")
        print(tiers.to_string(index=False) if len(tiers) else "  (no tiers — structural metrics unavailable)")
        result["tiers"] = tiers
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark interface scorers: GT vs modelled vs negatives.")
    ap.add_argument("--gt", required=True, help="ground-truth complex PDB directory")
    ap.add_argument("--model", required=True, help="modelled complex PDB directory (filenames match GT)")
    ap.add_argument("--neg", required=True, help="negative (artificial) modelled complex PDB directory")
    ap.add_argument("--out", default="scorer_benchmark_out", help="output directory")
    ap.add_argument("--scorers", default="all", help="ifscore scorer set (e.g. all, default, geometry_scoring)")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--new-anarci", action="store_true", help="use ANARCII (pip) for chains (default)")
    ap.add_argument("--legacy-anarci", action="store_true", help="use legacy ANARCI (bioconda) for chains")
    ap.add_argument("--limit", type=int, default=None, help="cap positives/negatives (for a quick run)")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--loader", choices=("native", "stcrpy"), default="native",
                    help="backend for chain ID + CDR annotation: native (kinapse) or "
                         "stcrpy (external OPIG env; set KINAPSE_STCRPY_PYTHON)")
    ap.add_argument("--no-structural", action="store_true", help="skip model-vs-GT Cα-iRMSD/tiers")
    args = ap.parse_args(argv)
    run_scorer_benchmark(args.gt, args.model, args.neg, out_dir=args.out, scorers=args.scorers,
                         test_frac=args.test_frac, seed=args.seed,
                         legacy_anarci=args.legacy_anarci and not args.new_anarci,
                         limit=args.limit, n_jobs=args.jobs, structural=not args.no_structural,
                         loader=args.loader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
