#!/usr/bin/env python3
"""Isolated STCRpy runner — executed by an *external* interpreter that has STCRpy
installed (never imported into kinapse's environment).

Protocol: reads ``{"pdb": "<path>", "geometry": bool, "interactions": bool}`` as
JSON on stdin (geometry/interactions default True), writes a single JSON object on
stdout. Always emits ``receptor_chains`` / ``ligand_chains`` (TCR alpha/beta vs
peptide+MHC) and ``regions`` (per-CDR/FR Ca, matching kinapse's region keys). Adds
docking-geometry ``angles`` and PLIP-typed ``interactions`` only when requested.
Depends only on STCRpy (+ its own deps) — nothing from kinapse.

stdout is reserved for the JSON result; STCRpy and its deps often print, so all
library output is redirected to stderr for the duration of the run.
"""
import json
import os
import sys

# Running this script by path puts its own directory on sys.path[0]; that directory
# also holds kinapse's ``stcrpy.py``, which would shadow the real STCRpy package on
# ``import stcrpy``. Drop our own directory so the external STCRpy wins.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

_REAL_STDOUT = sys.stdout


def _emit(obj):
    """Write the JSON result to the *real* stdout (the only thing the caller parses)."""
    json.dump(obj, _REAL_STDOUT, default=str)
    _REAL_STDOUT.flush()


def _chain_id(obj):
    """Best-effort original PDB chain id of a STCRpy chain/entity."""
    cid = getattr(obj, "id", None)
    if isinstance(cid, str):
        return cid
    return str(cid) if cid is not None else None


def _extract_chains(tcr):
    """(receptor, ligand) original chain ids: TCR alpha/beta vs peptide + MHC (+ b2m)."""
    rec, lig = [], []
    for getter in ("get_VA", "get_VB"):
        try:
            cid = _chain_id(getattr(tcr, getter)())
            if cid and cid not in rec:
                rec.append(cid)
        except Exception:
            pass

    def add(container):
        items = container if isinstance(container, (list, tuple)) else [container]
        for it in items:
            if it is None:
                continue
            get_chains = getattr(it, "get_chains", None)
            ids = ([_chain_id(c) for c in get_chains()] if callable(get_chains)
                   else [_chain_id(it)])
            for cid in ids:
                if cid and cid not in rec and cid not in lig:
                    lig.append(cid)

    for getter in ("get_antigen", "get_MHC"):
        try:
            add(getattr(tcr, getter)())
        except Exception:
            pass
    return rec, lig


def _parse_region(fid):
    """STCRpy fragment id ('cdra1', 'fwb2', ...) -> (kinapse region name, chain letter).

    Char after the prefix is the chain type (a/b alpha/beta, d/g delta/gamma); last
    char is the region number. Maps to kinapse's 'A_CDR1' / 'B_FR2' style (a->A, b->B)."""
    fid = (fid or "").lower()
    if fid.startswith("cdr"):
        kind, ct, num = "CDR", fid[3:4], fid[-1:]
    elif fid.startswith("fw"):
        kind, ct, num = "FR", fid[2:3], fid[-1:]
    else:
        return None
    if not num.isdigit():
        return None
    canon = "A" if ct in ("a", "d") else "B"
    return f"{canon}_{kind}{num}", canon


def _extract_regions(tcr):
    """Per-residue Ca for every CDR/FR fragment: [{region, chain, imgt, xyz}]."""
    frags = []
    for getter in ("get_CDRs", "get_frameworks"):
        try:
            frags.extend(list(getattr(tcr, getter)()))
        except Exception:
            pass
    out = []
    for f in frags:
        parsed = _parse_region(getattr(f, "id", ""))
        if not parsed:
            continue
        region, canon = parsed
        residues = f.get_residues() if hasattr(f, "get_residues") else f
        for res in residues:
            try:
                if "CA" not in res:
                    continue
                out.append({"region": region, "chain": canon,
                            "imgt": int(res.id[1]),
                            "xyz": [float(x) for x in res["CA"].coord]})
            except Exception:
                continue
    return out


def _jsonable(obj):
    """Best-effort JSON serialisation of STCRpy return values (DataFrame/dict/obj)."""
    to_dict = getattr(obj, "to_dict", None)
    if to_dict is not None:
        try:
            return obj.to_dict(orient="records")   # pandas DataFrame
        except TypeError:
            try:
                return obj.to_dict()
            except Exception:
                pass
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def main():
    args = json.loads(sys.stdin.read() or "{}")
    pdb = args.get("pdb")
    want_geometry = args.get("geometry", True)
    want_interactions = args.get("interactions", True)

    # From here on, keep stdout clean: send any library chatter to stderr.
    sys.stdout = sys.stderr

    try:
        import stcrpy
    except Exception as e:  # noqa: BLE001
        _emit({"status": "error", "error": f"import stcrpy failed: {e}"})
        return

    try:
        tcr = stcrpy.load_TCR(pdb)
        if isinstance(tcr, (list, tuple)) and tcr:
            tcr = tcr[0]
    except Exception as e:  # noqa: BLE001
        _emit({"status": "error", "error": f"load_TCR failed: {e}"})
        return

    out = {"status": "ok"}
    # chains + CDR/FR annotation — always (cheap; numbering already done by load_TCR)
    try:
        rec, lig = _extract_chains(tcr)
        out["receptor_chains"], out["ligand_chains"] = rec, lig
    except Exception as e:  # noqa: BLE001
        out["chains_error"] = str(e)
    try:
        out["regions"] = _extract_regions(tcr)
    except Exception as e:  # noqa: BLE001
        out["regions_error"] = str(e)

    if want_geometry:
        try:
            tcr.calculate_geometry()
            out["angles"] = _jsonable(tcr.get_TCR_angles())
        except Exception as e:  # noqa: BLE001
            out["geometry_error"] = str(e)
    if want_interactions:
        try:
            prof = tcr.profile_peptide_interactions()   # needs PLIP; H-bonds/salt-bridges/...
            out["interactions"] = _jsonable(prof)
        except Exception as e:  # noqa: BLE001
            out["interactions_error"] = str(e)

    _emit(out)


if __name__ == "__main__":
    main()
