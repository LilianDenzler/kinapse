#!/usr/bin/env python3
"""Isolated STCRpy runner — executed by an *external* interpreter that has STCRpy
installed (never imported into kinapse's environment).

Protocol: reads ``{"pdb": "<path>"}`` as JSON on stdin, writes a JSON object on
stdout with STCRpy's typed peptide interactions, docking-geometry angles, and any
annotations. Depends only on STCRpy (+ its own deps) — nothing from kinapse.
"""
import json
import sys


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

    try:
        import stcrpy
    except Exception as e:  # noqa: BLE001
        json.dump({"status": "error", "error": f"import stcrpy failed: {e}"}, sys.stdout)
        return

    try:
        tcr = stcrpy.load_TCR(pdb)
        if isinstance(tcr, (list, tuple)) and tcr:
            tcr = tcr[0]
    except Exception as e:  # noqa: BLE001
        json.dump({"status": "error", "error": f"load_TCR failed: {e}"}, sys.stdout)
        return

    out = {"status": "ok"}
    try:
        tcr.calculate_geometry()
        out["angles"] = _jsonable(tcr.get_TCR_angles())
    except Exception as e:  # noqa: BLE001
        out["geometry_error"] = str(e)
    try:
        prof = tcr.profile_peptide_interactions()   # needs PLIP; H-bonds/salt-bridges/…
        out["interactions"] = _jsonable(prof)
    except Exception as e:  # noqa: BLE001
        out["interactions_error"] = str(e)

    json.dump(out, sys.stdout, default=str)


if __name__ == "__main__":
    main()
