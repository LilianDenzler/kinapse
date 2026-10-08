"""IMGT insertion-code integrity preflight.

Guards against silently dropping CDR residues that carry IMGT insertion codes
(the 111A/112A... apex of long CDR3s). The mdtraj topology exposes only integer
resSeq; if it merged two insertion residues sharing a base resSeq (e.g. 111 and
111A) we would select ONE CA where there should be TWO. We detect that by
comparing, per CDR:

  n_mdtraj  = len(TrajectoryView.domain_idx([CDR], {"CA"}))     # what compute uses
  n_biopdb  = len(TCRPairView.domain_idx([CDR], {"CA"}))        # Bio.PDB, icodes intact
  n_seq     = len(cdr_fr_sequences()[CDR])                      # sequence ground truth

Any mismatch => residues are being dropped; compute must refuse that (system,CDR).

Usage:
  python preflight.py            # scan all systems, print a table, exit 1 on any mismatch
Import:
  from preflight import cdr_ca_counts, assert_no_dropped_residues
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import glob, warnings
warnings.simplefilter("ignore")
import numpy as np
from kinapse.structures import load_tcr
import config as C


def discover(data=C.DATA):
    out = []
    for d in sorted(glob.glob(os.path.join(data, "*"))):
        sid = os.path.basename(d)
        if len(sid) == 4 and os.path.exists(os.path.join(d, f"{sid}.pdb")) \
           and os.path.exists(os.path.join(d, f"{sid}.xtc")):
            out.append(sid)
    return out


def _load_pair(sid):
    pdb = os.path.join(C.DATA, sid, f"{sid}.pdb")
    kw = {"manual_chain_types": C.CHAIN_OVERRIDES[sid]} if sid in C.CHAIN_OVERRIDES else {}
    # load PDB as its own 1-frame trajectory: identical topology to the xtc path (top=pdb)
    return load_tcr(pdb, traj=pdb, **kw).pairs[0]


def cdr_ca_counts(pair, cdrs=C.CDRS):
    """Per-CDR (n_mdtraj, n_biopdb, n_seq, resnums_mdtraj). resnums show insertions
    as repeated base numbers (e.g. ...111,111,112,112...)."""
    tv = pair.traj
    seqs = pair.cdr_fr_sequences()
    rows = {}
    for cdr in cdrs:
        try:
            mid, mnames = tv.domain_idx([cdr], atom_names={"CA"}, pass_names=True)
        except Exception:
            mid, mnames = [], []
        try:
            bid = pair.domain_idx([cdr], atom_names={"CA"})
        except Exception:
            bid = []
        rows[cdr] = dict(
            n_mdtraj=len(mid),
            n_biopdb=len(bid),
            n_seq=len(seqs.get(cdr, "")),
            resnums=[r for (_, r, _) in mnames],
        )
    return rows


def assert_no_dropped_residues(pair, cdrs=C.CDRS):
    """Raise if the mdtraj CA selection loses any residue vs the icode-preserving
    Bio.PDB / sequence ground truth. Call at the top of every compute job."""
    rows = cdr_ca_counts(pair, cdrs)
    bad = {c: r for c, r in rows.items()
           if not (r["n_mdtraj"] == r["n_biopdb"] == r["n_seq"]) and r["n_seq"] > 0}
    if bad:
        msg = "; ".join(f"{c}: mdtraj={r['n_mdtraj']} biopdb={r['n_biopdb']} seq={r['n_seq']}"
                        for c, r in bad.items())
        raise RuntimeError(f"IMGT insertion-code drop detected -> {msg}")
    return rows


def main():
    systems = C.SYSTEMS or discover()
    print(f"insertion-code preflight over {len(systems)} systems\n")
    hdr = "system " + " ".join(f"{c:>8}" for c in C.CDRS)
    print(hdr); print("-" * len(hdr))
    any_bad = False
    longest = {}
    for sid in systems:
        try:
            pair = _load_pair(sid)
            rows = cdr_ca_counts(pair)
        except Exception as e:
            print(f"{sid:6}  LOAD-FAIL: {type(e).__name__}: {str(e)[:60]}")
            any_bad = True
            continue
        cells = []
        for c in C.CDRS:
            r = rows[c]
            ok = (r["n_mdtraj"] == r["n_biopdb"] == r["n_seq"]) or r["n_seq"] == 0
            any_bad = any_bad or not ok
            longest[c] = max(longest.get(c, 0), r["n_seq"])
            flag = "" if ok else "!"
            cells.append(f"{r['n_mdtraj']}/{r['n_biopdb']}/{r['n_seq']}{flag}".rjust(8))
        print(f"{sid:6} " + " ".join(cells))
    print("\nlongest per CDR (n_seq):", {c: longest[c] for c in C.CDRS})
    # show one long CDR3's resnum stream to eyeball the insertions
    for sid in systems:
        try:
            pair = _load_pair(sid)
            rows = cdr_ca_counts(pair, ["A_CDR3", "B_CDR3"])
        except Exception:
            continue
        for c in ("A_CDR3", "B_CDR3"):
            if rows[c]["n_seq"] >= 13:
                print(f"\n{sid} {c} resnums (insertions = repeats):\n  {rows[c]['resnums']}")
                break
        else:
            continue
        break
    print("\nRESULT:", "MISMATCH — residues dropped" if any_bad else "OK — no residues dropped")
    raise SystemExit(1 if any_bad else 0)


if __name__ == "__main__":
    main()
