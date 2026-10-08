#!/usr/bin/env python
"""Extract each CDR's one-letter sequence per TCR (for composition-vs-flexibility correlation)."""
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import glob, json, warnings, traceback
warnings.simplefilter("ignore")
from graph_build import load_md, CDR_RANGES, DATA, HERE

CDRS = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
T3 = {"GLY": "G", "PRO": "P", "ALA": "A", "SER": "S", "THR": "T", "CYS": "C", "VAL": "V", "LEU": "L",
      "ILE": "I", "MET": "M", "PHE": "F", "TYR": "Y", "TRP": "W", "HIS": "H", "LYS": "K", "ARG": "R",
      "ASP": "D", "GLU": "E", "ASN": "N", "GLN": "Q"}


def main():
    systems = sorted(os.path.basename(p) for p in glob.glob(f"{DATA}/*")
                     if len(os.path.basename(p)) == 4 and os.path.exists(f"{p}/{os.path.basename(p)}.xtc"))
    out = {}
    for i, s in enumerate(systems):
        try:
            tv, xyz, imap = load_md(s, stride=1500)
            top = tv.mdtraj.topology
            d = {}
            for cdr in CDRS:
                ch = cdr[0]; lo, hi = CDR_RANGES[cdr[2:]]
                lk = sorted(k for k in imap[ch] if lo <= k <= hi)
                d[cdr] = "".join(T3.get(top.atom(imap[ch][k]).residue.name, "X") for k in lk)
            out[s] = d
            print(f"[{i+1}/{len(systems)}] {s}: CDR3 {d['A_CDR3']}/{d['B_CDR3']}", flush=True)
        except Exception as e:
            print(f"[{s}] ERR {type(e).__name__}: {str(e)[:100]}", flush=True); traceback.print_exc()
    json.dump(out, open(f"{HERE}/cdr_sequences.json", "w"), indent=2)
    print("SEQ_DONE", flush=True)


if __name__ == "__main__":
    main()
