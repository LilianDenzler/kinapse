#!/usr/bin/env python
"""Simplified alignment-free TCR CDR graph builder.

Load a TCR MD trajectory with kinapse (which applies IMGT numbering), then turn ONE CDR loop into a
DISTANCE GRAPH over C-alpha atoms:

    nodes = the loop's CA atoms  +  rigid framework "anchor" CA atoms flanking the loop
    edges = CA-CA distances, in two families
              loop <-> loop   (d_LL)   the loop's internal shape  (a rigid hinge cannot change these)
              loop <-> frame  (d_LF)   the loop's pose relative to the framework

Each edge stores the distance AVERAGED over the trajectory (D_mean) and its FLUCTUATION (D_std, the
std over frames). Nothing is ever aligned/superposed -- pairwise distances are invariant to how the
whole molecule tumbles, which is the point of the graph representation.

CLI:  python graph_build.py 3QH3 B_CDR3          # build + save one graph to graphs/
API:  load_md(sysid) -> (tv, xyz_A, imap)
      build_cdr_graph(sysid, chain, cdr) -> dict
"""
from __future__ import annotations
import os, sys
if os.environ.get("PYTHONNOUSERSITE") != "1":                 # kinapse env must not see user site-packages
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
import tempfile, warnings, functools
warnings.simplefilter("ignore")
import numpy as np, mdtraj as md
from kinapse.structures import load_tcr

# --- config (mirrors ../config.py; kept here so this folder stands alone) ---
DATA = "/mnt/larry/lilian/DATA/CORY_ORIOL_MERGED_MD"          # <ID>/<ID>.{pdb,xtc}
STRIDE = 25                                                   # frame subsampling
CHAIN_OVERRIDES = {"8YJ3": {"A": "B", "B": "A"}}              # ANARCII mistypes 8YJ3's chains
CDR_RANGES = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}   # IMGT, same for both chains
HERE = os.path.dirname(os.path.abspath(__file__))


@functools.lru_cache(maxsize=1)
def consensus():
    """{'A': set, 'B': set} of curated RIGID framework IMGT positions (kinapse packaged data).
    These exclude the CDR loops AND the mobile FR loops, so they are a trustworthy rigid reference."""
    import kinapse.geometry as g
    base = str(g.DATA_PATH)
    out = {}
    for ch in "AB":
        txt = open(f"{base}/chain_{ch}/consensus_alignment_residues.txt").read()
        out[ch] = {int(x) for x in txt.split(",") if x.strip()}
    return out


def load_md(sysid, stride=STRIDE):
    """Load a TCR MD with kinapse. Returns (tv, xyz_angstrom(T,Natom,3), imap) where
    imap[chain] maps an IMGT residue number -> that residue's CA atom index."""
    pdb, xtc = f"{DATA}/{sysid}/{sysid}.pdb", f"{DATA}/{sysid}/{sysid}.xtc"
    tmp = tempfile.mktemp(suffix=".xtc")
    try:
        md.load(xtc, top=pdb, stride=stride).save_xtc(tmp)
        kw = {"manual_chain_types": CHAIN_OVERRIDES[sysid]} if sysid in CHAIN_OVERRIDES else {}
        tv = load_tcr(pdb, traj=tmp, **kw).pairs[0].traj      # kinapse renumbers to IMGT here
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    xyz = tv.mdtraj.xyz * 10.0                                # nm -> Angstrom
    imap = {}
    for ch in "AB":
        vi, vn = tv.domain_idx([f"{ch}_variable"], atom_names={"CA"}, pass_names=True)
        m = {}; seen = {}
        for i, n in zip(vi, vn):
            p = int(n[1])
            if p in seen:                                     # IMGT insertion: kinapse repeats the base number
                seen[p] += 1; key = round(p + seen[p] / 10.0, 1)   # 112 & 112A -> 112, 112.1  (keeps both CAs)
            else:
                seen[p] = 0; key = p
            m[key] = int(i)
        imap[ch] = m
    return tv, xyz, imap


def _graph_from_nodes(sysid, loop_chain, cdr, xyz, imap, nodes, ntype):
    """Shared: nodes = ordered list of (chain, imgt) tuples. Compute the CA distance graph over the MD.
    node_label is 'A41'/'B41' style when >1 chain is present, else the bare IMGT number."""
    ai = np.array([imap[ch][n] for ch, n in nodes])
    P = xyz[:, ai]                                                                # (T, N, 3) CA coords
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)              # (T, N, N) distances
    node_chain = np.array([ch for ch, _ in nodes]); node_imgt = np.array([n for _, n in nodes])
    multi = len(set(node_chain.tolist())) > 1
    node_label = np.array([f"{ch}{n}" if multi else str(n) for ch, n in nodes])
    return dict(
        sysid=sysid, chain=loop_chain, cdr=cdr,
        node_imgt=node_imgt, node_chain=node_chain, node_label=node_label, node_type=np.array(ntype),
        n_loop=int(sum(t == "loop" for t in ntype)), n_flank=int(sum(t == "flank" for t in ntype)),
        n_frame=int(sum(t == "frame" for t in ntype)), n_frames=int(len(P)),
        mean_xyz=P.mean(0).astype(np.float32),                                    # (N,3) mean positions
        D_mean=D.mean(0).astype(np.float32), D_std=D.std(0).astype(np.float32),   # (N,N) graph + fluctuation
    )


def flanking_residues(chain, cdr):
    """Sequential FLANKING residues: those sitting BETWEEN the CDR and the first rigid-set residue on
    each side (not distance-wise). Returns the IMGT numbers (usually empty -- the rigid set starts right
    at the loop; only beta 26 before CDR1 and beta 66 after CDR2 are non-empty for these consensus sets)."""
    con = consensus()[chain]; lo, hi = CDR_RANGES[cdr]
    L = lo - 1
    while L >= min(con) and L not in con:
        L -= 1
    left = list(range(L + 1, lo))                       # rigid at L, loop at lo -> gap between
    R = hi + 1
    while R <= max(con) and R not in con:
        R += 1
    right = list(range(hi + 1, R))
    return left + right


def build_cdr_graph(sysid, chain, cdr, framework="full", fw_chains=None, flank=8, stride=STRIDE, md_cache=None):
    """Build the distance graph for one CDR loop, with three node families:
        loop  : the CDR residues                                   (from `chain`)
        flank : sequential transition residues loop<->rigid set    (from `chain`, see flanking_residues)
        frame : the super-rigid CONSENSUS framework residues        (from `fw_chains`)

    fw_chains : which chains contribute framework nodes. None -> [chain] (same chain only).
                ["A","B"] -> BOTH chains' rigid framework combined (alpha+beta reference).
    framework : "full" (whole rigid set) or "local" (only consensus within +/-flank of the loop)."""
    tv, xyz, imap = md_cache if md_cache is not None else load_md(sysid, stride)
    lo, hi = CDR_RANGES[cdr]
    fw_chains = list(fw_chains) if fw_chains is not None else [chain]
    loop = [(chain, n) for n in range(lo, hi + 1) if n in imap[chain]]
    flk = [(chain, n) for n in flanking_residues(chain, cdr) if n in imap[chain]]
    anch = []
    for fc in fw_chains:
        con = consensus()[fc]
        if framework == "local" and fc == chain:
            anch += [(fc, n) for n in range(lo - flank, hi + flank + 1)
                     if n in imap[fc] and n in con and not (lo <= n <= hi)]
        else:
            anch += [(fc, n) for n in sorted(con) if n in imap[fc] and not (fc == chain and lo <= n <= hi)]
    return _graph_from_nodes(sysid, chain, cdr, xyz, imap, loop + flk + anch,
                             ["loop"] * len(loop) + ["flank"] * len(flk) + ["frame"] * len(anch))


def build_framework_graph(sysid, chains, stride=STRIDE, md_cache=None):
    """Distance graph of JUST the super-rigid framework residues. chains='A', 'B' or 'AB'.
    Useful to (a) see the rigid scaffold and (b) CHECK it is rigid -- D_std should be uniformly small."""
    tv, xyz, imap = md_cache if md_cache is not None else load_md(sysid, stride)
    nodes = [(fc, n) for fc in chains for n in sorted(consensus()[fc]) if n in imap[fc]]
    return _graph_from_nodes(sysid, chains, "FRAMEWORK", xyz, imap, nodes, ["frame"] * len(nodes))


def build_chain_graph(sysid, chain, stride=STRIDE, md_cache=None):
    """Distance graph over ALL variable-domain residues of one chain (CDR loops + framework),
    node_type = 'loop' for CDR residues, 'frame' otherwise. Used to derive a rigid set from fluctuation."""
    tv, xyz, imap = md_cache if md_cache is not None else load_md(sysid, stride)
    nodes = [(chain, n) for n in sorted(imap[chain])]
    ntype = ["loop" if any(lo <= n <= hi for lo, hi in CDR_RANGES.values()) else "frame" for _, n in nodes]
    return _graph_from_nodes(sysid, chain, "ALLRES", xyz, imap, nodes, ntype)


def save_graph(g, path=None):
    path = path or f"{HERE}/graphs/{g['sysid']}_{g['chain']}_{g['cdr']}.npz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **g)
    return path


def _parse_cdr(token):
    """'B_CDR3' -> ('B','CDR3');  'CDR3' -> ('A','CDR3');  'A3' -> ('A','CDR3')."""
    if "_" in token:
        ch, c = token.split("_")
    elif token[0] in "AB" and token[1:].isdigit():
        ch, c = token[0], "CDR" + token[1:]
    else:
        ch, c = "A", token
    return ch, (c if c.startswith("CDR") else "CDR" + c)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build one CDR distance graph and save it.")
    ap.add_argument("sysid", help="4-char TCR id, e.g. 3QH3")
    ap.add_argument("cdr", help="e.g. B_CDR3, A_CDR1, CDR3")
    ap.add_argument("--flank", type=int, default=6)
    a = ap.parse_args()
    chain, cdr = _parse_cdr(a.cdr)
    g = build_cdr_graph(a.sysid, chain, cdr, flank=a.flank)
    p = save_graph(g)
    print(f"{a.sysid} {chain}_{cdr}: {g['n_loop']} loop + {g['n_frame']} frame nodes, "
          f"{g['n_frames']} frames  ->  {p}")
