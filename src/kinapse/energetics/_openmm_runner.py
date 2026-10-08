#!/usr/bin/env python3
"""Isolated OpenMM runner — energy-minimise a structure and (optionally) compute a
single-trajectory MM/GBSA binding energy. Executed by an *external* interpreter
that has OpenMM + PDBFixer (never imported into kinapse's own env).

Protocol — one JSON object on stdin::

    {"jobs": [{"pdb": str, "out_pdb": str,
               "rec_chains": [..], "lig_chains": [..], "mmgbsa": bool}],
     "max_iter": 0, "solvent": "gbn2", "platform": "auto",
     "add_hydrogens": true, "remove_heterogens": true}

and one JSON object on stdout::

    {"status": "ok", "results": [{"pdb", "out_pdb", "status", "error",
        "e_complex_kcal", "e_rec_kcal", "e_lig_kcal", "dG_bind_mmgb_kcal",
        "e_before_kcal", "e_after_kcal"}]}

MM/GBSA here is the standard single-trajectory scheme: minimise the COMPLEX, then
evaluate the amber14 + implicit-solvent (GB polar + ACE nonpolar surface term)
potential energy of the complex and of the receptor / ligand chain sets extracted
*from the same minimised coordinates* — so ΔG_bind = E(complex) − E(rec) − E(lig)
is the interaction energy plus the change in (polar+nonpolar) solvation on binding.
It contains no configurational-entropy (−TΔS) term.

stdout is reserved for the JSON result; OpenMM chatter goes to stderr.
"""
import json
import os
import sys

_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

KJ_PER_KCAL = 4.184
_SOLVENT_XML = {"gbn2": "implicit/gbn2.xml", "obc2": "implicit/obc2.xml",
                "obc1": "implicit/obc1.xml", "hct": "implicit/hct.xml", "none": None}


def _emit(obj):
    json.dump(obj, _REAL_STDOUT, default=str)
    _REAL_STDOUT.flush()


def _fixed_topology(pdb, add_hydrogens=True, remove_heterogens=True):
    """PDBFixer: add missing heavy atoms + H, but NOT missing residues (no loop
    modelling), drop non-standard heterogens/waters. Returns (topology, positions)."""
    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=pdb)
    fixer.findMissingResidues()
    fixer.missingResidues = {}                 # do NOT build missing loops
    if remove_heterogens:
        fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if add_hydrogens:
        fixer.addMissingHydrogens(7.0)
    return fixer.topology, fixer.positions


def _make_ff(solvent):
    from openmm.app import ForceField
    xml = _SOLVENT_XML.get(solvent, "implicit/gbn2.xml")
    return ForceField("amber14-all.xml", xml) if xml else ForceField("amber14-all.xml")


def _platform(name):
    from openmm import Platform
    if name and name != "auto":
        return Platform.getPlatformByName(name)
    for p in ("CUDA", "OpenCL", "CPU"):
        try:
            return Platform.getPlatformByName(p)
        except Exception:  # noqa: BLE001
            continue
    return None


def _energy_kcal(topology, positions, ff, platform):
    """Potential energy (kcal/mol) of a topology+positions under the implicit-solvent FF."""
    from openmm import LangevinIntegrator, unit
    from openmm.app import Simulation, NoCutoff
    system = ff.createSystem(topology, nonbondedMethod=NoCutoff, constraints=None)
    integ = LangevinIntegrator(300, 1.0, 0.001)
    sim = (Simulation(topology, system, integ, platform) if platform
           else Simulation(topology, system, integ))
    sim.context.setPositions(positions)
    e = sim.context.getState(getEnergy=True).getPotentialEnergy()
    return e.value_in_unit(unit.kilojoule_per_mole) / KJ_PER_KCAL


def _subset_energy(topology, positions, keep_chains, ff, platform):
    """Energy of just `keep_chains`, coordinates taken from the full minimised complex."""
    from openmm.app import Modeller
    mod = Modeller(topology, positions)
    to_delete = [ch for ch in mod.topology.chains() if ch.id not in set(keep_chains)]
    if to_delete:
        mod.delete(to_delete)
    return _energy_kcal(mod.topology, mod.positions, ff, platform)


def _minimise_one(job, ff, platform, max_iter, add_h, rm_het):
    from openmm import LangevinIntegrator, unit
    from openmm.app import Simulation, NoCutoff, PDBFile
    rec = {"pdb": job["pdb"], "out_pdb": job.get("out_pdb"), "status": "error", "error": None,
           "e_before_kcal": None, "e_after_kcal": None,
           "e_complex_kcal": None, "e_rec_kcal": None, "e_lig_kcal": None,
           "dG_bind_mmgb_kcal": None}
    try:
        topology, positions = _fixed_topology(job["pdb"], add_h, rm_het)
        system = ff.createSystem(topology, nonbondedMethod=NoCutoff, constraints=None)
        integ = LangevinIntegrator(300, 1.0, 0.001)
        sim = (Simulation(topology, system, integ, platform) if platform
               else Simulation(topology, system, integ))
        sim.context.setPositions(positions)
        rec["e_before_kcal"] = (sim.context.getState(getEnergy=True).getPotentialEnergy()
                                .value_in_unit(unit.kilojoule_per_mole) / KJ_PER_KCAL)
        sim.minimizeEnergy(maxIterations=int(max_iter))
        state = sim.context.getState(getEnergy=True, getPositions=True)
        rec["e_after_kcal"] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / KJ_PER_KCAL
        min_positions = state.getPositions()
        if job.get("out_pdb"):
            os.makedirs(os.path.dirname(os.path.abspath(job["out_pdb"])), exist_ok=True)
            with open(job["out_pdb"], "w") as fh:
                PDBFile.writeFile(topology, min_positions, fh, keepIds=True)
        # single-trajectory MM/GBSA on the minimised complex
        if job.get("mmgbsa") and job.get("rec_chains") and job.get("lig_chains"):
            ec = _energy_kcal(topology, min_positions, ff, platform)
            er = _subset_energy(topology, min_positions, job["rec_chains"], ff, platform)
            el = _subset_energy(topology, min_positions, job["lig_chains"], ff, platform)
            rec["e_complex_kcal"], rec["e_rec_kcal"], rec["e_lig_kcal"] = ec, er, el
            rec["dG_bind_mmgb_kcal"] = ec - er - el
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    args = json.loads(sys.stdin.read() or "{}")
    jobs = args.get("jobs") or []
    if not jobs:
        _emit({"status": "error", "error": "no jobs"})
        return
    try:
        ff = _make_ff(args.get("solvent", "gbn2"))
        platform = _platform(args.get("platform", "auto"))
    except Exception as e:  # noqa: BLE001
        _emit({"status": "error", "error": f"openmm setup failed: {type(e).__name__}: {e}"})
        return
    results = [_minimise_one(j, ff, platform, args.get("max_iter", 0),
                             args.get("add_hydrogens", True),
                             args.get("remove_heterogens", True)) for j in jobs]
    _emit({"status": "ok", "results": results})


if __name__ == "__main__":
    main()
