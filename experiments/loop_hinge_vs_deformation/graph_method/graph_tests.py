"""Validation gate (joint hinge+deformation model). No MD needed.
(1) SE(3)+reflection invariance; (2) framework-reference invariance of q_t; (3) LARGE-angle pure hinge ->
f_hinge~1, F_deform~0; (4) breathing deformation -> low f_hinge, F_deform>0; (5) ROTATION-MIMICKING drift ->
low f_hinge (deformation NOT swallowed by the hinge -- the joint-model fix); (6) distance fit == Kabsch oracle."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import graph_fit as GF, rotations as RO, graph_decomp as GD, hinge_validity as HV

rng = np.random.default_rng(0)
N, M, T = 12, 40, 120
loop0 = rng.normal(scale=5, size=(N, 3)); c = loop0.mean(0)
fw0 = rng.normal(scale=13, size=(M, 3))


def rot(deg, ax):
    ax = ax / np.linalg.norm(ax); return Rot.from_rotvec(np.radians(deg) * ax).as_matrix()


def in_lab(loop_fr, fw_fr):
    L = np.empty_like(loop_fr); Fw = np.empty_like(fw_fr)
    for t in range(len(loop_fr)):
        G = rot(rng.uniform(0, 360), rng.normal(size=3)); tg = rng.normal(scale=30, size=3)
        L[t] = loop_fr[t] @ G.T + tg; Fw[t] = fw_fr[t] @ G.T + tg
    return L, Fw


FW = np.repeat(fw0[None], T, 0)

# ---- 1. SE(3)+reflection invariance ---------------------------------------
lp = np.repeat(loop0[None], 5, 0); fp = np.repeat(fw0[None], 5, 0)
b_LF, b_LL = GD.dLF_vector(lp, fp), GD.dLL_pairs(lp); worst = 0.0
for _ in range(20):
    Rm = rot(rng.uniform(0, 360), rng.normal(size=3)); t = rng.normal(scale=40, size=3)
    if rng.random() < 0.5: Rm = Rm @ np.diag([1, 1, -1])
    worst = max(worst, np.abs(GD.dLF_vector(lp @ Rm.T + t, fp @ Rm.T + t) - b_LF).max(),
                np.abs(GD.dLL_pairs(lp @ Rm.T + t) - b_LL).max())
print(f"[1] invariance          max dev = {worst:.2e}"); assert worst < 1e-9

# ---- large-angle PURE HINGE ------------------------------------------------
axis = rng.normal(size=3); angles = rng.uniform(-40, 40, T)
loop_lab, fw_lab = in_lab(np.stack([(loop0 - c) @ rot(a, axis).T + c for a in angles]), FW)

# ---- 2. framework-reference invariance of q_t ------------------------------
qa = GD.run_cdr(loop_lab, fw_lab, fw_ref=fw_lab[0])["theta_deg"]
qb = GD.run_cdr(loop_lab, fw_lab, fw_ref=fw_lab[9])["theta_deg"]
print(f"[2] fw-ref invariance   max|dtheta| = {np.abs(qa - qb).max():.2e} deg"); assert np.abs(qa - qb).max() < 1e-4

# ---- 3. pure hinge: f_hinge~1, F_deform~0 ----------------------------------
r3 = GD.run_cdr(loop_lab, fw_lab)
print(f"[3] large pure hinge    f_hinge={r3['f_hinge']:.4f}  f_deform_LF={r3['f_deform_LF']:.4f}  "
      f"F_deform={r3['F_deform']:.2e}  modes={r3['n_deform_modes']}")
assert r3["f_hinge"] > 0.99 and r3["F_deform"] < 1e-5

# ---- 4. breathing deformation + small rotation: low f_hinge, F_deform>0 -----
th = rng.normal(0, np.radians(3), T); axd = rng.normal(size=3); sc = 1.0 + rng.normal(0, 0.06, T)
ld, fd = in_lab(np.stack([sc[t] * ((loop0 - c) @ rot(np.degrees(th[t]), axd).T) + c for t in range(T)]), FW)
r4 = GD.run_cdr(ld, fd)
print(f"[4] breathing deform    f_hinge={r4['f_hinge']:.3f}  f_deform_LF={r4['f_deform_LF']:.3f}  F_deform={r4['F_deform']:.3f}")
assert r4["f_hinge"] < 0.5 and r4["F_deform"] > 0.1

# ---- 5. ROTATION-MIMICKING drift (the GPA failure mode): must NOT be hinge --
drift = np.linspace(-2.0, 2.0, T)[:, None, None] * np.zeros((1, N, 3))
dirv = rng.normal(size=3); dirv /= np.linalg.norm(dirv)
loop_drift = np.repeat(loop0[None], T, 0).copy()
loop_drift[:, [1, 2, 3]] += np.linspace(-2, 2, T)[:, None, None] * dirv[None, None, :]   # coherent directional drift
lm, fm = in_lab(loop_drift, FW)
r5 = GD.run_cdr(lm, fm)
print(f"[5] rotation-mimic drift f_hinge={r5['f_hinge']:.3f}  f_deform_LF={r5['f_deform_LF']:.3f}  "
      f"F_deform={r5['F_deform']:.3f}  (deformation, not hinge)")
assert r5["f_hinge"] < 0.5 and r5["F_deform"] > 0.1

# ---- 6. pure-distance rigid fit == Kabsch oracle --------------------------
def _kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0); U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); return Vt.T @ np.diag([1, 1, d]) @ U.T
Rh = rot(27.0, np.array([0.3, -1.0, 0.5])); lh = (loop0 - c) @ Rh.T + c
Gg = rot(rng.uniform(0, 360), rng.normal(size=3)); tg = rng.normal(scale=30, size=3)
lt = lh @ Gg.T + tg; ft = fw0 @ Gg.T + tg
Rfw = _kabsch(ft, fw0); R_or = _kabsch(loop0, lt @ Rfw.T + (fw0.mean(0) - (ft @ Rfw.T).mean(0)))
R_di, _, _ = GF.distance_rigid_fit(loop0, fw0, GF.cross(lt, ft))
print(f"[6] distance==oracle    theta_oracle={GF.theta_of(R_or):.3f}  theta_dist={GF.theta_of(R_di):.3f}")
assert abs(GF.theta_of(R_or) - GF.theta_of(R_di)) < 1e-2

# ---- 7. CORRELATED simultaneous hinge + deformation (the hard regime) ------
#     loop hinges by alpha_t AND deforms with amplitude linear in alpha_t (hinge & deformation correlated).
#     Because z comes only from d_LL (which the rigid hinge cannot move), the hinge angle must still be
#     recovered and the deformation must still be detected, with the correlation showing up in f_coupling.
alpha = np.sort(rng.uniform(-25, 25, T)); haxis = rng.normal(size=3)
hinge = np.stack([(loop0 - c) @ rot(a, haxis).T + c for a in alpha])
defmode = np.zeros((N, 3)); defmode[[2, 5, 8]] = rng.normal(size=(3, 3))
amp = (alpha - alpha.mean()) / 25.0 * 1.5                                    # deformation amplitude ~ hinge
loop_corr = hinge + amp[:, None, None] * defmode[None]
lc, fc = in_lab(loop_corr, FW)
r7 = GD.run_cdr(lc, fc)
q_rec = r7["theta_deg"]; corr = abs(np.corrcoef(q_rec, alpha - alpha.mean())[0, 1])
budget = r7["f_hinge"] + r7["f_deform_LF"] + r7["f_coupling"] + r7["f_residual"]
print(f"[7] correlated H+D      q~alpha corr={corr:.3f}  f_hinge={r7['f_hinge']:.2f} f_deform_LF={r7['f_deform_LF']:.2f} "
      f"f_coupling={r7['f_coupling']:.2f} f_resid={r7['f_residual']:.2f}  (sum={budget:.3f})  F_deform={r7['F_deform']:.2f}")
assert corr > 0.9 and r7["F_deform"] > 0.1 and r7["f_deform_LF"] > 0.03 and abs(budget - 1.0) < 1e-6

# ---- 8. single-hinge validity: PURE one-axis motion -> P_1D~1, dE~0 --------
v8 = HV.single_hinge_report(loop_lab, fw_lab)                                # the ±40° one-axis ensemble
print(f"[8] one-axis validity   P_1D={v8['P_1D']:.4f}  r_perp={v8['r_perp']:.4f}  delta_rms={v8['delta_rms_deg']:.2f}deg  "
      f"dE_axis={v8['dE_axis']:.3f}A  block_axis_agree={v8['block_axis_agree_deg']:.2f}deg")
assert v8["P_1D"] > 0.99 and v8["delta_rms_deg"] < 1.0 and v8["dE_axis"] < 0.05 and v8["block_axis_agree_deg"] < 2.0

# ---- 9. TWO-axis motion -> P_1D drops, off-axis error large, dE large -------
a1 = rng.normal(size=3); a2 = np.cross(a1, rng.normal(size=3))               # a2 !~ a1
frames = []
for k in range(T):
    ax = a1 if k % 2 == 0 else a2
    frames.append((loop0 - c) @ rot(rng.uniform(-25, 25), ax).T + c)         # alternate the rotation axis
l9, f9 = in_lab(np.stack(frames), FW)
v9 = HV.single_hinge_report(l9, f9)
print(f"[9] two-axis motion     P_1D={v9['P_1D']:.3f}  delta_rms={v9['delta_rms_deg']:.2f}deg  dE_axis={v9['dE_axis']:.3f}A")
assert v9["P_1D"] < 0.9 and v9["delta_rms_deg"] > 3.0 and v9["dE_axis"] > v8["dE_axis"]

print("ALL GRAPH-METHOD TESTS PASSED")
