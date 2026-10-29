"""Out-of-sample test of the receiver placement rule against measured data: Vesterholm's Mdau2 (another M. daubentonii
specimen, measured in-ear HRTF of the left ear). The rule (scripts/apply_placement_rule.py: candidates around the most
occluded spot of the rear head, score z(front/back ratio) + z(occlusion), smoothed) was learned on P. discolor and
M. microtis only. Every candidate is solved at ten frequencies (25-100 kHz) on the full sphere (for the front/back
ratio) and at the measurement directions, and compared with the measured magnitudes, with their rotation and with the
rotation fitted by scripts/vesterholm_rotation_fit.py. Reported: where the rule's choice ranks among the candidates,
against the best, the median and the most occluded candidate, and against their Mesh2HRTF receiver position.
Usage: python scripts/vesterholm_rule_test.py   -> results/validation/vesterholm_rule_test.json"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import gpubem, grid, mesh, paths, reference, sources
from apply_placement_rule import SMOOTH
from ear_landmarks import ear_candidates
from learn_placement_rule import descriptors
from placement_study import EAR_PATCH, EAR_REGION, SPACING, sample
from validate_vesterholm import FLOOR, SUB, load_measured, load_sofa, measured_mag, unit

N_FREQ = 10


def main():
    d = paths.work_dir().parent / "external" / "vesterholm2025" / "files"
    meas, sofa = load_measured(d), load_sofa(d)
    R3 = sio.loadmat(d / SUB / "Measured HRTF" / "Mdau2_RotationMatrix.mat")["RotoTranslationMatrix"][:3, :3]
    fit = json.loads((paths.RESULTS / "validation" / "vesterholm_rotation_fit.json").read_text())
    Q = Rotation.from_euler("zyx", fit["ours"]["extra_rotation_euler_zyx_deg"], degrees=True).as_matrix()
    u_meas = unit(meas["az"], meas["el"])
    u_their, u_fit = u_meas @ R3.T, u_meas @ (Q @ R3).T
    freqs = np.linspace(25e3, 100e3, N_FREQ + 2)[1:-1]
    Mm, w = measured_mag(meas, freqs), meas["w"]

    m = mesh.load(paths.work_dir() / "vesterholm" / "Mdau2_full.stl")
    f_ao = paths.work_dir() / "vesterholm" / "Mdau2_full_ao.npz"
    if f_ao.exists():
        ao = np.load(f_ao)["ao"]
    else:
        ao = mesh.ambient_occlusion(m)
        np.savez(f_ao, ao=ao)
    centre = ear_candidates(m, ao)["L"][0]
    P = sample(m, centre, EAR_REGION, SPACING)
    rec = sofa["receiver"]
    names = [f"earL{k:03d}" for k in range(len(P))] + ["their_receiver"]
    pos = np.vstack([P, rec])
    patches = {n: [sources.disc_patch(m, p, EAR_PATCH)] for n, p in zip(names, pos)}
    print(f"{len(P)} candidates around {np.round(centre * 1e3, 2)} mm; their receiver "
          f"{np.linalg.norm(rec - centre) * 1e3:.2f} mm from the centre; {len(m.faces)} faces", flush=True)

    sph, _ = grid.unique_points("sphere")
    pts = np.vstack([sph, u_their, u_fit])
    n0, n1 = len(sph), len(sph) + len(u_meas)
    H = gpubem.solve_lu(m, patches, freqs, pts, log=lambda s: print(s, flush=True))

    tree = cKDTree(m.vertices)
    res = {}
    for n in names:
        h = H[n]
        res[n] = {"front_back_db": descriptors(gpubem.to_grid(h[:n0], "sphere"))["front_back_db"],
                  "occlusion": float(ao[tree.query(pos[names.index(n)])[1]]),
                  "corr_their_rotation": float(np.mean(reference.metrics_points(np.abs(h[n0:n1]), Mm, w, u_meas, FLOOR)["corr"])),
                  "corr_fitted_rotation": float(np.mean(reference.metrics_points(np.abs(h[n1:]), Mm, w, u_meas, FLOOR)["corr"]))}
    cand = names[:-1]
    z = lambda v: (v - v.mean()) / v.std()
    sc = z(np.array([res[n]["front_back_db"] for n in cand])) + z(np.array([res[n]["occlusion"] for n in cand]))
    t = cKDTree(P)
    sc = np.array([sc[t.query_ball_point(p, SMOOTH)].mean() for p in P])
    rule = cand[int(np.argmax(sc))]
    occl = cand[int(np.argmax([res[n]["occlusion"] for n in cand]))]
    out = {"what": __doc__.split("\n")[0], "freqs_khz": (freqs / 1e3).tolist(), "n_candidates": len(cand),
           "rule_choice": rule, "rule_distance_to_their_receiver_mm": float(np.linalg.norm(P[cand.index(rule)] - rec) * 1e3)}
    for key in ("corr_their_rotation", "corr_fitted_rotation"):
        c = np.array([res[n][key] for n in cand])
        r = res[rule][key]
        out[key] = {"rule": r, "rule_percentile": float((c < r).mean() * 100), "best": float(c.max()),
                    "median": float(np.median(c)), "worst": float(c.min()), "most_occluded": res[occl][key],
                    "their_receiver": res["their_receiver"][key]}
        print(f"{key}: rule {r:.3f} ({out[key]['rule_percentile']:.0f}th percentile), best {c.max():.3f}, median "
              f"{np.median(c):.3f}, most occluded {res[occl][key]:.3f}, their receiver {res['their_receiver'][key]:.3f}",
              flush=True)
    out["candidates"] = {n: dict(res[n], position_m=pos[names.index(n)].tolist()) for n in names}
    fn = paths.RESULTS / "validation" / "vesterholm_rule_test.json"
    fn.write_text(json.dumps(out, indent=1))
    print(fn)


if __name__ == "__main__":
    main()
