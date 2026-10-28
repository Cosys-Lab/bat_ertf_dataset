"""Pilot runs for the two species with reference ERTFs, to calibrate head orientation and source positions.

Every candidate source patch is simulated as a separate NumCalc source on the full-sphere grid at five frequencies:
- ears: the four most occluded spots per side (landmark_candidates.py),
- nostrils: forward-facing surface points within NOSE_RADIUS of a nose centre, at least 0.8 mm apart.
Because the BEM problem is linear, any combination of candidate patches (e.g. a nostril pair) is a complex sum.
Usage: python scripts/pilot_calibration.py key   (then scripts/fit_calibration.py key)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import mesh, numcalc, paths, sources
from landmark_candidates import clusters

PILOT = {"pdisc": {"freqs": [30e3, 45e3, 60e3, 75e3, 90e3], "nose": [18.5e-3, 0.0, 1.0e-3]},
         "mmicr": {"freqs": [50e3, 75e3, 100e3, 125e3, 150e3], "nose": [7.0e-3, -2.5e-3, -2.5e-3]}}
PILOT["mmicr_ct"] = PILOT["mmicr"]
NOSE_RADIUS = 3.0e-3
SPACING = 0.8e-3


def nose_candidates(m, center, forward):
    v, n = m.vertices, m.vertex_normals
    ok = (np.linalg.norm(v - center, axis=1) < NOSE_RADIUS) & (n @ forward > 0.2)
    idx = np.where(ok)[0]
    out = [idx[np.argmin(np.linalg.norm(v[idx] - center, axis=1))]]
    while True:                                    # farthest-point sampling at >= SPACING
        d = np.min(np.linalg.norm(v[idx][:, None] - v[out][None], axis=2), axis=1)
        if d.max() < SPACING:
            break
        out.append(idx[np.argmax(d)])
    return v[out]


def main():
    key = sys.argv[1]
    cfg = PILOT[key]
    sp = paths.species()[key]
    m = mesh.load(sp["stl"])
    ao = np.load(paths.work_dir() / "occlusion" / f"{key}.npz")["ao"]
    v = m.vertices
    xr = np.ptp(v[:, 0])
    cand = {}
    for side, mask in {"L": (v[:, 1] > 2e-3) & (v[:, 0] < v[:, 0].max() - 0.3 * xr),
                       "R": (v[:, 1] < -2e-3) & (v[:, 0] < v[:, 0].max() - 0.3 * xr)}.items():
        for k, i in enumerate(clusters(v, ao, mask)):
            cand[f"ear{side}{k}"] = (v[i], 0.6e-3)
    nose = np.array(cfg["nose"])
    fwd = np.array([1.0, 0, 0])
    for k, p in enumerate(nose_candidates(m, nose, fwd)):
        cand[f"nose{k:02d}"] = (p, 0.4e-3)
    patches, used = {}, set()
    for name, (p, r) in cand.items():
        f = sources.disc_patch(m, p, r)
        f = np.array([x for x in f if x not in used])
        if len(f) == 0:
            continue
        used.update(f.tolist())
        patches[name] = [f]
    proj = paths.work_dir() / "runs" / f"pilot2_{key}"
    meta = numcalc.write_project(proj, m, patches, cfg["freqs"], grid_name="sphere")
    (proj / "candidates.json").write_text(json.dumps({k: list(map(float, cand[k][0])) for k in patches}))
    print(f"{key}: {len(patches)} candidate sources x {len(cfg['freqs'])} frequencies", flush=True)
    ok, n = numcalc.run_steps(numcalc.steps(proj), paths.local()["numcalc"], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    print(f"{key}: {ok}/{n} steps ok")


if __name__ == "__main__":
    main()
