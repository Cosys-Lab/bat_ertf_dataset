"""Calibration runs on the GPU solver for the species with reference ERTFs (replaces the NumCalc pilot).

Candidate source patches as in pilot_calibration.py (four most occluded spots per ear; nose points at least
SPACING apart on the forward-facing noseleaf base), all solved as separate right-hand sides on the full-sphere grid
at N_FREQ frequencies spread over the reference band.
Usage: python scripts/calibrate_gpu.py key   -> <work>/calib/<key>.npz (then scripts/fit_calibration.py key)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_calibration as pc
from batertf import gpubem, grid, mesh, paths, reference, sources
from landmark_candidates import clusters

N_FREQ = 10
pc.SPACING = 0.6e-3
pc.NOSE_RADIUS = 3.5e-3


def main():
    key = sys.argv[1]
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
    for k, p in enumerate(pc.nose_candidates(m, np.array(pc.PILOT[key]["nose"]), np.array([1.0, 0, 0]))):
        cand[f"nose{k:02d}"] = (p, 0.4e-3)
    patches, used = {}, set()
    for name, (p, r) in cand.items():
        f = np.array([x for x in sources.disc_patch(m, p, r) if x not in used])
        if len(f):
            used.update(f.tolist())
            patches[name] = [f]
    rf = reference.load(key)["freqs"]
    freqs = rf[np.linspace(0, len(rf) - 1, N_FREQ + 2).round().astype(int)[1:-1]]
    print(f"{key}: {len(patches)} candidate sources, {len(freqs)} frequencies {freqs/1e3} kHz", flush=True)
    pts, back = grid.unique_points("sphere")
    H = gpubem.solve(m, patches, freqs, pts, log=lambda s: print(s, flush=True))
    out = paths.work_dir() / "calib"
    out.mkdir(exist_ok=True)
    np.savez(out / f"{key}.npz", freqs=freqs, names=np.array(list(patches)),
             H=np.stack([H[n][back] for n in patches]).astype(np.complex64),
             area=np.array([m.area_faces[patches[n][0]].sum() for n in patches]),
             pos=np.array([cand[n][0] for n in patches]))
    print("saved", out / f"{key}.npz")


if __name__ == "__main__":
    main()
