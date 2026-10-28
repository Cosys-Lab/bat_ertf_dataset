"""Fit head orientation, ear patches and nostril pair of a pilot run (pilot_calibration.py) to the reference ERTFs.

1. Orientation: yaw/pitch/roll maximizing the mean pattern correlation of the best left and right ear candidates
   (coarse grid, then Nelder-Mead).
2. Nostrils: in that orientation, the pair of nose candidates (equal normal velocity, complex sum) whose emission
   pattern correlates best with the reference emitter.
Usage: python scripts/fit_calibration.py key     -> <work>/runs/pilot2_<key>/fit.json and printed summary"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import numcalc, paths, reference, rotate


def load(key):
    """Candidate sources of the GPU calibration run (calibrate_gpu.py) if present, else of the NumCalc pilot."""
    calib = paths.work_dir() / "calib" / f"{key}.npz"
    if calib.exists():
        d = np.load(calib)
        names = [str(n) for n in d["names"]]
        proj = paths.work_dir() / "calib" / key
        proj.mkdir(exist_ok=True)
        meta = {"patch_area": dict(zip(names, d["area"].tolist()))}
        cand = dict(zip(names, d["pos"].tolist()))
        f, H = d["freqs"], dict(zip(names, d["H"]))
    else:
        proj = paths.work_dir() / "runs" / f"pilot2_{key}"
        meta = json.loads((proj / "project.json").read_text())
        cand = json.loads((proj / "candidates.json").read_text())
        H = {}
        for s in meta["ranges"]:
            f, H[s] = numcalc.read_source(proj, s)
    ref = reference.load(key)
    ki = [int(np.argmin(abs(ref["freqs"] - x))) for x in f]
    return proj, meta, cand, f, H, {k: ref[k][..., ki] for k in ("left", "right", "emitter")}


def corr(sim_hemi_mag, ref_mag):
    return reference.metrics(sim_hemi_mag, ref_mag)["corr"].mean()


def main():
    key = sys.argv[1]
    proj, meta, cand, f, H, ref = load(key)
    ears = {s: [n for n in H if n.startswith(f"ear{s}")] for s in "LR"}
    absH = {n: abs(h) for n, h in H.items()}

    def ear_score(angles, detail=False):
        R = rotate.matrix(*angles)
        best = {}
        for s, side in (("L", "left"), ("R", "right")):
            sc = {n: corr(rotate.hemisphere_from_sphere(absH[n], R), ref[side]) for n in ears[s]}
            best[side] = max(sc.items(), key=lambda x: x[1])
        return best if detail else -(best["left"][1] + best["right"][1]) / 2

    coarse = [(y, p, r) for y in range(-40, 41, 10) for p in range(-80, 31, 10) for r in range(-30, 31, 15)]
    scores = [ear_score(a) for a in coarse]
    a0 = coarse[int(np.argmin(scores))]
    res = minimize(ear_score, a0, method="Nelder-Mead", options={"xatol": 0.5, "fatol": 1e-4, "initial_simplex":
                   np.array(a0) + np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]])})
    ang = res.x
    R = rotate.matrix(*ang)
    best_ears = ear_score(ang, True)
    print(f"{key}: orientation yaw {ang[0]:.1f} pitch {ang[1]:.1f} roll {ang[2]:.1f} deg, "
          f"ear corr {-res.fun:.3f} ({best_ears})")

    nose = [n for n in H if n.startswith("nose")]
    hemi = {n: rotate.hemisphere_from_sphere(H[n].real, R) + 1j * rotate.hemisphere_from_sphere(H[n].imag, R)
            for n in nose}
    # equal velocity on both nostrils: sum pressures weighted by patch area (H is normalized by each patch's Q)
    area = meta["patch_area"]
    pairs = []
    for a, b in itertools.combinations(nose, 2):
        h = hemi[a] * area[a] + hemi[b] * area[b]
        pairs.append((corr(abs(h), ref["emitter"]), a, b))
    singles = [(corr(abs(hemi[a]), ref["emitter"]), a) for a in nose]
    pairs.sort(reverse=True)
    singles.sort(reverse=True)
    pos = {n: np.array(cand[n]) for n in nose}
    print("best nostril pairs (corr, a, b, spacing mm, midpoint mm):")
    for c, a, b in pairs[:10]:
        print(f"  {c:.3f} {a} {b} {np.linalg.norm(pos[a]-pos[b])*1e3:.2f} {np.round((pos[a]+pos[b])/2*1e3, 2)}")
    print("best single patches:", [(round(c, 3), a, np.round(pos[a] * 1e3, 2).tolist()) for c, a in singles[:3]])
    out = {"angles": ang.tolist(), "ear_corr": -res.fun,
           "ears": {s: {"cand": n, "corr": c, "pos": cand[n]} for s, (n, c) in best_ears.items()},
           "nostrils": {"pair": pairs[0][1:], "corr": pairs[0][0], "pos": [cand[pairs[0][1]], cand[pairs[0][2]]]},
           "top_pairs": [(c, a, b) for c, a, b in pairs[:20]]}
    (proj / "fit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
