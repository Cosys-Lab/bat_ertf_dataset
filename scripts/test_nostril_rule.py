"""Reproducible nostril rule, tested on the calibrated species: among the forward-facing nose candidates, the pair that
are mirror images across the symmetry plane (mismatch < MIRROR_TOL), MIN_SPACING..MAX_SPACING apart, and most
recessed (highest summed ambient occlusion). Prints the rule's pair and its emitter correlation next to the best pair.
Usage: python scripts/test_nostril_rule.py"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import mesh, paths, rotate, symmetry
from fit_calibration import corr, load

MIRROR_TOL = 0.6e-3
MIN_SPACING, MAX_SPACING = 1.0e-3, 3.5e-3


def mirror(p, yaw, roll, y0):
    n = rotate.matrix(yaw, 0, roll) @ np.array([0.0, 1.0, 0.0])
    d = (p - [0, y0 * 1e-3, 0]) @ n
    return p - 2 * d * n


def rule_pairs(pos, ao, sym):
    out = []
    for a, b in itertools.combinations(pos, 2):
        s = np.linalg.norm(pos[a] - pos[b])
        mm = np.linalg.norm(mirror(pos[a], *sym) - pos[b])
        if mm < MIRROR_TOL and MIN_SPACING <= s <= MAX_SPACING:
            out.append((ao[a] + ao[b], a, b, s, mm))
    return sorted(out, reverse=True)


def main():
    for key in ["pdisc", "mmicr"]:
        proj, meta, cand, f, H, ref = load(key)
        fit = json.loads((proj / "fit.json").read_text())
        m = mesh.load(paths.species()[key]["stl"])
        sym, _ = symmetry.fit(m)
        aov = np.load(paths.work_dir() / "occlusion" / f"{key}.npz")["ao"]
        tree = cKDTree(m.vertices)
        pos = {n: np.array(p) for n, p in cand.items() if n.startswith("nose")}
        ao = {n: aov[tree.query(p)[1]] for n, p in pos.items()}
        R = rotate.matrix(*fit["angles"])
        area = meta["patch_area"]
        rs = lambda h: rotate.hemisphere_from_sphere(h.real, R) + 1j * rotate.hemisphere_from_sphere(h.imag, R)
        c = lambda a, b: corr(abs(rs(H[a] * area[a] + H[b] * area[b])), ref["emitter"])
        pairs = rule_pairs(pos, ao, sym)
        print(f"{key}: {len(pairs)} mirror pairs; best calibrated pair {fit['nostrils']['pair']} corr {fit['nostrils']['corr']:.3f}")
        for sc, a, b, s, mm in pairs[:5]:
            print(f"   rule pair {a} {b}: ao {sc:.2f}, spacing {s*1e3:.2f} mm, mirror mismatch {mm*1e3:.2f} mm, "
                  f"midpoint {np.round((pos[a]+pos[b])/2*1e3, 2)} mm, emitter corr {c(a, b):.3f}")


if __name__ == "__main__":
    main()
