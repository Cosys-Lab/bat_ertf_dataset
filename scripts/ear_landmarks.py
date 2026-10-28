"""Ear canal entrance rule for species without reference ERTFs. Ear region: |y| > 4 mm and the back 55 % of the head
(x < x_min + 0.55 x_range). Per side, the four most occluded spots (landmark_candidates.clusters); the left/right pair
that is the best mirror image across the symmetry plane (symmetry.fit) is chosen, ties broken by occlusion.
Usage: python scripts/ear_landmarks.py key [...]  -> prints ear_left / ear_right in metres (mesh frame)"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import mesh, paths, symmetry
from landmark_candidates import clusters
from test_nostril_rule import mirror


def ear_candidates(m, ao):
    v = m.vertices
    back = v[:, 0] < v[:, 0].min() + 0.55 * np.ptp(v[:, 0])
    return {s: v[clusters(v, ao, mask & back)] for s, mask in (("L", v[:, 1] > 4e-3), ("R", v[:, 1] < -4e-3))}


def ears(m, ao, sym):
    c = ear_candidates(m, ao)
    pairs = [(np.linalg.norm(mirror(l, *sym) - r) + 1e-4 * (i + j), i, j)
             for (i, l), (j, r) in itertools.product(enumerate(c["L"]), enumerate(c["R"]))]
    _, i, j = min(pairs)
    return c["L"][i], c["R"][j], i, j


if __name__ == "__main__":
    for key in sys.argv[1:]:
        m = mesh.load(paths.species()[key]["stl"])
        ao = np.load(paths.work_dir() / "occlusion" / f"{key}.npz")["ao"]
        sym, _ = symmetry.fit(m)
        l, r, i, j = ears(m, ao, sym)
        print(f"# {key}: candidates L{i}, R{j}, mirror mismatch {np.linalg.norm(mirror(l, *sym) - r)*1e3:.2f} mm")
        print(f"ear_left = [{l[0]:.6e}, {l[1]:.6e}, {l[2]:.6e}]\near_right = [{r[0]:.6e}, {r[1]:.6e}, {r[2]:.6e}]")
