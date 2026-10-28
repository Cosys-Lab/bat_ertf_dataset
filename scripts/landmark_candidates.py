"""Propose ear canal and nostril landmarks from the occlusion maps and render zoomed views to check them.
Candidates: clusters of the most occluded vertices, per region (left ear y>0, right ear y<0, nose = front 25 %).
Usage: python scripts/landmark_candidates.py key [x y z ...]   (extra points in mm are drawn for checking)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import mesh, paths
from occlusion_maps import draw, rot_oblique


def clusters(v, ao, mask, n=4, sep=1.5e-3):
    """n local maxima of occlusion inside mask, at least sep apart."""
    idx = np.where(mask)[0][np.argsort(-ao[mask])]
    out = []
    for i in idx:
        if all(np.linalg.norm(v[i] - v[j]) > sep for j in out):
            out.append(i)
        if len(out) == n:
            break
    return out


def main():
    key = sys.argv[1]
    sp = paths.species()[key]
    m = mesh.load(sp["stl"])
    ao = np.load(paths.work_dir() / "occlusion" / f"{key}.npz")["ao"]
    v = m.vertices
    xr = v[:, 0].max() - v[:, 0].min()
    regions = {"L": (v[:, 1] > 2e-3) & (v[:, 0] < v[:, 0].max() - 0.3 * xr),
               "R": (v[:, 1] < -2e-3) & (v[:, 0] < v[:, 0].max() - 0.3 * xr),
               "N": (v[:, 0] > v[:, 0].max() - 0.25 * xr) & (np.abs(v[:, 1]) < 4e-3)}
    marks = {}
    for r, mask in regions.items():
        for k, i in enumerate(clusters(v, ao, mask)):
            marks[f"{r}{k}"] = v[i]
            print(f"{r}{k}: [{v[i][0]*1e3:7.2f}, {v[i][1]*1e3:7.2f}, {v[i][2]*1e3:7.2f}] mm  ao={ao[i]:.2f}")
    extra = np.array(sys.argv[2:], float).reshape(-1, 3) * 1e-3 if len(sys.argv) > 2 else []
    for k, p in enumerate(extra):
        marks[f"X{k}"] = p
    views = {"front": [[0, 1, 0], [0, 0, 1], [1, 0, 0]], "left side": [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
             "right side": [[1, 0, 0], [0, 0, 1], [0, -1, 0]], "top": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             "front-left": rot_oblique(-1), "front-right": rot_oblique(1)}
    fig, axs = plt.subplots(2, 3, figsize=(20, 13))
    for ax, (t, R) in zip(axs.ravel(), views.items()):
        draw(ax, m, R, ao, t + " [mm]", marks)
    fig.suptitle(f"{sp['name']}: landmark candidates")
    fig.tight_layout()
    out = paths.work_dir() / "occlusion" / f"{key}_cand.png"
    fig.savefig(out, dpi=75)
    print(out)


if __name__ == "__main__":
    main()
