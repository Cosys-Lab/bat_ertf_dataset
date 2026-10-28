"""Compute per-vertex ambient occlusion for every species and render it in six views with mm axes.
Used to locate the ear canal entrances and nostrils (the most occluded spots in the ear and nose regions).
Output: <work>/occlusion/<key>.npz and <key>.png
Usage: python scripts/occlusion_maps.py [keys...]"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import mesh, paths

VIEWS = {"front (+x toward viewer; y right, z up)": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
         "left side (+y toward viewer; -x right, z up)": [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
         "right side (-y toward viewer; x right, z up)": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
         "top (+z toward viewer; x right, y up)": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "front-left oblique": None, "front-right oblique": None}


def rot_oblique(sign):
    a = np.deg2rad(45 * sign)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return np.array(VIEWS["front (+x toward viewer; y right, z up)"]) @ rz


def draw(ax, m, R, val, title, marks=None):
    R = np.asarray(R, float)
    v = m.vertices @ R.T
    tri = v[m.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vis = n[:, 2] > 0
    o = np.argsort(tri[vis][:, :, 2].mean(1))
    fv = val[m.faces].mean(1)[vis][o]
    shade = np.clip((n[vis][o] / np.linalg.norm(n[vis][o], axis=1, keepdims=True)) @ [0.3, 0.4, 0.87], 0, 1)
    col = plt.cm.inferno(fv) * (0.45 + 0.55 * shade)[:, None]
    col[:, 3] = 1
    ax.add_collection(PolyCollection(tri[vis][o][:, :, :2] * 1e3, facecolors=col, edgecolors="none"))
    if marks is not None:
        for name, p in marks.items():
            q = np.asarray(p) @ R.T * 1e3
            ax.plot(q[0], q[1], "c+", ms=10, mew=2)
            ax.annotate(name, q[:2], color="c", fontsize=7)
    ax.set_xlim(v[:, 0].min() * 1e3, v[:, 0].max() * 1e3)
    ax.set_ylim(v[:, 1].min() * 1e3, v[:, 1].max() * 1e3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def main():
    sp = paths.species()
    keys = sys.argv[1:] or list(sp)
    out = paths.work_dir() / "occlusion"
    out.mkdir(exist_ok=True)
    for key in keys:
        m = mesh.load(sp[key]["stl"])
        f = out / f"{key}.npz"
        if f.exists():
            ao = np.load(f)["ao"]
        else:
            ao = mesh.ambient_occlusion(m)
            np.savez(f, ao=ao, vertices=m.vertices)
        fig, axs = plt.subplots(2, 3, figsize=(18, 11))
        for ax, (t, R) in zip(axs.ravel(), VIEWS.items()):
            if R is None:
                R = rot_oblique(-1 if "left" in t else 1)
            draw(ax, m, R, ao, t + " [mm]")
        fig.suptitle(f"{sp[key]['name']}: ambient occlusion (bright = enclosed)")
        fig.tight_layout()
        fig.savefig(out / f"{key}.png", dpi=80)
        plt.close(fig)
        print(key, "done", flush=True)


if __name__ == "__main__":
    main()
