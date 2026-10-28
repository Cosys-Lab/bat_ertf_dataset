"""Zoomed renders of the nose / mouth region in the symmetry frame (midline drawn), with a mm grid, for placing the
emitter landmarks. Optional points (mesh frame, mm) are drawn on top.
Usage: python scripts/nose_views.py key [half_width_mm] [x y z ...]  -> <work>/landmarks/<key>_nose.png
Environment variable CENTER="x y z" (mesh frame, mm) sets the view centre (default: 3 mm behind the most forward
midline point)."""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import mesh, paths, rotate, symmetry


def render(ax, v, faces, R, c, hw, pts, title):
    w = (v - c) @ R.T
    tri = w[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    sel = (n[:, 2] > 0) & (np.abs(tri[:, :, 0]).max(1) < 1.5 * hw) & (np.abs(tri[:, :, 1]).max(1) < 1.5 * hw)
    o = np.argsort(tri[sel][:, :, 2].mean(1))
    t = tri[sel][o]
    dep = t[:, :, 2].mean(1)
    col = plt.cm.viridis((dep - dep.min()) / (np.ptp(dep) + 1e-12)) * (0.35 + 0.65 * np.clip(n[sel][o] @ [0.3, 0.4, 0.87], 0, 1))[:, None]
    col[:, 3] = 1
    ax.add_collection(PolyCollection(t[:, :, :2] * 1e3, facecolors=col, edgecolors=(0, 0, 0, 0.12), linewidths=0.2))
    for k, p in enumerate(pts):
        q = (p - c) @ R.T * 1e3
        ax.plot(q[0], q[1], "r+", ms=12, mew=2)
        ax.annotate(f"P{k}", q[:2], color="r")
    ax.set_xlim(-hw * 1e3, hw * 1e3)
    ax.set_ylim(-hw * 1e3, hw * 1e3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.5)
    ax.set_title(title, fontsize=9)


def main():
    key = sys.argv[1]
    hw = (float(sys.argv[2]) if len(sys.argv) > 2 else 5.0) * 1e-3
    pts = np.array(sys.argv[3:], float).reshape(-1, 3) * 1e-3 if len(sys.argv) > 3 else np.zeros((0, 3))
    m = mesh.load(paths.species()[key]["stl"])
    (yaw, roll, y0), _ = symmetry.fit(m)
    S = rotate.matrix(yaw, 0, roll)                  # symmetry frame -> mesh frame
    vs = m.vertices @ S                              # vertices in the symmetry frame (y = 0 is the midline + y0)
    near = np.abs(vs[:, 1] - y0 * 1e-3) < 1.5e-3
    tip = vs[near][np.argmax(vs[near][:, 0])]
    c = tip - [3e-3, 0, 0]
    if os.environ.get("CENTER"):
        c = np.array(os.environ["CENTER"].split(), float) * 1e-3 @ S
    ps = pts @ S
    views = {"front (sym. frame: y right, z up)": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
             "front-below 35 deg": None, "top (x right, y up)": np.eye(3), "side (from left: -x right, z up)": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])}
    fig, axs = plt.subplots(1, 4, figsize=(26, 7))
    for ax, (t, R) in zip(axs, views.items()):
        if R is None:
            a = np.deg2rad(35)
            R = views["front (sym. frame: y right, z up)"] @ np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
        render(ax, vs, m.faces, R, c, hw, ps, t)
    fig.suptitle(f"{key}: centre {np.round(c @ S.T * 1e3, 2)} mm (mesh frame); symmetry yaw {yaw:.1f} roll {roll:.1f} y0 {y0:.2f} mm;"
                 f" axes are mm relative to the centre in the symmetry frame")
    fig.tight_layout()
    out = paths.work_dir() / "landmarks"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{key}_nose.png", dpi=65)
    print(out / f"{key}_nose.png", "centre mesh frame mm", np.round(c @ S.T * 1e3, 2), "S=", np.round(S, 4).tolist())


if __name__ == "__main__":
    main()
