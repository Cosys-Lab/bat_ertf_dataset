"""Zoomed renders of an ear region looking along the mean surface normal of a given point (into the concha),
for anatomical placement of the ear canal entrance. Optional extra points (mm, CT frame) are drawn.
Usage: python scripts/ear_views.py key side half_width_mm x y z [x y z ...]  -> <work>/landmarks/<key>_ear_<side>.png
(the first point is the view centre)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import mesh, paths, sources
from nose_views import render


def frame_from(n):
    z = n / np.linalg.norm(n)
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z])


def main():
    key, side, hw = sys.argv[1], sys.argv[2], float(sys.argv[3]) * 1e-3
    pts = np.array(sys.argv[4:], float).reshape(-1, 3) * 1e-3
    m = mesh.load(paths.species()[key]["stl"])
    c = pts[0]
    f = sources.disc_patch(m, c, 2.5e-3)
    n = (m.face_normals[f] * m.area_faces[f][:, None]).sum(0)
    lat = np.array([0, 1.0 if side == "left" else -1.0, 0])
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (t, d) in zip(axs, {"along local normal": n, "from the side (lateral)": lat,
                                "from front-lateral 45 deg": lat + np.array([1.0, 0, 0])}.items()):
        R = frame_from(d)
        render(ax, m.vertices, m.faces, R, c, hw, pts, t)
    fig.suptitle(f"{key} {side} ear around {np.round(c*1e3, 2)} mm (P0 = centre)")
    fig.tight_layout()
    out = paths.work_dir() / "landmarks" / f"{key}_ear_{side}.png"
    fig.savefig(out, dpi=65)
    print(out)


if __name__ == "__main__":
    main()
