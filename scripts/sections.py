"""Draw planar cross-sections of a mesh through a point, to see ear canals and nostrils.
Usage: python scripts/sections.py key x y z [half_width_mm] [extra points x y z ...]  (mm)
Draws the x=, y= and z= sections through (x, y, z), zoomed to +-half_width."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import mesh, paths


def main():
    key = sys.argv[1]
    p = np.array(sys.argv[2:5], float) * 1e-3
    hw = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
    extra = np.array(sys.argv[6:], float).reshape(-1, 3) * 1e-3 if len(sys.argv) > 6 else np.zeros((0, 3))
    m = mesh.load(paths.species()[key]["stl"])
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (ax_n, (i, j)) in zip(axs, {"x": (1, 2), "y": (0, 2), "z": (0, 1)}.items()):
        n = np.eye(3)["xyz".index(ax_n)]
        sec = m.section(plane_origin=p, plane_normal=n)
        if sec is not None:
            for e in sec.discrete:
                ax.plot(e[:, i] * 1e3, e[:, j] * 1e3, "k-", lw=0.8)
        ax.plot(p[i] * 1e3, p[j] * 1e3, "r+", ms=14, mew=2)
        for k, q in enumerate(extra):
            ax.plot(q[i] * 1e3, q[j] * 1e3, "b+", ms=10, mew=2)
            ax.annotate(f"X{k}", (q[i] * 1e3, q[j] * 1e3), color="b")
        ax.set_xlim(p[i] * 1e3 - hw, p[i] * 1e3 + hw)
        ax.set_ylim(p[j] * 1e3 - hw, p[j] * 1e3 + hw)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("xyz"[i] + " [mm]")
        ax.set_ylabel("xyz"[j] + " [mm]")
        ax.set_title(f"section {ax_n} = {p['xyz'.index(ax_n)]*1e3:.2f} mm")
    fig.suptitle(f"{key} sections through {np.round(p*1e3, 2)} mm")
    fig.tight_layout()
    out = paths.work_dir() / "occlusion" / f"{key}_sec.png"
    fig.savefig(out, dpi=70)
    print(out)


if __name__ == "__main__":
    main()
