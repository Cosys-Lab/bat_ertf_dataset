"""Turn points picked in the front view of nose_views.py into surface points: rays along -X of the symmetry frame,
first hit on the mesh. Prints the hits in the mesh frame (mm) and redraws nose_views with them.
Usage: CENTER="x y z" python scripts/pick_front.py key half_width_mm  yr zr [yr zr ...]
(yr, zr: horizontal and vertical position in the front view, mm relative to CENTER, as on the nose_views axes)"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import mesh, paths, rotate, symmetry


def main():
    key, hw = sys.argv[1], sys.argv[2]
    picks = np.array(sys.argv[3:], float).reshape(-1, 2) * 1e-3
    m = mesh.load(paths.species()[key]["stl"])
    (yaw, roll, _), _ = symmetry.fit(m)
    S = rotate.matrix(yaw, 0, roll)
    c = np.array(os.environ["CENTER"].split(), float) * 1e-3 @ S          # centre in the symmetry frame
    hits = []
    for yr, zr in picks:
        o = S @ (c + [0.05, yr, zr])                                        # 5 cm in front, in the mesh frame
        d = S @ np.array([-1.0, 0, 0])
        loc, _, _ = m.ray.intersects_location([o], [d])
        p = loc[np.argmin(np.linalg.norm(loc - o, axis=1))]
        hits.append(p)
        print(f"pick ({yr*1e3:.2f}, {zr*1e3:.2f}) -> mesh frame [{p[0]*1e3:.3f}, {p[1]*1e3:.3f}, {p[2]*1e3:.3f}] mm")
    args = [str(x) for p in hits for x in np.round(np.array(p) * 1e3, 3)]
    subprocess.run([sys.executable, str(Path(__file__).parent / "nose_views.py"), key, hw] + args, check=True,
                   stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
