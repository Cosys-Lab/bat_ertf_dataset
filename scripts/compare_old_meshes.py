"""Relate the original CT meshes (data/meshes/Old, mm) to the meshes used so far (data/meshes/original, m):
rigid alignment (ICP from N_START random orientations plus PCA starts, best trimmed distance kept), then the surface distance both ways.
Usage: python scripts/compare_old_meshes.py  -> results/meshes/old_vs_original.json"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import mesh, paths, rotate

OLD = {"pdisc": "phyllo.stl", "mmicr": "micro.stl", "mmacr": "macro.stl", "rrouxi": "rouxi.stl",
       "rform": "formosae.stl", "efusc": "eptesicus.stl", "mdaub": "myotis.stl"}


N_START = 48


def best_alignment(old, new, seed=0):
    rng = np.random.default_rng(seed)
    src = old.sample(3000)
    tree = cKDTree(new.sample(20000))
    c_old, c_new = old.centroid, new.centroid
    best = (np.inf, None)
    starts = [trimesh.registration.mesh_other(old, new, samples=2000, scale=False, icp_first=10, icp_final=10)[0]]
    for _ in range(N_START):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        T0 = trimesh.transformations.quaternion_matrix(q)
        T0[:3, 3] = c_new - T0[:3, :3] @ c_old
        starts.append(T0)
    for T0 in starts:
        T, _, _ = trimesh.registration.icp(src, tree.data, initial=T0, threshold=1e-9, max_iterations=60, scale=False)
        d = tree.query(trimesh.transform_points(src, T))[0]
        cost = np.mean(np.sort(d)[: int(0.9 * len(d))])
        if cost < best[0]:
            best = (cost, T)
    return best[1]


def main():
    out = {}
    for key, f in OLD.items():
        old = trimesh.load(paths.ROOT / "data" / "meshes" / "Old" / f, process=True)
        old.apply_scale(1e-3)
        new = mesh.load(paths.species()[key]["stl"])
        T = best_alignment(old, new)
        o = old.copy().apply_transform(T)
        po = o.sample(20000, return_index=False)
        pn = new.sample(20000, return_index=False)
        d1 = cKDTree(new.vertices).query(po)[0]
        d2 = cKDTree(o.vertices).query(pn)[0]
        R = T[:3, :3]
        ypr = np.degrees([np.arctan2(R[1, 0], R[0, 0]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[2, 1], R[2, 2])])
        out[key] = {"old": f, "faces_old": len(old.faces), "faces_new": len(new.faces),
                    "area_ratio_new_old": new.area / old.area, "dist_old_to_new_mm_median_p95": [np.median(d1) * 1e3, np.percentile(d1, 95) * 1e3],
                    "dist_new_to_old_mm_median_p95": [np.median(d2) * 1e3, np.percentile(d2, 95) * 1e3],
                    "transform_old_to_new": T.tolist(), "yaw_pitch_roll_deg_approx": ypr.tolist()}
        print(f"{key}: {f} {len(old.faces)} -> {len(new.faces)} faces, area new/old {new.area/old.area:.3f}, "
              f"dist old->new median {np.median(d1)*1e3:.3f} p95 {np.percentile(d1,95)*1e3:.3f} mm, "
              f"new->old median {np.median(d2)*1e3:.3f} p95 {np.percentile(d2,95)*1e3:.3f} mm, rot ~ {np.round(ypr,1)}", flush=True)
    d = paths.RESULTS / "meshes"
    d.mkdir(parents=True, exist_ok=True)
    (d / "old_vs_original.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
