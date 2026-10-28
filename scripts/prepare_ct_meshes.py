"""Prepare the original CT meshes (data/meshes/Old, mm): repair (pymeshlab: remove duplicate and non-manifold elements,
close small holes), convert to metres, and move them rigidly into the frame of data/meshes/original with the alignment
of scripts/compare_old_meshes.py, so landmarks and frames carry over.
Usage: python scripts/prepare_ct_meshes.py   -> data/meshes/ct/<key>.stl (+ micro_highres.stl)"""
import json
import sys
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

EXTRA = {"mmicr_highres": ("micro High Res.stl", "mmicr")}


def repair(src, dst):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=50)
    ms.meshing_re_orient_faces_coherently()
    ms.save_current_mesh(str(dst))


def main():
    align = json.loads((paths.RESULTS / "meshes" / "old_vs_original.json").read_text())
    out = paths.ROOT / "data" / "meshes" / "ct"
    out.mkdir(parents=True, exist_ok=True)
    jobs = {k: (v["old"], k) for k, v in align.items()} | EXTRA
    for name, (old, key) in jobs.items():
        dst = out / f"{name}.stl"
        repair(paths.ROOT / "data" / "meshes" / "Old" / old, dst)
        m = trimesh.load(dst, process=True)
        m.apply_scale(1e-3)
        m.apply_transform(np.array(align[key]["transform_old_to_new"]))
        parts = m.split(only_watertight=False)
        m = max(parts, key=lambda p: len(p.faces))
        if m.volume < 0:
            m.invert()
        m.export(dst)
        print(f"{name}: {len(m.faces)} faces, watertight {m.is_watertight}, winding {m.is_winding_consistent}, "
              f"edge mean {m.edges_unique_length.mean()*1e3:.3f} mm, bodies dropped {len(parts)-1}")


if __name__ == "__main__":
    main()
