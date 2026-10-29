"""Dataset meshes from the original CT meshes (data/meshes/Old, mm).

1. Repair (pymeshlab): duplicates, null faces, non-manifold edges and vertices, small holes.
2. Isotropic remeshing (target edge grown until the mesh has at most N_MAX faces) on the CT surface (re-projection and a surface-distance check), target edge length
   L = max(L_MIN, sqrt(A / (0.433 N_MAX))) so the mesh stays within the GPU solver's budget of N_MAX faces.
3. Local repair until clean (closed, 2-manifold, one body, no self-intersections: defect regions are cut out,
   the holes closed and remeshed); meshes that do not pass are not written. Then outward orientation, metres, and the CT frame with its axes relabelled to
   X forward, Y left, Z up (X = CT z, Y = CT x, Z = CT y). This is the frame of the original lab simulations
   (Vanderelst et al. 2010: the M. microtis reference ERTFs were computed in it, calibrate_gpu.py recovers it within 2-5 deg).
4. Deviation from the CT surface (two-sided, sampled) and element statistics to data/meshes/dataset/meshes.json.
Usage: python scripts/remesh_ct.py [key ...]   -> data/meshes/dataset/<Genus_species>.stl"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

L_MIN = 0.27e-3
N_MAX = 58000
C = 343.0
SOURCES = {"pdisc": "phyllo.stl", "mmicr": "micro High Res.stl", "mmacr": "macro.stl", "rrouxi": "rouxi.stl",
           "rform": "formosae.stl", "efusc": "eptesicus.stl", "mdaub": "myotis.stl"}
CT_TO_FRAME = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)      # rows: frame X, Y, Z in CT coordinates
# head pose after the CT frame (v_new = v @ M): P. discolor in the pose of De Mey et al. 2008 (scripts/rotate_pdisc_pose.py)
POSE = {"pdisc": np.array([[0.84626, 0.221636, -0.48448], [-0.138887, 0.969694, 0.201008], [0.514347, -0.102818, 0.851396]])}


def to_frame(m, key):
    """CT scan (mm) -> dataset frame (m), including the head pose of the species."""
    m.apply_scale(1e-3)
    T = np.eye(4)
    T[:3, :3] = CT_TO_FRAME
    m.apply_transform(T)
    if key in POSE:
        m.vertices = m.vertices @ POSE[key]
    return m


def remesh(src, L_mm):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src))
    for f in ("meshing_remove_duplicate_vertices", "meshing_remove_duplicate_faces", "meshing_remove_null_faces"):
        getattr(ms, f)()
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=100)
    ms.meshing_isotropic_explicit_remeshing(iterations=12, targetlen=pymeshlab.PureValue(L_mm), featuredeg=30,
                                            checksurfdist=True, maxsurfdist=pymeshlab.PureValue(0.1 * L_mm))
    prev, stuck = None, 0
    for it in range(25):                                # local repair until clean
        t = state(ms)
        if t == (0, 0, 0, 1, 0):
            break
        stuck = stuck + 1 if t == prev else 0
        prev = t
        bad = defect_faces(ms)
        if bad.any():                                   # cut out the defects plus a few rings of elements
            cut(ms, bad, rings=2 + 2 * stuck)
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=2000)
        ms.set_selection_none()                         # close the holes and remesh the patches
        ms.meshing_close_holes(maxholesize=8000, newfaceselected=True, selfintersection=stuck == 0, refinehole=True)
        ms.apply_selection_dilatation()
        ms.meshing_isotropic_explicit_remeshing(iterations=6, selectedonly=True, targetlen=pymeshlab.PureValue(L_mm),
                                                checksurfdist=False)
    ms.meshing_re_orient_faces_coherently()
    m = ms.current_mesh()
    return trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=False), is_clean(ms)


def defect_faces(ms):
    """Union of self-intersecting faces, faces on non-manifold edges and faces at non-manifold vertices.
    (pymeshlab selection filters replace the selection, so each mask is read separately.)"""
    masks = []
    for f in (ms.compute_selection_by_self_intersections_per_face, ms.compute_selection_by_non_manifold_edges_per_face):
        ms.set_selection_none()
        f()
        masks.append(ms.current_mesh().face_selection_array().copy())
    ms.set_selection_none()
    ms.compute_selection_by_non_manifold_per_vertex()
    ms.compute_selection_transfer_vertex_to_face(inclusive=False)
    masks.append(ms.current_mesh().face_selection_array().copy())
    ms.set_selection_none()
    return np.logical_or.reduce(masks)


def cut(ms, bad, rings):
    """Remove the faces in `bad` grown by `rings` vertex-sharing rings; replaces the current mesh."""
    m = ms.current_mesh()
    V, F = m.vertex_matrix(), m.face_matrix()
    sel = bad.copy()
    for _ in range(rings):
        vs = np.zeros(len(V), bool)
        vs[F[sel].ravel()] = True
        sel = vs[F].any(1)
    F = F[~sel]
    used = np.unique(F)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    ms.add_mesh(pymeshlab.Mesh(V[used], remap[F]))
    ms.meshing_remove_duplicate_vertices()


def state(ms):
    """(non-manifold edges, non-manifold vertices, boundary edges, bodies, self-intersecting faces)."""
    t = ms.get_topological_measures()
    ms.set_selection_none()
    ms.compute_selection_by_self_intersections_per_face()
    si = ms.current_mesh().selected_face_number()
    ms.set_selection_none()
    return (t["non_two_manifold_edges"], t["non_two_manifold_vertices"], t["boundary_edges"],
            t["connected_components_number"], si)


def is_clean(ms):
    """Closed, 2-manifold, one body, no self-intersections."""
    return state(ms) == (0, 0, 0, 1, 0)


def remesh_isolated(src, L_mm):
    """remesh() in a fresh process: pymeshlab keeps state between calls, after which remeshing silently does nothing."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(remesh, str(src), L_mm).result()


def main():
    keys = sys.argv[1:] or list(SOURCES)
    if len(keys) > 1:                                   # pymeshlab keeps state between meshes in one process:
        for key in keys:                                # after one mesh, later remeshing silently does nothing
            subprocess.run([sys.executable, __file__, key], check=False)
        return
    out = paths.ROOT / "data" / "meshes" / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    meta_f = out / "meshes.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    for key in keys:
        name = paths.species()[key]["name"]
        src = paths.ROOT / "data" / "meshes" / "Old" / SOURCES[key]
        ct = trimesh.load(src, process=True)
        ct.apply_scale(1e-3)
        L = max(L_MIN, np.sqrt(ct.area / (0.433 * N_MAX)))
        m, clean = remesh_isolated(src, L * 1e3)
        while len(m.faces) > N_MAX:                     # the remesher overshoots the area estimate
            L *= np.sqrt(len(m.faces) / N_MAX) * 1.01
            m, clean = remesh_isolated(src, L * 1e3)
        if not clean:
            print(f"{key}: NOT CLEAN after repair, not written to the dataset", flush=True)
            continue
        to_frame(m, key)
        T = np.eye(4)
        T[:3, :3] = CT_TO_FRAME
        ct.apply_transform(T)
        if key in POSE:
            ct.vertices = ct.vertices @ POSE[key]
        assert m.is_watertight and m.is_winding_consistent, key
        if m.volume < 0:
            m.invert()
        fn = out / f"{name.replace(' ', '_')}.stl"
        m.export(fn)
        a, b = m.sample(40000), ct.sample(40000)
        d1 = trimesh.proximity.closest_point(ct, a[:8000])[1]
        d2 = trimesh.proximity.closest_point(m, b[:8000])[1]
        e = m.edges_unique_length
        meta[key] = {"species": name, "file": fn.name, "source_ct_mesh": SOURCES[key], "frame": "CT frame, X forward "
                     "(CT z), Y left (CT x), Z up (CT y), metres", "n_faces": int(len(m.faces)),
                     "n_vertices": int(len(m.vertices)), "watertight": bool(m.is_watertight),
                     "winding_consistent": bool(m.is_winding_consistent), "self_intersections": 0, "bodies": 1, "target_edge_mm": L * 1e3,
                     "edge_mm_mean_min_max": [e.mean() * 1e3, e.min() * 1e3, e.max() * 1e3],
                     "f_max_6_per_wavelength_khz": C / (6 * e.mean()) / 1e3, "area_mm2": m.area * 1e6,
                     "area_ct_mm2": ct.area * 1e6, "volume_mm3": m.volume * 1e6 * 1e3,
                     "deviation_to_ct_mm_median_p95_max": [float(np.median(d1) * 1e3), float(np.percentile(d1, 95) * 1e3), float(d1.max() * 1e3)],
                     "deviation_from_ct_mm_median_p95_max": [float(np.median(d2) * 1e3), float(np.percentile(d2, 95) * 1e3), float(d2.max() * 1e3)]}
        r = meta[key]
        print(f"{key}: {r['n_faces']} faces, watertight {r['watertight']}, edge {r['edge_mm_mean_min_max'][0]:.3f} mm "
              f"(f6 {r['f_max_6_per_wavelength_khz']:.0f} kHz), dev to CT median/p95/max {np.round(r['deviation_to_ct_mm_median_p95_max'], 3)} mm, "
              f"CT to mesh {np.round(r['deviation_from_ct_mm_median_p95_max'], 3)} mm", flush=True)
        meta_f.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
