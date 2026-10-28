"""Mesh loading, cleanup and occlusion.

Meshes are in metres, frame X forward (snout), Y left, Z up (checked for all seven species).
"""
import numpy as np
import trimesh

from . import paths


def load(stl_name):
    """Load an STL and keep only its largest connected body (drops 1-2 face slivers). Paths are relative to
    data/meshes/original (CT meshes: "../ct/<key>.stl"). Closed meshes must be consistently wound; nearly closed CT
    meshes (a few non-manifold edges or pinholes) get every face oriented outward by orient_outward()."""
    m = trimesh.load(paths.MESH_DIR / stl_name, process=True)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: len(p.faces))
    m.remove_unreferenced_vertices()
    if m.is_watertight and m.is_winding_consistent:
        if m.volume < 0:
            m.invert()
    else:
        orient_outward(m)
    return m


def orient_outward(m, eps=1e-6):
    """Flip every face whose normal points into the body: a ray from just off the face along its normal crosses the
    surface an odd number of times if it starts inside. Robust to pinholes and non-manifold edges."""
    c = m.triangles_center + m.face_normals * eps
    hits = m.ray.intersects_location(c, m.face_normals, multiple_hits=True)[1]
    n = np.bincount(hits, minlength=len(m.faces))
    flip = n % 2 == 1
    if flip.any():
        f = m.faces.copy()
        f[flip] = f[flip][:, ::-1]
        m.faces = f
    return int(flip.sum())


def edge_stats(m):
    e = m.edges_unique_length
    return {"n_faces": len(m.faces), "n_vertices": len(m.vertices), "edge_mean": e.mean(),
            "edge_max": e.max(), "area": m.area, "volume": m.volume}


def f_max_six_per_wavelength(m, c=343.0):
    """Highest frequency with six mean-size elements per wavelength."""
    return c / (6 * m.edges_unique_length.mean())


def ambient_occlusion(m, n_rays=96, seed=0):
    """Fraction of the outward hemisphere that is blocked, per vertex (0 = open, 1 = fully enclosed)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_rays, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = m.vertex_normals
    o = m.vertices + n * 2e-5
    hit = np.zeros(len(m.vertices))
    cnt = np.zeros(len(m.vertices))
    for k in range(n_rays):
        dk = np.where((n @ d[k])[:, None] >= 0, d[k], -d[k])          # flip into the outward hemisphere
        h = m.ray.intersects_any(o, np.broadcast_to(dk, o.shape))
        hit += h
        cnt += 1
    return hit / cnt
