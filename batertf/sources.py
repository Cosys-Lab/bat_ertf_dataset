"""Source patches: groups of mesh faces that vibrate in a NumCalc run.

An ear is a disc-shaped patch at the ear canal entrance (by reciprocity, the far-field pressure it radiates equals the
pressure averaged over the patch for a far-field source). The nasal emitter is two in-phase patches, one per nostril.
The oral emitter is a disc-shaped piston patch at the mouth.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components


def disc_patch(m, center, radius):
    """Faces connected to the face nearest `center` whose centroids lie within `radius` of it (geodesic growth,
    so a fold of skin on the other side of a narrow gap is not included)."""
    c = m.triangles_center
    seed = int(np.argmin(np.linalg.norm(c - center, axis=1)))
    inside = np.linalg.norm(c - c[seed], axis=1) <= radius
    adj = m.face_adjacency
    keep = adj[inside[adj[:, 0]] & inside[adj[:, 1]]]
    idx = np.where(inside)[0]
    remap = -np.ones(len(c), int)
    remap[idx] = np.arange(len(idx))
    import scipy.sparse as sp
    g = sp.coo_matrix((np.ones(len(keep)), (remap[keep[:, 0]], remap[keep[:, 1]])), shape=(len(idx), len(idx)))
    _, lab = connected_components(g, directed=False)
    return idx[lab == lab[remap[seed]]]


def patch_info(m, faces):
    a = m.area_faces[faces]
    return {"n_faces": len(faces), "area": float(a.sum()),
            "centroid": (m.triangles_center[faces] * a[:, None]).sum(0) / a.sum(),
            "normal": (m.face_normals[faces] * a[:, None]).sum(0) / np.linalg.norm((m.face_normals[faces] * a[:, None]).sum(0))}
