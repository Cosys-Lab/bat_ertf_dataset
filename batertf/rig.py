"""Skeletons and skinning weights for the pinnae and noseleaves of the dataset meshes, so they can be deformed
parametrically and the deformed mesh fed to the BEM (docs/rigging.md).

Frame: the dataset CT frame (X forward, Y left, Z up, metres). Faces are never changed by a deformation, so source
patches and landmarks map by index.

Pipeline (build_rig):
1. Head core: the solid is voxelized and morphologically opened with a ball of radius `open_radius_mm`. Thin
   structures (pinnae, tragus, noseleaf) vanish from the opened solid; `dcore` = distance of every vertex from it.
2. Segmentation: vertices with dcore > `seed_mm` split into connected components (mesh edges). The pinna of a side
   is the largest component within `ear_search_mm` of that ear landmark, plus smaller ones on the same side within
   `tragus_search_mm` (tragus, antitragus); the noseleaf is the largest component near `nose_centre` with its
   centroid within `midline_mm` of the midline. Overrides per species in config/rig.toml (include / exclude points).
3. Appendage weight h: harmonic (edge-length-weighted graph Laplacian) between 1 on the appendage where dcore >=
   <pinna|noseleaf>_band_mm[1] and 0 on the head (dcore <= band[0]), on other thin structures, on other appendages
   and on the snout part cut off below a noseleaf; then smoothstep, so the surface blends continuously into the head.
4. Chain coordinate t: geodesic (Dijkstra) distance from the h = 0.5 iso-line at the appendage base, normalised by
   its 99th percentile on the appendage. Joints are centroids of iso-t bands (t = 0: base ring centroid, t_k,
   t = 1: tip); a bone runs from joint k to joint k + 1, base parented to the fixed head root.
5. Bone weights: h times smooth partition-of-unity hats in t (blend half-width `blend`); the head root gets 1 - h;
   then Gaussian spatial smoothing (`smooth_mm`) so both skins of a thin sheet move together.

Deformation: forward kinematics from per-joint rotations in degrees about named axes, dual-quaternion skinning.
Joint axes (rest pose, CT frame), with d the bone direction (joint -> next joint):
- flex: about -Y made perpendicular to d. Positive turns the bone from +X toward +Z and from +Z toward -X, i.e. an
  upright pinna or noseleaf tilts backward, a forward-pointing one (M. microtis pinnae, noseleaves of M. microtis and
  M. macrophyllum) tilts up. The same axis and sign on both sides (mirror-symmetric).
- abduct: about cross(d, n), n = lateral outward direction (+Y left, -Y right) made perpendicular to d. Positive
  moves the tip outward (away from the midline). Noseleaf: n = +Y, positive tilts it to the bat's left.
- twist: about side * d (left/noseleaf +d, right -d), mirror-symmetric.
Rotation at a joint: R = R_flex R_abduct R_twist (twist first), about the joint, in the frame of the parent bone.

Usage:
    rig = load_rig("mmicr")
    m2 = rig.deform({"pinna_left.base": {"flex": 20}, "pinna_left.distal": {"flex": 30}})
    lm2 = rig.deform_landmarks(pose)     # ear / nostril / mouth landmarks, following the surface
    rig.check(m2)                        # watertight, self-intersections, inverted faces, edge stretch
"""
import json
import tomllib
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy import ndimage
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve

from . import mesh, paths

RIG_DIR = paths.RESULTS / "rig"
CONFIG = paths.ROOT / "config" / "rig.toml"

DEFAULTS = {
    "voxel_mm": 0.1,            # voxel pitch for the head core
    "open_radius_mm": 1.0,      # opening radius: structures thinner than about 2x this are appendage candidates
    "seed_mm": 0.5,             # dcore threshold of the appendage components
    "pinna_band_mm": [0.1, 3.5],     # transition band in dcore: weight 0 below, 1 above (harmonic in between)
    "noseleaf_band_mm": [0.1, 1.5],  # (narrower: the nostrils just below the noseleaf base must stay fixed)
    "ear_search_mm": 4.0,       # the pinna component must come this close to the ear landmark
    "tragus_search_mm": 4.5,    # further components on the same side this close to the ear landmark join the pinna
    "min_component": 20,        # smaller components are ignored
    "nose_search_mm": 6.0,      # the noseleaf component must come this close to nose_centre
    "midline_mm": 2.5,          # |y| of the noseleaf component centroid
    "noseleaf_cut_mm": 0.5,     # the noseleaf starts this far above the nostril midpoint, along nostrils -> leaf tip
    "noseleaf_ramp_mm": 1.0,    # ... and is fully rigid from this distance to the dropped (fixed) part on
    "pinna_joints": [0.35, 0.7],     # chain coordinate t of the inner joints (3 bones)
    "noseleaf_joints": [0.5],        # (2 bones)
    "blend": 0.2,               # half-width (in t) of the bone-to-bone blend at a joint
    "tip_quantile": 99.0,       # t = 1 at this percentile of the geodesic distance on the appendage
    "smooth_mm": 0.4,           # Gaussian (sigma) spatial smoothing of the weights, so both skins of a thin sheet
                                # (pinna, noseleaf: 0.2-0.4 mm) get the same transform (0: off)
}
AXES = ("yaw", "flex", "abduct", "twist")


def yaw_axis(name):
    """Head-fixed yaw axis of a joint (about the vertical Z axis of the CT frame, through the joint). Positive yaw
    turns a pinna and its opening forward (left: about -Z, right: about +Z, mirror-symmetric) and the noseleaf to the
    bat's left (about +Z). Unlike twist, which turns a bone about its own axis, yaw is the same turn for any bone
    direction, e.g. also for the forward-pointing pinnae of M. microtis."""
    if name.startswith("pinna_left"):
        return np.array([0, 0, -1.0])
    if name.startswith("pinna_right"):
        return np.array([0, 0, 1.0])
    return np.array([0, 0, 1.0])
STRETCH_LIMITS = (0.25, 2.0)    # valid deformation: edge length ratio deformed / rest within these limits
STRETCH_MIN_EDGE = 0.5          # ... over edges at least this fraction of the mean rest edge length
JOINT_NAMES = {2: ["base", "distal"], 3: ["base", "mid", "distal"], 4: ["base", "mid", "distal", "tip"]}


# ----------------------------------------------------------------------------------------------------------- config

def config(key):
    """DEFAULTS, updated by [defaults] and [species.<key>] of config/rig.toml (if present). Appendage-level overrides
    live in [species.<key>.<appendage>] (include / exclude points in mm, enabled = false)."""
    cfg = json.loads(json.dumps(DEFAULTS))
    raw = tomllib.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    cfg.update(raw.get("defaults", {}))
    s = raw.get("species", {}).get(key, {})
    cfg.update({k: v for k, v in s.items() if not isinstance(v, dict)})
    cfg["appendages"] = {k: v for k, v in s.items() if isinstance(v, dict)}
    return cfg


def appendage_names(sp_cfg):
    names = ["pinna_left", "pinna_right"]
    if sp_cfg.get("emitter") == "nasal":
        names.append("noseleaf")
    return names


# ----------------------------------------------------------------------------------------------------- geometry

def edge_graph(m, mask=None):
    """Sparse symmetric graph of mesh edges weighted by length (optionally only edges inside a vertex mask)."""
    e = m.edges_unique
    w = m.edges_unique_length
    if mask is not None:
        k = mask[e[:, 0]] & mask[e[:, 1]]
        e, w = e[k], w[k]
    n = len(m.vertices)
    return sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()


def core_distance(m, open_radius=1.0e-3, pitch=0.1e-3):
    """Distance of each vertex from the morphological opening (radius open_radius) of the solid [m]."""
    vg = trimesh.voxel.creation.voxelize(m, pitch).fill()
    pad = int(np.ceil(open_radius / pitch)) + 3
    M = np.pad(vg.matrix, pad)
    eroded = ndimage.distance_transform_edt(M) * pitch > open_radius
    opened = ndimage.distance_transform_edt(~eroded) * pitch <= open_radius
    dcore = ndimage.distance_transform_edt(~opened) * pitch
    ijk = (m.vertices - vg.transform[:3, 3]) / pitch + pad
    return ndimage.map_coordinates(dcore, ijk.T, order=1)


def components(m, mask, min_size):
    g = edge_graph(m, mask)
    _, lab = connected_components(g, directed=False)
    lab = np.where(mask, lab, -1)
    out = []
    for c in np.unique(lab[mask]):
        idx = np.where(lab == c)[0]
        if len(idx) >= min_size:
            out.append(idx)
    return out


def _min_dist(m, idx, p):
    return float(np.linalg.norm(m.vertices[idx] - np.asarray(p), axis=1).min())


def segment(m, dcore, sp_cfg, cfg):
    """Seed vertex sets (dcore > seed_mm) of each appendage, and the list of all thin components."""
    comps = components(m, dcore > cfg["seed_mm"] * 1e-3, cfg["min_component"])
    lm = sp_cfg["landmarks"]
    used = set()
    seeds, notes, cut = {}, {}, {}
    for name in appendage_names(sp_cfg):
        ov = cfg["appendages"].get(name, {})
        if ov.get("enabled", True) is False:
            continue
        chosen = []
        if name.startswith("pinna"):
            side = 1 if name.endswith("left") else -1
            ear = np.array(lm["ear_left" if side > 0 else "ear_right"])
            cand = [(len(c), i) for i, c in enumerate(comps) if i not in used
                    and side * m.vertices[c, 1].mean() > 0 and _min_dist(m, c, ear) < cfg["ear_search_mm"] * 1e-3]
            if cand:
                main = max(cand)[1]
                chosen = [main] + [i for i, c in enumerate(comps) if i != main and i not in used
                                   and side * m.vertices[c, 1].mean() > 0
                                   and _min_dist(m, c, ear) < cfg["tragus_search_mm"] * 1e-3]
        else:
            nc = np.array(sp_cfg["nose_centre"])
            cand = [(len(c), i) for i, c in enumerate(comps) if i not in used
                    and abs(m.vertices[c, 1].mean()) < cfg["midline_mm"] * 1e-3
                    and _min_dist(m, c, nc) < cfg["nose_search_mm"] * 1e-3]
            if cand:
                chosen = [max(cand)[1]]
        for p in ov.get("include", []):                 # manual: add the component nearest to each point (mm)
            p = np.asarray(p) * 1e-3
            i = int(np.argmin([_min_dist(m, c, p) for c in comps]))
            if i not in chosen:
                chosen.append(i)
        for p in ov.get("exclude", []):                 # manual: drop the component nearest to each point (mm)
            p = np.asarray(p) * 1e-3
            i = int(np.argmin([_min_dist(m, c, p) for c in comps]))
            chosen = [c for c in chosen if c != i]
        if not chosen:
            notes[name] = "no component found"
            continue
        used.update(chosen)
        seeds[name] = np.unique(np.concatenate([comps[i] for i in chosen]))
        notes[name] = f"{len(chosen)} component(s): " + ", ".join(
            f"{len(comps[i])} v at {np.round(m.vertices[comps[i]].mean(0) * 1e3, 1).tolist()} mm" for i in chosen)
        if name == "noseleaf" and "nostrils" in lm:     # keep only the leaf above the nostrils, not the snout
            s = seeds[name]
            nm = np.mean(lm["nostrils"], 0)
            tip = s[np.argmax(np.linalg.norm(m.vertices[s] - nm, axis=1))]
            ax = (m.vertices[tip] - nm) / np.linalg.norm(m.vertices[tip] - nm)
            hgt = (m.vertices - nm) @ ax
            keep = s[hgt[s] > cfg["noseleaf_cut_mm"] * 1e-3]
            lat = ov.get("max_lateral_mm")              # manual: drop lateral flaps (|y - y_nostrils| > lat)
            if lat is not None:
                keep = keep[np.abs(m.vertices[keep, 1] - nm[1]) < lat * 1e-3]
            g = edge_graph(m, np.isin(np.arange(len(m.vertices)), keep))
            _, lab = connected_components(g, directed=False)
            keep = keep[lab[keep] == lab[tip]]
            cut[name] = {"axis": ax, "origin": nm, "height": hgt, "dropped": np.setdiff1d(s, keep)}
            seeds[name] = keep
            notes[name] += (f"; cut {cfg['noseleaf_cut_mm']} mm above the nostril midpoint along "
                            f"{np.round(ax, 2).tolist()}" + (f", |y| < {lat} mm" if lat is not None else "")
                            + f": {len(keep)} v kept")
    return seeds, comps, notes, cut


def harmonic(m, fixed, values):
    """Harmonic interpolation (1/length edge weights) of `values` at `fixed` vertices to the others."""
    g = edge_graph(m)
    g.data = 1.0 / np.maximum(g.data, 1e-9)
    L = sp.diags(np.asarray(g.sum(1)).ravel()) - g
    L = L.tocsr()
    free = ~fixed
    x = np.zeros(len(m.vertices))
    x[fixed] = values[fixed]
    rhs = -L[free][:, fixed] @ x[fixed]
    Lff = L[free][:, free]
    # vertices not connected to any constraint keep 0 (regularised solve)
    x[free] = spsolve((Lff + sp.identity(Lff.shape[0]) * 1e-12).tocsc(), rhs)
    return np.clip(x, 0, 1)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def appendage_weights(m, dcore, seeds, comps, cfg, cut=None):
    """Smooth weight h in [0, 1] per appendage (1 on the appendage, 0 on the head)."""
    band = {n: np.array(cfg["pinna_band_mm" if n.startswith("pinna") else "noseleaf_band_mm"]) * 1e-3 for n in seeds}
    other = np.zeros(len(m.vertices), bool)             # thin structures not in any appendage stay with the head
    for c in comps:
        other[c] = True
    for s in seeds.values():
        other[s] = False
    for name, c in (cut or {}).items():                 # the snout part below a noseleaf cut stays with the head
        other[c["dropped"]] = True
    core = np.zeros(len(m.vertices), bool)
    out = {}
    one = {}
    for name, s in seeds.items():
        o = np.zeros(len(m.vertices), bool)
        o[s] = dcore[s] >= band[name][1]
        if name in (cut or {}) and len(cut[name]["dropped"]):   # ramp away from the dropped (fixed) part
            from scipy.spatial import cKDTree
            dd = cKDTree(m.vertices[cut[name]["dropped"]]).query(m.vertices[s])[0]
            o[s] &= dd >= cfg["noseleaf_ramp_mm"] * 1e-3
        one[name] = o
    for name in seeds:
        fixed = (dcore <= band[name][0]) | other | core
        val = np.zeros(len(m.vertices))
        for n2, o in one.items():
            fixed |= o
            if n2 == name:
                val[o] = 1.0
        out[name] = smoothstep(harmonic(m, fixed, val))
    tot = sum(out.values())
    if len(out) > 1 and tot.max() > 1:
        for name in out:
            out[name] = out[name] / np.maximum(tot, 1)
    return out


def chain_coordinate(m, h, seed, quantile):
    """Geodesic distance from the h = 0.5 iso-line, normalised to 1 at the tip; base ring vertex indices."""
    e = m.edges_unique
    up = h >= 0.5
    cross = e[up[e[:, 0]] != up[e[:, 1]]]
    base = np.unique(np.where(up[cross[:, 0]], cross[:, 1], cross[:, 0]))      # head-side vertices of the crossing
    base = base[h[base] > 1e-3]
    g = dijkstra(edge_graph(m), directed=False, indices=base, min_only=True)
    app = h >= 0.5
    gmax = np.percentile(g[app], quantile)
    t = np.where(app, np.clip(g / gmax, 0, 1), 0.0)
    return t, base, g


def hats(t, knots, blend):
    """Partition of unity over len(knots) + 1 bones along t."""
    s = [smoothstep((t - (k - blend)) / (2 * blend)) for k in knots]
    w = []
    for i in range(len(knots) + 1):
        lo = s[i - 1] if i > 0 else 1.0
        hi = s[i] if i < len(knots) else 0.0
        w.append(lo - hi)
    return np.stack(w, 1)


def spatial_smooth(V, X, sigma):
    """Gaussian average of X (N, k) over Euclidean neighbours within 3 sigma (preserves a partition of unity)."""
    from scipy.spatial import cKDTree
    tr = cKDTree(V)
    D = tr.sparse_distance_matrix(tr, 3 * sigma, output_type="coo_matrix")
    G = sp.coo_matrix((np.exp(-0.5 * (D.data / sigma) ** 2), (D.row, D.col)), shape=D.shape).tocsr()
    G = G + sp.identity(len(V))                         # the diagonal (distance 0) is not stored
    G = sp.diags(1.0 / np.asarray(G.sum(1)).ravel()) @ G
    return G @ X


def perp(v, d):
    v = v - (v @ d) * d
    return v / np.linalg.norm(v)


def joint_axes(d, name):
    """Rest-pose rotation axes (unit vectors, CT frame) of a joint whose child bone points along d."""
    side = -1.0 if name.endswith("right") else 1.0
    lat = np.array([0, side, 0.0])
    if abs(d[1]) > 0.95:                                # bone along Y: fall back to X for the flex reference
        f = perp(np.array([0, 0, 1.0]), d)
    else:
        f = perp(np.array([0, -1.0, 0]), d)
    n = perp(lat, d) if abs(d @ lat) < 0.95 else np.cross(f, d) * side
    a = np.cross(d, n)
    a /= np.linalg.norm(a)
    return {"flex": f, "abduct": a, "twist": side * d}


def build_rig(key):
    """Segmentation, skeleton and weights of one species; returns (rig dict for JSON, arrays for NPZ)."""
    sp_cfg = paths.species()[key]
    cfg = config(key)
    m = mesh.load(sp_cfg["stl"])
    dcore = core_distance(m, cfg["open_radius_mm"] * 1e-3, cfg["voxel_mm"] * 1e-3)
    seeds, comps, notes, cut = segment(m, dcore, sp_cfg, cfg)
    H = appendage_weights(m, dcore, seeds, comps, cfg, cut)
    bones, joints, W, T = [], {}, [], {}
    for name, h in H.items():
        knots = cfg["pinna_joints"] if name.startswith("pinna") else cfg["noseleaf_joints"]
        nb = len(knots) + 1
        t, base, g = chain_coordinate(m, h, seeds[name], cfg["tip_quantile"])
        T[name] = t
        app = h >= 0.5
        pts = [m.vertices[base].mean(0)]
        for k in knots:
            band = app & (np.abs(t - k) < 0.04)
            pts.append(m.vertices[band].mean(0))
        pts.append(m.vertices[app & (t >= 0.96)].mean(0))
        hw = hats(t, knots, cfg["blend"]) * h[:, None]
        jn = JOINT_NAMES[nb]
        for i in range(nb):
            d = pts[i + 1] - pts[i]
            d = d / np.linalg.norm(d)
            jname = f"{name}.{jn[i]}"
            joints[jname] = {"position": pts[i].tolist(), "parent": "head" if i == 0 else f"{name}.{jn[i - 1]}",
                             "child_end": pts[i + 1].tolist(), "length_mm": float(np.linalg.norm(pts[i + 1] - pts[i]) * 1e3),
                             "t": 0.0 if i == 0 else knots[i - 1],
                             "axes": {k: v.tolist() for k, v in joint_axes(d, name).items()}}
            bones.append(jname)
            W.append(hw[:, i])
        joints[f"{name}.end"] = {"position": pts[-1].tolist(), "parent": f"{name}.{jn[-1]}", "end": True}
    W = np.stack(W, 1) if W else np.zeros((len(m.vertices), 0))
    if cfg["smooth_mm"] > 0 and W.shape[1]:
        W = spatial_smooth(m.vertices, W, cfg["smooth_mm"] * 1e-3)
        W[W < 1e-4] = 0
    head = 1 - W.sum(1)
    lm = landmark_attachments(m, sp_cfg)
    lm_w = {k: _point_weights(m, W, v["face"], v["bary"]) for k, v in lm.items()}
    meta = {"key": key, "species": sp_cfg.get("taxon", sp_cfg["name"]), "stl": sp_cfg["stl"],
            "frame": "CT frame, X forward, Y left, Z up, metres", "n_vertices": int(len(m.vertices)),
            "n_faces": int(len(m.faces)), "config": cfg, "segmentation": notes,
            "appendages": {n: {"n_vertices_moving": int((h > 1e-3).sum()), "n_vertices_full": int((h > 0.999).sum()),
                               "area_mm2": float(_area(m, h >= 0.5) * 1e6), "chain_length_mm":
                               float(sum(joints[b]["length_mm"] for b in bones if b.startswith(n + ".")))}
                           for n, h in H.items()},
            "bones": bones, "joints": joints, "axes_order": "R = R_flex R_abduct R_twist (degrees)",
            "landmarks": {k: {**v, "max_bone_weight": float(lm_w[k].max()) if len(lm_w[k]) else 0.0,
                              "bone": bones[int(np.argmax(lm_w[k]))] if len(lm_w[k]) and lm_w[k].max() > 0 else "head"}
                          for k, v in lm.items()},
            "skinning": "dual quaternion"}
    arrays = {"weights": W.astype(np.float32), "head_weight": head.astype(np.float32), "dcore": dcore.astype(np.float32),
              "bones": np.array(bones)}
    for n, h in H.items():
        arrays[f"h_{n}"] = h.astype(np.float32)
        arrays[f"t_{n}"] = T[n].astype(np.float32)
    return meta, arrays, m


def _area(m, vmask):
    f = vmask[m.faces].all(1)
    return m.area_faces[f].sum()


def landmark_attachments(m, sp_cfg):
    """Each landmark (ear_left, ear_right, nostril_0/1, mouth) as face index + barycentric coordinates of its
    closest point on the rest mesh, plus its offset along the face normal."""
    out = {}
    lm = sp_cfg["landmarks"]
    pts = {"ear_left": lm["ear_left"], "ear_right": lm["ear_right"]}
    if "nostrils" in lm:
        pts.update({f"nostril_{i}": p for i, p in enumerate(lm["nostrils"])})
    if "mouth" in lm:
        pts["mouth"] = lm["mouth"]
    if "nose_centre" in sp_cfg:
        pts["nose_centre"] = sp_cfg["nose_centre"]
    P = np.array(list(pts.values()))
    cp, _, fi = trimesh.proximity.closest_point(m, P)
    for (k, p), c, f in zip(pts.items(), cp, fi):
        b = trimesh.triangles.points_to_barycentric(m.triangles[[f]], [c])[0]
        off = float((np.asarray(p) - c) @ m.face_normals[f])
        out[k] = {"rest": list(map(float, p)), "face": int(f), "bary": b.tolist(), "normal_offset_m": off}
    return out


def _point_weights(m, W, f, b):
    if W.shape[1] == 0:
        return np.zeros(0)
    return np.asarray(b) @ W[m.faces[f]]


# ------------------------------------------------------------------------------------------- kinematics / skinning

def axis_angle(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def quat_from_matrix(R):
    """(w, x, y, z), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def qmul(a, b):
    """Hamilton product, broadcasting over leading axes."""
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def dq_from_rt(R, t):
    qr = quat_from_matrix(R)
    qd = 0.5 * qmul(np.r_[0.0, t], qr)
    return qr, qd


def dqs(V, W, transforms):
    """Dual-quaternion skinning. V (N, 3), W (N, B) bone weights (the head root, identity, gets 1 - sum),
    transforms: list of B (R, t). Returns deformed vertices and per-vertex rotation matrices (N, 3, 3)."""
    QR = np.array([[1.0, 0, 0, 0]] + [dq_from_rt(R, t)[0] for R, t in transforms])
    QD = np.array([[0.0, 0, 0, 0]] + [dq_from_rt(R, t)[1] for R, t in transforms])
    sgn = np.where(QR[:, 0] < 0, -1.0, 1.0)            # same hemisphere as the identity (rotations < 180 deg)
    QR, QD = QR * sgn[:, None], QD * sgn[:, None]
    Wf = np.concatenate([np.clip(1 - W.sum(1, keepdims=True), 0, 1), W], 1)
    br = Wf @ QR
    bd = Wf @ QD
    n = np.linalg.norm(br, axis=1, keepdims=True)
    br, bd = br / n, bd / n
    w, x, y, z = br.T
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    conj = br * np.array([1, -1, -1, -1.0])
    t = 2 * qmul(bd, conj)[:, 1:]
    return np.einsum("nij,nj->ni", R, V) + t, R


@dataclass
class Rig:
    key: str
    meta: dict
    rest: trimesh.Trimesh
    W: np.ndarray
    arrays: dict = field(default_factory=dict)

    @property
    def bones(self):
        return list(self.meta["bones"])

    @property
    def joints(self):
        return self.meta["joints"]

    def expand_pose(self, pose):
        """Pose dict {joint: {axis: deg}}; "pinna_both.<joint>" applies to both pinnae (mirror-symmetric axes)."""
        out = {}
        for k, v in (pose or {}).items():
            if k.startswith("pinna_both."):
                for s in ("left", "right"):
                    out[k.replace("both", s)] = v
            else:
                out[k] = v
        for k, v in out.items():
            if k not in self.bones:
                raise KeyError(f"unknown joint {k!r}; joints: {self.bones}")
            for a in v:
                if a not in AXES:
                    raise KeyError(f"unknown axis {a!r} at {k}; axes: {AXES}")
        return out

    def global_transforms(self, pose):
        """Rigid transform (R, t), x' = R x + t, of every bone for a pose (forward kinematics)."""
        pose = self.expand_pose(pose)
        G = {"head": (np.eye(3), np.zeros(3))}
        for b in self.bones:                                # bones are stored parent before child
            j = self.joints[b]
            p = np.array(j["position"])
            ang = pose.get(b, {})
            R = np.eye(3)
            for a in AXES:                                  # R = R_yaw R_flex R_abduct R_twist
                if ang.get(a, 0):
                    ax = j["axes"].get(a) if a != "yaw" else j["axes"].get("yaw", yaw_axis(b))
                    R = R @ axis_angle(ax, ang[a])
            Rp, tp = G[j["parent"]]
            G[b] = (Rp @ R, Rp @ (p - R @ p) + tp)
        return [G[b] for b in self.bones], G

    def deform(self, pose, return_rotations=False):
        """Deformed mesh (trimesh, same faces) for a pose."""
        Ts, _ = self.global_transforms(pose)
        V, Rv = dqs(self.rest.vertices, self.W, Ts)
        m = trimesh.Trimesh(V, self.rest.faces.copy(), process=False)
        return (m, Rv) if return_rotations else m

    def deform_landmarks(self, pose, m=None):
        """Landmarks on the deformed surface (same face, same barycentric coordinates, same normal offset)."""
        m = self.deform(pose) if m is None else m
        out = {}
        for k, a in self.meta["landmarks"].items():
            tri = m.triangles[a["face"]]
            p = np.asarray(a["bary"]) @ tri
            out[k] = p + a["normal_offset_m"] * m.face_normals[a["face"]]
        return out

    def joint_positions(self, pose):
        _, G = self.global_transforms(pose)
        out = {}
        for k, j in self.joints.items():
            R, t = G[j["parent"]] if j.get("end") else G[k]
            out[k] = R @ np.array(j["position"]) + t
        return out

    def check(self, m, pose=None, si=True):
        """Validity of a deformed mesh: watertight / consistently wound, self-intersections (pymeshlab, as in
        scripts/remesh_ct.state), faces inverted against their skinned rest normal, edge stretch."""
        out = {"watertight": bool(m.is_watertight), "winding_consistent": bool(m.is_winding_consistent)}
        if pose is not None:
            _, Rv = self.deform(pose, return_rotations=True)
            nr = np.einsum("fij,fj->fi", Rv[self.rest.faces].mean(1), self.rest.face_normals)
            dots = np.einsum("fi,fi->f", nr, m.face_normals)
            out["inverted_faces"] = int((dots < 0).sum())
            out["max_normal_deviation_deg"] = float(np.degrees(np.arccos(np.clip(dots.min(), -1, 1))))
        el0 = np.linalg.norm(np.diff(self.rest.vertices[self.rest.edges_unique], axis=1)[:, 0], axis=1)
        el1 = np.linalg.norm(np.diff(m.vertices[self.rest.edges_unique], axis=1)[:, 0], axis=1)
        big = el0 >= STRETCH_MIN_EDGE * el0.mean()        # very short edges (repair spots) give meaningless ratios
        r = el1[big] / el0[big]
        out["edge_stretch_min_max"] = [float(r.min()), float(r.max())]
        a0, a1 = self.rest.area_faces, m.area_faces
        out["area_ratio_min"] = float((a1 / a0).min())
        out["edge_max_mm"] = float(el1.max() * 1e3)
        out["volume_change"] = float(m.volume / self.rest.volume - 1)
        if si:
            out["self_intersecting_faces"] = self_intersections(m)
        out["valid"] = bool(out["watertight"] and out.get("self_intersecting_faces", 0) == 0
                            and out.get("inverted_faces", 0) == 0 and STRETCH_LIMITS[0] <= r.min() and r.max() <= STRETCH_LIMITS[1])
        return out

    def valid_range(self, joint, axis, step=5.0, max_deg=90.0):
        """Largest |angle| (both signs, in `step` increments) for which the mesh stays valid."""
        res = {}
        for sgn in (1, -1):
            last = 0.0
            a = step
            while a <= max_deg + 1e-9:
                if not self.check(self.deform({joint: {axis: sgn * a}}), pose={joint: {axis: sgn * a}})["valid"]:
                    break
                last = a
                a += step
            res["+" if sgn > 0 else "-"] = last
        return [-res["-"], res["+"]]


def self_intersections(m):
    """Number of self-intersecting faces (pymeshlab, as in scripts/remesh_ct.state)."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(m.vertices, float), np.asarray(m.faces, np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    return int(ms.current_mesh().selected_face_number())


def save_rig(meta, arrays):
    RIG_DIR.mkdir(parents=True, exist_ok=True)
    (RIG_DIR / f"{meta['key']}_rig.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(RIG_DIR / f"{meta['key']}_weights.npz", **arrays)


def load_rig(key):
    meta = json.loads((RIG_DIR / f"{key}_rig.json").read_text())
    z = np.load(RIG_DIR / f"{key}_weights.npz")
    arrays = {k: z[k] for k in z.files}
    m = mesh.load(meta["stl"])
    assert len(m.vertices) == meta["n_vertices"] and len(m.faces) == meta["n_faces"], "mesh changed since the rig was built"
    return Rig(key, meta, m, arrays["weights"].astype(float), arrays)


def load_rig_folder(folder):
    """Rig of a species folder of the released dataset (<folder>/rig/rig.json, rig/weights.npz, mesh.stl), for users
    who have the dataset but not this repository's results/rig."""
    from pathlib import Path
    folder = Path(folder)
    meta = json.loads((folder / "rig" / "rig.json").read_text())
    z = np.load(folder / "rig" / "weights.npz")
    arrays = {k: z[k] for k in z.files}
    m = mesh.load(str((folder / "mesh.stl").resolve()))
    assert len(m.vertices) == meta["n_vertices"] and len(m.faces) == meta["n_faces"], "mesh.stl does not match the rig"
    return Rig(meta.get("key", folder.name), meta, m, arrays["weights"].astype(float), arrays)


# ------------------------------------------------------------------------------------------------- glTF export

_GLTF_FROM_CT = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])     # glTF (Y up) = (x, z, -y) of the CT frame


def joint_matrices(rig):
    """Rest-pose 4x4 matrices (CT frame, metres) of the head root and every joint: origin at the joint, local Y along
    the bone, local X = flex axis, local Z = X x Y (= abduct axis on the left side, -abduct on the right)."""
    out = {"head": np.eye(4)}
    for name, j in rig.joints.items():
        M = np.eye(4)
        M[:3, 3] = j["position"]
        if j.get("end"):
            par = out[j["parent"]]
            M[:3, :3] = par[:3, :3]
        else:
            y = np.array(j["child_end"]) - np.array(j["position"])
            y /= np.linalg.norm(y)
            x = np.array(j["axes"]["flex"])
            x = x - (x @ y) * y
            x /= np.linalg.norm(x)
            M[:3, :3] = np.stack([x, y, np.cross(x, y)], 1)
        out[name] = M
    return out


def export_gltf(rig, path, scale=1000.0):
    """Binary glTF (GLB) of the rest mesh with a skin: joints = head root + the bones + an end joint per chain,
    4 weights per vertex. Units: millimetres by default (scale 1000; Blender's default clip range does not suit a
    30 mm head in metres). Axes: glTF Y up; Blender's importer turns it back into X forward, Y left, Z up."""
    import pygltflib as gl
    Ms = joint_matrices(rig)
    names = ["head"] + list(rig.joints)
    C = np.eye(4)
    C[:3, :3] = _GLTF_FROM_CT * 1.0
    Mg = {}                                               # joint frames in glTF coordinates (columns: local axes)
    for n in names:
        Mg[n] = np.eye(4)
        Mg[n][:3, :3] = np.eye(3) if n == "head" else C[:3, :3] @ Ms[n][:3, :3]
        Mg[n][:3, 3] = scale * (C[:3, :3] @ Ms[n][:3, 3])
    V = (rig.rest.vertices @ _GLTF_FROM_CT.T * scale).astype(np.float32)
    N = (rig.rest.vertex_normals @ _GLTF_FROM_CT.T).astype(np.float32)
    F = rig.rest.faces.astype(np.uint32)
    Wfull = np.concatenate([np.clip(1 - rig.W.sum(1, keepdims=True), 0, 1), rig.W], 1)   # columns: head, bones
    col_joint = [0] + [names.index(b) for b in rig.bones]
    order = np.argsort(-Wfull, axis=1)[:, :4]
    J = np.take(np.array(col_joint), order).astype(np.uint16)
    Wt = np.take_along_axis(Wfull, order, 1)
    Wt = (Wt / Wt.sum(1, keepdims=True)).astype(np.float32)
    J[Wt == 0] = 0
    IBM = np.stack([np.linalg.inv(Mg[n]).T for n in names]).astype(np.float32)      # column-major
    blobs, views, accs = [], [], []

    def add(arr, target, ctype, typ, mm=False):
        off = sum(len(b) for b in blobs)
        b = arr.tobytes()
        b += b"\0" * ((-len(b)) % 4)
        blobs.append(b)
        views.append(gl.BufferView(buffer=0, byteOffset=off, byteLength=arr.nbytes, target=target))
        a = gl.Accessor(bufferView=len(views) - 1, componentType=ctype, count=len(arr), type=typ)
        if mm:
            a.min, a.max = arr.min(0).tolist(), arr.max(0).tolist()
        accs.append(a)
        return len(accs) - 1

    a_pos = add(V, gl.ARRAY_BUFFER, gl.FLOAT, gl.VEC3, mm=True)
    a_nrm = add(N, gl.ARRAY_BUFFER, gl.FLOAT, gl.VEC3)
    a_jnt = add(J, gl.ARRAY_BUFFER, gl.UNSIGNED_SHORT, gl.VEC4)
    a_wgt = add(Wt, gl.ARRAY_BUFFER, gl.FLOAT, gl.VEC4)
    a_idx = add(F.reshape(-1), gl.ELEMENT_ARRAY_BUFFER, gl.UNSIGNED_INT, gl.SCALAR)
    a_ibm = add(IBM, None, gl.FLOAT, gl.MAT4)
    nodes = [gl.Node(name=f"{rig.key}_head", mesh=0, skin=0)]
    jnode = {}
    for n in names:
        jnode[n] = len(nodes)
        par = rig.joints[n]["parent"] if n != "head" else None
        L = Mg[n] if par is None else np.linalg.inv(Mg[par]) @ Mg[n]
        q = quat_from_matrix(L[:3, :3])
        nodes.append(gl.Node(name=n, translation=L[:3, 3].tolist(), rotation=[q[1], q[2], q[3], q[0]]))
    for n in names:
        kids = [jnode[c] for c, j in rig.joints.items() if j["parent"] == n]
        if kids:
            nodes[jnode[n]].children = kids
    g = gl.GLTF2(asset=gl.Asset(generator="batertf.rig", version="2.0"), scene=0,
                 scenes=[gl.Scene(nodes=[0, jnode["head"]])], nodes=nodes,
                 meshes=[gl.Mesh(name=rig.key, primitives=[gl.Primitive(
                     attributes=gl.Attributes(POSITION=a_pos, NORMAL=a_nrm, JOINTS_0=a_jnt, WEIGHTS_0=a_wgt),
                     indices=a_idx)])],
                 skins=[gl.Skin(name=f"{rig.key}_rig", joints=[jnode[n] for n in names], skeleton=jnode["head"],
                                inverseBindMatrices=a_ibm)],
                 accessors=accs, bufferViews=views)
    data = b"".join(blobs)
    g.buffers = [gl.Buffer(byteLength=len(data))]
    g.set_binary_blob(data)
    g.extras = {"units": "mm" if scale == 1000 else f"scale {scale}", "frame": "CT frame via glTF Y-up (x, z, -y)",
                "axes": "joint local X = flex, local Y = bone, local Z = X x Y"}
    g.save_binary(str(path))
    return path


def _gltf_globals(g):
    from scipy.spatial.transform import Rotation as Rot
    G = {}

    def rec(i, Mp):
        n = g.nodes[i]
        if n.matrix:
            L = np.array(n.matrix, float).reshape(4, 4).T
        else:
            L = np.eye(4)
            L[:3, :3] = Rot.from_quat(n.rotation or [0, 0, 0, 1]).as_matrix() * np.asarray(n.scale or [1, 1, 1])
            L[:3, 3] = n.translation or [0, 0, 0]
        G[i] = Mp @ L
        for c in n.children or []:
            rec(c, G[i])

    kids = {c for n in g.nodes for c in (n.children or [])}
    for i in range(len(g.nodes)):
        if i not in kids:
            rec(i, np.eye(4))
    return G


def pose_from_gltf(rig, path):
    """Pose dict (degrees, flex / abduct / twist) from a posed glTF/GLB of this rig, e.g. exported from Blender after
    posing the armature. Uses the skinning matrices (posed global x inverse bind), so it does not depend on how the
    importer oriented the bones; only bone names must be kept."""
    import pygltflib as gl
    from scipy.spatial.transform import Rotation as Rot
    g = gl.GLTF2().load(str(path))
    sk = g.skins[0]
    a = g.accessors[sk.inverseBindMatrices]
    v = g.bufferViews[a.bufferView]
    blob = g.binary_blob() if g.binary_blob() is not None else None
    if blob is None:
        raise ValueError("only GLB with an embedded buffer is supported")
    IBM = np.frombuffer(blob[v.byteOffset + (a.byteOffset or 0):][:a.count * 64], np.float32).reshape(-1, 4, 4).transpose(0, 2, 1)
    G = _gltf_globals(g)
    Sk = {g.nodes[j].name: G[j] @ IBM[k].astype(float) for k, j in enumerate(sk.joints)}
    C = _GLTF_FROM_CT
    pose = {}
    for b in rig.bones:
        j = rig.joints[b]
        par = j["parent"] if j["parent"] in Sk else "head"
        Rrel = np.linalg.inv(Sk[par][:3, :3]) @ Sk[b][:3, :3] if par in Sk else Sk[b][:3, :3]
        Rrel = C.T @ Rrel @ C                                  # CT frame, world rest coordinates
        M = joint_matrices(rig)[b][:3, :3]                     # columns: flex, bone, X x Y
        Rloc = M.T @ Rrel @ M
        fx, az, ty = Rot.from_matrix(Rloc).as_euler("XZY")     # R = Rx(flex) Rz Ry (intrinsic) = R_flex R_abduct R_twist
        side = -1.0 if b.startswith("pinna_right") else 1.0
        ang = {"flex": np.degrees(fx), "abduct": side * np.degrees(az), "twist": side * np.degrees(ty)}
        ang = {k: round(float(x), 3) for k, x in ang.items() if abs(x) > 1e-3}
        if ang:
            pose[b] = ang
    return pose
