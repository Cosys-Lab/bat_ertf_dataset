"""Midsagittal plane of a head mesh from its mirror symmetry.

The plane has normal n = R(yaw, roll) [0, 1, 0] through the point [0, y0, 0] (rotate.matrix convention, pitch 0).
Cost: trimmed mean distance between the mirrored vertices and the mesh (the TRIM fraction of best-matching vertices),
so asymmetric parts (deformed pinnae, a turned noseleaf) do not dominate.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from . import rotate

TRIM = 0.7


def cost(params, v, tree):
    yaw, roll, y0 = params
    n = rotate.matrix(yaw, 0, roll) @ np.array([0.0, 1.0, 0.0])
    d = (v - [0, y0 * 1e-3, 0]) @ n
    mirrored = v - 2 * d[:, None] * n
    dist, _ = tree.query(mirrored)
    return np.sort(dist)[: int(TRIM * len(v))].mean() * 1e3


def fit(m, n_sub=8000, seed=0):
    """(yaw, roll, y0_mm) of the best symmetry plane; tries several starts."""
    v = m.vertices[np.random.default_rng(seed).choice(len(m.vertices), min(n_sub, len(m.vertices)), replace=False)]
    tree = cKDTree(m.vertices)
    best = None
    for yaw0 in (-20, 0, 20):
        for roll0 in (-10, 0, 10):
            r = minimize(cost, [yaw0, roll0, 0.0], args=(v, tree), method="Nelder-Mead",
                         options={"xatol": 0.05, "fatol": 1e-5, "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    return best.x, best.fun
