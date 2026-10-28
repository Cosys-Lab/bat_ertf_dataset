"""Evaluation grids, at 1 m from the origin.

"hemisphere": frontal hemisphere, azimuth and elevation -90..90 deg in 2.5 deg steps (the reference grid of
              De Mey et al. 2008 and Vanderelst et al. 2010); the grid of the published ERTFs.
"sphere":     full sphere, azimuth -180..177.5, elevation -90..90 deg, 2.5 deg steps (pilot runs, to fit the
              head orientation by rotating the field).

Direction convention (matches the reference files): azimuth theta positive to the bat's left (+Y), elevation phi
positive up (+Z), unit vector u = [cos(phi) cos(theta), cos(phi) sin(theta), sin(phi)]. Arrays are [el, az, ...].
"""
import numpy as np

AZ = np.arange(-90, 90.01, 2.5)
EL = np.arange(-90, 90.01, 2.5)
GRIDS = {"hemisphere": (AZ, EL), "sphere": (np.arange(-180, 179.99, 2.5), EL)}
RADIUS = 1.0


def unit_vectors(az=AZ, el=EL):
    t, p = np.meshgrid(np.deg2rad(az), np.deg2rad(el))          # [el, az]
    return np.stack([np.cos(p) * np.cos(t), np.cos(p) * np.sin(t), np.sin(p)], -1)


def unique_points(name="hemisphere"):
    """Unique evaluation directions (the poles el = +-90 are one point each) and the map back to the [el, az] grid."""
    u = unit_vectors(*GRIDS[name])
    flat = u.reshape(-1, 3)
    _, first, inv = np.unique(np.round(flat, 9), axis=0, return_index=True, return_inverse=True)
    return flat[first] * RADIUS, inv.reshape(u.shape[:2])
