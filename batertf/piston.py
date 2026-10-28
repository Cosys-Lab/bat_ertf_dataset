"""Analytical emitter for oral (mouth-emitting) species: a circular piston with the Kirchhoff obliquity factor,
radius calibrated to published half-amplitude angles (docs/mouth_emitter_calibration.md).

    H(u, f) = 2 * [2 J1(x) / x] * (1 + cos t) / 2 * exp(+j k u.p),   x = k a sin t,  cos t = u.n
t is the angle from the piston axis n, p the piston centre (mouth landmark), a the radius. Referenced like the BEM
sources (pressure / free-field monopole of the same volume velocity at the origin, exp(+j w t) convention): on axis
|H| = 2, the baffled-piston value. Near the axis this is the baffled piston of Strother & Mogus (1970) used by
Surlykke et al. (2009); the obliquity factor keeps it smooth and finite over the whole sphere.
"""
import numpy as np
from scipy.special import j1

from . import sources

C = 343.0
AXIS_PATCH_RADIUS = 2.0e-3      # region around the mouth landmark whose mean outward normal is the piston axis


def directivity(x, cos_t):
    d = np.where(x < 1e-9, 1.0, 2 * j1(x) / np.maximum(x, 1e-9))
    return d * (1 + cos_t) / 2


def transfer(u, freqs, centre, axis, radius):
    """u: unit vectors (..., 3); returns complex (..., n_freqs)."""
    cos_t = np.clip(u @ axis, -1, 1)
    sin_t = np.sqrt(1 - cos_t ** 2)
    k = 2 * np.pi * np.asarray(freqs) / C
    x = k * radius * sin_t[..., None]
    return 2 * directivity(x, cos_t[..., None]) * np.exp(1j * k * (u @ centre)[..., None])


def mouth_axis(m, centre):
    """Area-weighted mean outward normal of the surface within AXIS_PATCH_RADIUS of the mouth landmark."""
    f = sources.disc_patch(m, np.asarray(centre), AXIS_PATCH_RADIUS)
    n = (m.face_normals[f] * m.area_faces[f][:, None]).sum(0)
    return n / np.linalg.norm(n)
