"""Rotations between the mesh frame and the frame in which ERTFs are reported.

R = Rz(yaw) Ry(pitch) Rx(roll) maps a direction in the reporting frame to the mesh frame: u_mesh = R u_ref.
To bring a mesh into the reporting frame, rotate its vertices by R^T. Positive yaw turns toward +Y (left),
positive pitch turns +X toward -Z (nose down, right-hand rule about +Y), positive roll turns +Y toward +Z.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import grid


def matrix(yaw, pitch, roll):
    a, b, c = np.deg2rad([yaw, pitch, roll])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return rz @ ry @ rx


def to_az_el(u):
    return np.degrees(np.arctan2(u[..., 1], u[..., 0])), np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))


class SphereSampler:
    """Bilinear interpolation of data on the full-sphere grid [el, az, ...] at arbitrary directions."""

    def __init__(self, data):
        az, el = grid.GRIDS["sphere"]
        az_ext = np.concatenate([az, [az[0] + 360]])
        d = np.concatenate([data, data[:, :1]], axis=1)
        self.f = RegularGridInterpolator((el, az_ext), d)
        self.az0 = az[0]

    def __call__(self, u):
        az, el = to_az_el(u)
        az = (az - self.az0) % 360 + self.az0
        return self.f(np.stack([el, az], -1))


def hemisphere_from_sphere(data, R):
    """Data on the full-sphere grid (mesh frame) resampled on the hemisphere grid of the reporting frame."""
    u = grid.unit_vectors() @ R.T                           # u_mesh = R u_ref
    return SphereSampler(data)(u)
