"""Reference ERTFs (De Mey et al. 2008 for P. discolor; Vanderelst et al. 2010 for M. microtis) and comparison metrics.

Reference files hold linear magnitudes on the same [el, az, f] grid (frontal hemisphere, 2.5 deg, 1 m), in arbitrary
units: only patterns and relative spectra are compared, never absolute levels.
"""
import numpy as np
import scipy.io as sio

from . import grid, paths


def load(key):
    b = sio.loadmat(paths.REF_DIR / paths.species()[key]["reference"], squeeze_me=True,
                    struct_as_record=False)["bat_hrtf"]
    assert np.allclose(b.az_vec, grid.AZ) and np.allclose(b.el_vec, grid.EL)
    return {"freqs": np.asarray(b.freq_vec, float), "left": b.hrtf_left, "right": b.hrtf_right,
            "emitter": b.hrtf_em}


def solid_angle_weights():
    """Weights of the [el, az] grid cells on the sphere (cos(el)), zero-weight duplicate pole points handled by cos."""
    w = np.cos(np.deg2rad(grid.EL))[:, None] * np.ones(len(grid.AZ))[None, :]
    return w / w.sum()


def pattern_db(mag):
    """Per-frequency directivity in dB re its maximum over the hemisphere. mag: [el, az, f] magnitudes."""
    return 20 * np.log10(mag / mag.max(axis=(0, 1), keepdims=True))


def metrics(sim_mag, ref_mag, floor_db=-30.0):
    """Per-frequency comparison of directivity patterns (each normalized to its own maximum, clipped at floor_db):
    rms dB difference (solid-angle weighted), correlation of the dB patterns, angle between the peak directions."""
    a = np.maximum(pattern_db(sim_mag), floor_db)
    b = np.maximum(pattern_db(ref_mag), floor_db)
    w = solid_angle_weights()[..., None]
    rms = np.sqrt((w * (a - b) ** 2).sum((0, 1)))
    am = a - (w * a).sum((0, 1))
    bm = b - (w * b).sum((0, 1))
    corr = (w * am * bm).sum((0, 1)) / np.sqrt((w * am ** 2).sum((0, 1)) * (w * bm ** 2).sum((0, 1)))
    u = grid.unit_vectors().reshape(-1, 3)
    pa = u[np.argmax(sim_mag.reshape(-1, sim_mag.shape[-1]), 0)]
    pb = u[np.argmax(ref_mag.reshape(-1, ref_mag.shape[-1]), 0)]
    peak = np.degrees(np.arccos(np.clip((pa * pb).sum(1), -1, 1)))
    return {"rms_db": rms, "corr": corr, "peak_deg": peak}


def metrics_points(sim_mag, ref_mag, w, u, floor_db=-30.0):
    """metrics() on an arbitrary set of directions: sim_mag, ref_mag (P, F) magnitudes, w (P,) solid-angle weights,
    u (P, 3) unit vectors. Same definitions (per-frequency max normalization, floor, weighted rms dB, weighted
    correlation of the dB patterns, angle between the peak directions)."""
    a = np.maximum(20 * np.log10(sim_mag / sim_mag.max(0)), floor_db)
    b = np.maximum(20 * np.log10(ref_mag / ref_mag.max(0)), floor_db)
    w = (np.asarray(w, float) / np.sum(w))[:, None]
    rms = np.sqrt((w * (a - b) ** 2).sum(0))
    am, bm = a - (w * a).sum(0), b - (w * b).sum(0)
    corr = (w * am * bm).sum(0) / np.sqrt((w * am ** 2).sum(0) * (w * bm ** 2).sum(0))
    pa, pb = u[np.argmax(sim_mag, 0)], u[np.argmax(ref_mag, 0)]
    peak = np.degrees(np.arccos(np.clip((pa * pb).sum(1), -1, 1)))
    return {"rms_db": rms, "corr": corr, "peak_deg": peak}
