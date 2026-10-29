"""Is the remaining difference between the measured and the simulated HRTF of Vesterholm's Mdau2 a registration
problem? Fits a free rotation (3 parameters) between the measurement directions and the mesh, on top of their
RotoTranslationMatrix, for our simulation and for their Mesh2HRTF simulation, by maximizing the mean pattern
correlation with the measurement. The rotation is fitted on the even frequencies and evaluated on the odd ones (and
on all), so that the gain is not an artefact of the fit. Uses the transfer functions of validate_vesterholm.py on
their Mesh2HRTF grid (<work>/vesterholm/solve_full.npz), no new BEM run.
Usage: python scripts/vesterholm_rotation_fit.py   -> results/validation/vesterholm_rotation_fit.json"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import paths, reference
from validate_vesterholm import FLOOR, SUB, load_measured, load_sofa, measured_mag, unit


def sampler(sofa, A):
    """Bilinear interpolation of magnitudes A (19900,) on the Mesh2HRTF grid at unit vectors u."""
    A = np.abs(A).reshape(100, 199)
    az = np.concatenate([sofa["az"], [sofa["az"][0] + 360]])
    f = RegularGridInterpolator((sofa["el"], az), np.concatenate([A, A[:, :1]], 1))

    def at(u):
        a = (np.degrees(np.arctan2(u[:, 1], u[:, 0])) - az[0]) % 360 + az[0]
        return f(np.stack([np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1))), a], -1))
    return at


def main():
    d = paths.work_dir().parent / "external" / "vesterholm2025" / "files"
    meas, sofa = load_measured(d), load_sofa(d)
    R3 = sio.loadmat(d / SUB / "Measured HRTF" / "Mdau2_RotationMatrix.mat")["RotoTranslationMatrix"][:3, :3]
    z = np.load(paths.work_dir() / "vesterholm" / "solve_full.npz")
    freqs = z["freqs"]
    P = len(meas["az"])
    H_grid = z["H"][2 * P:]                                 # our solution on their grid (mesh frame)
    u_meas = unit(meas["az"], meas["el"])
    Mm, w = measured_mag(meas, freqs), meas["w"]
    ks = np.array([int(np.argmin(abs(sofa["freqs"] - f))) for f in freqs])
    has = np.isclose(sofa["freqs"][ks], freqs)
    sims = {"ours": [sampler(sofa, H_grid[:, j]) for j in range(len(freqs))],
            "mesh2hrtf": [sampler(sofa, sofa["H"][:, ks[j]]) if has[j] else None for j in range(len(freqs))]}

    def corr(name, rv, idx):
        u = u_meas @ (Rotation.from_rotvec(rv).as_matrix() @ R3).T
        S = np.stack([sims[name][j](u) for j in idx], 1)
        return reference.metrics_points(S, Mm[:, idx], w, u_meas, FLOOR)

    out = {"what": __doc__.split("\n")[0], "freqs_khz": (freqs / 1e3).tolist()}
    for name in sims:
        avail = np.where(has)[0] if name == "mesh2hrtf" else np.arange(len(freqs))
        fit, test = avail[::2], avail[1::2]
        cost = lambda rv: -np.mean(corr(name, rv, fit)["corr"])
        best = min((minimize(cost, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 600})
                    for x0 in [np.zeros(3)] + [np.radians(15) * v for v in np.eye(3)] + [-np.radians(15) * v for v in np.eye(3)]),
                   key=lambda r: r.fun)
        rv = best.x
        r = {}
        for lab, idx in (("fit_freqs", fit), ("test_freqs", test), ("all", avail)):
            m0, m1 = corr(name, np.zeros(3), idx), corr(name, rv, idx)
            r[lab] = {"corr_median_their_rotation": float(np.median(m0["corr"])),
                      "corr_median_fitted": float(np.median(m1["corr"])),
                      "rms_db_median_their_rotation": float(np.median(m0["rms_db"])),
                      "rms_db_median_fitted": float(np.median(m1["rms_db"])),
                      "peak_deg_median_their_rotation": float(np.median(m0["peak_deg"])),
                      "peak_deg_median_fitted": float(np.median(m1["peak_deg"]))}
        r["extra_rotation_deg"] = float(np.degrees(np.linalg.norm(rv)))
        r["extra_rotation_euler_zyx_deg"] = Rotation.from_rotvec(rv).as_euler("zyx", degrees=True).tolist()
        r["per_frequency_corr_fitted"] = corr(name, rv, avail)["corr"].tolist()
        out[name] = r
        print(f"{name}: extra rotation {r['extra_rotation_deg']:.1f} deg; median corr (test freqs) "
              f"{r['test_freqs']['corr_median_their_rotation']:.3f} -> {r['test_freqs']['corr_median_fitted']:.3f}; "
              f"all {r['all']['corr_median_their_rotation']:.3f} -> {r['all']['corr_median_fitted']:.3f}; peak "
              f"{r['all']['peak_deg_median_their_rotation']:.1f} -> {r['all']['peak_deg_median_fitted']:.1f} deg", flush=True)
    fn = paths.RESULTS / "validation" / "vesterholm_rotation_fit.json"
    fn.write_text(json.dumps(out, indent=1))
    print(fn)


if __name__ == "__main__":
    main()
