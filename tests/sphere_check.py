"""Validation of the NumCalc pipeline, magnitude AND phase: rigid sphere (radius a = 5 mm) with a vibrating spherical
cap at +X, against the exact series solution (exp(-i w t) convention, Williams, Fourier Acoustics, ch. 6):
    p(r, theta) = i rho c  sum_n  v_n  h_n(kr) / h_n'(ka)  P_n(cos theta),
    v_n = V/2 [P_{n-1}(cos alpha) - P_{n+1}(cos alpha)]     (cap of half-angle alpha, normal velocity V)
Both are referenced and conjugated exactly like the bat ERTFs (numcalc.read_source).
Usage: python tests/sphere_check.py        -> results/validation/sphere_check.{png,txt}"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.special import eval_legendre, spherical_jn, spherical_yn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import grid, numcalc, paths, sources

A = 5e-3
FREQS = [20000.0, 40000.0, 80000.0, 120000.0]


def hankel(n, x, d=False):
    return spherical_jn(n, x, d) + 1j * spherical_yn(n, x, d)


def analytic(theta, f, alpha, r=grid.RADIUS, n_max=120):
    k = 2 * np.pi * f / numcalc.C
    p = np.zeros_like(theta, complex)
    for n in range(n_max):
        pm1 = 1.0 if n == 0 else eval_legendre(n - 1, np.cos(alpha))
        vn = numcalc.VELOCITY / 2 * (pm1 - eval_legendre(n + 1, np.cos(alpha)))
        p += vn * hankel(n, k * r) / hankel(n, k * A, True) * eval_legendre(n, np.cos(theta))
    p *= 1j * numcalc.RHO * numcalc.C
    q = numcalc.VELOCITY * 2 * np.pi * A ** 2 * (1 - np.cos(alpha))
    p_ff = -1j * 2 * np.pi * f * numcalc.RHO * q * np.exp(1j * k * r) / (4 * np.pi * r)
    return np.conj(p / p_ff)


def main():
    m = trimesh.creation.icosphere(subdivisions=5, radius=A)
    cap = sources.disc_patch(m, np.array([A, 0, 0]), 0.8e-3)
    alpha = np.arccos(1 - m.area_faces[cap].sum() / (2 * np.pi * A ** 2))   # cap with the patch's area
    proj = paths.work_dir() / "tests" / "sphere"
    shutil.rmtree(proj, ignore_errors=True)
    numcalc.write_project(proj, m, {"cap": [cap]}, FREQS)
    numcalc.run_steps(numcalc.steps(proj), paths.local()["numcalc"], len(FREQS))
    f, H = numcalc.read_source(proj, "cap")
    i0 = np.argmin(abs(grid.EL))
    az = grid.AZ
    out = paths.RESULTS / "validation"
    out.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, len(f), figsize=(4 * len(f), 6.5), sharex=True)
    lines = [f"sphere a = {A*1e3} mm, {len(m.faces)} faces, cap {len(cap)} faces, alpha = {np.degrees(alpha):.2f} deg"]
    for k in range(len(f)):
        h_sim = H[i0, :, k]
        h_an = analytic(np.deg2rad(np.abs(az)), f[k], alpha)
        err_db = 20 * np.log10(abs(h_sim) / abs(h_an))
        err_ph = np.degrees(np.angle(h_sim / h_an))
        lines.append(f"{f[k]/1e3:5.0f} kHz (ka = {2*np.pi*f[k]/numcalc.C*A:.2f}): magnitude error max {abs(err_db).max():.2f} dB, "
                     f"rms {np.sqrt((err_db**2).mean()):.2f} dB; phase error max {abs(err_ph).max():.1f} deg, rms {np.sqrt((err_ph**2).mean()):.1f} deg")
        axs[0, k].plot(az, 20 * np.log10(abs(h_an)), "k-", label="Analytical")
        axs[0, k].plot(az, 20 * np.log10(abs(h_sim)), "r--", label="NumCalc")
        axs[1, k].plot(az, np.degrees(np.unwrap(np.angle(h_an))), "k-")
        axs[1, k].plot(az, np.degrees(np.unwrap(np.angle(h_sim))), "r--")
        axs[0, k].set_title(f"{f[k]/1e3:.0f} kHz (ka = {2*np.pi*f[k]/numcalc.C*A:.1f})")
        axs[1, k].set_xlabel("Azimuth (deg)")
    axs[0, 0].set_ylabel("Magnitude (dB re Free Field)")
    axs[1, 0].set_ylabel("Phase (deg)")
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "sphere_check.png", dpi=120)
    (out / "sphere_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
