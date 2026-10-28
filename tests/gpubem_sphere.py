"""Validation of the GPU BEM (batertf/gpubem.py) against the series solution of tests/sphere_check.py: rigid sphere
a = 5 mm with a vibrating cap at +X, magnitude AND phase at 1 m, referenced and conjugated like numcalc.read_source.
1. Accuracy: icosphere subdivisions 4 and 5 (5120 / 20480 faces), 20, 34.3, 40, 80, 120 kHz, Burton-Miller + GMRES.
   Points on the circle z = 0, angles measured from the axis of the faceted cap (see angles()). Errors over the
   front half (theta <= 90 deg, as sphere_check.py) and the full circle (theta 0..180 deg).
2. Irregular frequency near ka = pi (34.3 kHz, first interior Dirichlet eigenfrequency; the discretised sphere
   shifts it slightly, so it is located as the minimum of the smallest singular value of the CBIE matrix): plain CBIE
   vs BM with alpha = +i/k and -i/k, solved directly (LU) so the near-singular CBIE is not masked by the Krylov solver.
Usage: python tests/gpubem_sphere.py  -> results/validation/gpubem_sphere.{png,txt}"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from scipy.optimize import minimize_scalar
from scipy.special import eval_legendre

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import gpubem, numcalc, paths, sources
from sphere_check import A, analytic, hankel

FREQS = [20000.0, 34300.0, 40000.0, 80000.0, 120000.0]
SWEEP = np.arange(33.8e3, 35.01e3, 0.1e3)          # smallest-singular-value sweep
TH = np.arange(0, 180.01, 1.0)


def surface_analytic(theta, f, alpha, n_max=120):
    k = 2 * np.pi * f / numcalc.C
    p = np.zeros_like(theta, complex)
    for n in range(n_max):
        pm1 = 1.0 if n == 0 else eval_legendre(n - 1, np.cos(alpha))
        vn = numcalc.VELOCITY / 2 * (pm1 - eval_legendre(n + 1, np.cos(alpha)))
        p += vn * hankel(n, k * A) / hankel(n, k * A, True) * eval_legendre(n, np.cos(theta))
    return 1j * numcalc.RHO * numcalc.C * p


def setup(sub):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=A)
    cap = sources.disc_patch(m, np.array([A, 0, 0]), 0.8e-3)
    alpha = np.arccos(1 - m.area_faces[cap].sum() / (2 * np.pi * A ** 2))
    axis = sources.patch_info(m, cap)["centroid"]
    return m, cap, alpha, axis / np.linalg.norm(axis)


def angles(pts, axis):
    """Angle from the cap axis. The faceted cap is centred 1-2 deg off +X; measuring from +X instead would add an
    error of up to ~1 dB in the deep shadow at ka = 11 that has nothing to do with the solver."""
    return np.arccos(np.clip(pts @ axis / np.linalg.norm(pts, axis=1), -1, 1))


def errors(h, h_an, sel=slice(None)):
    e = 20 * np.log10(abs(h) / abs(h_an))
    ph = np.degrees(np.angle(h / h_an))
    s = sel
    return abs(e[s]).max(), np.sqrt((e[s] ** 2).mean()), abs(ph[s]).max(), np.sqrt((ph[s] ** 2).mean())


def main():
    th = np.deg2rad(TH)
    pts = np.stack([np.cos(th), np.sin(th), 0 * th], 1)
    lines, res = [], {}
    lines.append("1. Burton-Miller (alpha = +i/k), GMRES tol 1e-5. Errors vs series solution: "
                 "max/rms dB, max/rms phase [deg]; front = theta <= 90 deg, full = 0..180 deg")
    for sub in (4, 5):
        m, cap, alpha, axis = setup(sub)
        tha = angles(pts, axis)
        H = gpubem.solve(m, {"cap": [cap]}, FREQS, pts, log=lambda s: None)["cap"]
        res[sub] = H
        lines.append(f"subdivision {sub}: {len(m.faces)} faces, mean edge {m.edges_unique_length.mean()*1e3:.3f} mm, "
                     f"cap {len(cap)} faces, alpha = {np.degrees(alpha):.2f} deg")
        for k, f in enumerate(FREQS):
            h_an = analytic(tha, f, alpha)
            ef, ea = errors(H[:, k], h_an, tha <= np.pi / 2), errors(H[:, k], h_an)
            lines.append(f"  {f/1e3:5.1f} kHz ka = {2*np.pi*f/numcalc.C*A:5.2f}  front {ef[0]:5.2f}/{ef[1]:5.2f} dB "
                         f"{ef[2]:5.1f}/{ef[3]:4.1f} deg   full {ea[0]:5.2f}/{ea[1]:5.2f} dB {ea[2]:5.1f}/{ea[3]:4.1f} deg")

    lines.append("2. Irregular frequency, subdivision 4, LU solve. f* = discrete CBIE eigenfrequency near ka = pi "
                 "(minimum of the smallest singular value of the CBIE matrix). Per formulation: far-field rms/max dB "
                 "error | error of the surface-mean pressure (the n = 0 term, where the spurious mode lives) | "
                 "smallest singular value of the Jacobi-scaled matrix")
    m, cap, alpha, axis = setup(4)
    tha = angles(pts, axis)
    w_a = m.area_faces
    c = m.triangles_center
    tc = angles(c, axis)
    asm = gpubem.Assembler(gpubem.Mesh(m))
    smin = lambda f, sg: torch.linalg.svdvals(asm.matrix(2 * np.pi * f / numcalc.C, sg * 1j * numcalc.C / (2 * np.pi * f))[0])[-1].item()
    f_star = minimize_scalar(lambda f: smin(f, 0), bracket=(34.2e3, 34.35e3, 34.5e3), tol=1e-7).x
    names = ["CBIE", "BM +i/k", "BM -i/k"]
    sw = {sg: [smin(f, sg) for f in SWEEP] for sg in (0, 1, -1)}
    fl = [34000.0, 34300.0, f_star, 34600.0]
    lines.append(f"  f* = {f_star:.1f} Hz (ka = {2*np.pi*f_star/numcalc.C*A:.4f}; nominal ka = pi at 34300 Hz)")
    lines.append("  f [Hz]    " + "".join(f"{n:>34s}" for n in names))
    rows = {f: "" for f in fl}
    for sg in (0, 1, -1):
        S = {}
        H = gpubem.solve(m, {"cap": [cap]}, fl, pts, alpha_sign=sg, surface=S, direct=True, log=lambda s: None)["cap"]
        for k, f in enumerate(fl):
            e = errors(H[:, k], analytic(tha, f, alpha))
            ps = surface_analytic(tc, f, alpha)
            es = abs((w_a * (S[f][:, 0] - ps)).sum() / (w_a * ps).sum())
            rows[f] += f"   {e[1]:6.2f}/{e[0]:6.2f} dB | {es:7.3f} | {smin(f, sg):.1e}"
    for f in fl:
        lines.append(f"  {f:8.1f}{rows[f]}")
    del asm
    torch.cuda.empty_cache()

    fig, axs = plt.subplots(2, len(FREQS) + 1, figsize=(3.6 * (len(FREQS) + 1), 6.5))
    m, cap, alpha, axis = setup(5)
    tha5 = angles(pts, axis)
    for k, f in enumerate(FREQS):
        h_an = analytic(tha5, f, alpha)
        axs[0, k].plot(TH, 20 * np.log10(abs(h_an)), "k-", label="Analytical")
        axs[1, k].plot(TH, np.degrees(np.unwrap(np.angle(h_an))), "k-")
        for sub, st in ((4, "b:"), (5, "r--")):
            axs[0, k].plot(TH, 20 * np.log10(abs(res[sub][:, k])), st, label=f"GPU BEM sub {sub}")
            axs[1, k].plot(TH, np.degrees(np.unwrap(np.angle(res[sub][:, k]))), st)
        axs[0, k].set_title(f"{f/1e3:.1f} kHz (ka = {2*np.pi*f/numcalc.C*A:.2f})")
        axs[1, k].set_xlabel("Azimuth in z = 0 (deg)")
    axs[0, 0].set_ylabel("Magnitude (dB re Free Field)")
    axs[1, 0].set_ylabel("Phase (deg)")
    axs[0, 0].legend(fontsize=8)
    for sg, st, n in zip((0, 1, -1), ("k-o", "r-s", "b-^"), names):
        axs[0, -1].semilogy(SWEEP / 1e3, sw[sg], st, ms=3, label=n)
    axs[0, -1].axvline(f_star / 1e3, color="0.5", lw=0.8)
    axs[0, -1].set_title("Smallest singular value (sub 4)")
    axs[0, -1].set_xlabel("Frequency (kHz)")
    axs[0, -1].legend(fontsize=8)
    S = {}
    m4, cap4, a4, _ = setup(4)
    H = gpubem.solve(m4, {"cap": [cap4]}, [f_star], pts, alpha_sign=0, direct=True, log=lambda s: None)["cap"][:, 0]
    axs[1, -1].plot(TH, 20 * np.log10(abs(analytic(tha, f_star, a4))), "k-", label="Analytical")
    axs[1, -1].plot(TH, 20 * np.log10(abs(H)), "k:", label="CBIE")
    Hb = gpubem.solve(m4, {"cap": [cap4]}, [f_star], pts, direct=True, log=lambda s: None)["cap"][:, 0]
    axs[1, -1].plot(TH, 20 * np.log10(abs(Hb)), "r--", label="BM +i/k")
    axs[1, -1].set_title(f"sub 4 at f* = {f_star/1e3:.3f} kHz")
    axs[1, -1].set_xlabel("Azimuth in z = 0 (deg)")
    axs[1, -1].legend(fontsize=8)
    fig.tight_layout()
    out = paths.RESULTS / "validation"
    fig.savefig(out / "gpubem_sphere.png", dpi=110)
    (out / "gpubem_sphere.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
