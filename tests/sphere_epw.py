"""Discretization error of the GPU BEM against the exact solution as a function of elements per wavelength
(referee item M1.1). Rigid sphere a = 5 mm with a vibrating cap of radius 0.8 mm at +X (sources.disc_patch, analytic
cap with the patch's area, angles from the faceted cap axis; as tests/sphere_check.py, tests/gpubem_sphere.py and
paper/figures/fig_validation.py). Meshes: geodesic spheres (icosahedron, every face split into n^2 triangles,
vertices projected onto the sphere) and the trimesh icospheres of subdivision 3, 4, 5 (1,280 / 5,120 / 20,480 faces;
subdivision 4 is the 5,120-face sphere of tests/gpubem_sphere.py), all <= 20,480 faces. Each mesh is solved with
gpubem.solve_lu at FREQS (ka = 3.7 ... 22), so elements per wavelength (wavelength / mean edge) spans about 2-45 and
the band 3-15 is covered at every ka. The problem depends only on ka and edge/wavelength (and the cap), so a larger
sphere at a lower frequency is the same test.
Metrics over N_DIR equal-area (Fibonacci) directions at 1 m on the full sphere, against the series solution:
magnitude error 20 log10|H/H_exact| (rms, max), phase error (rms, max; deg), both over all directions and over the
directions where the exact pattern is above its maximum - 30 dB ("_above_floor"; the smooth source has a shadow more
than 60 dB down, where all-direction dB and phase errors are meaningless), pattern correlation and rms difference of
the dB patterns (each normalised to its maximum, -30 dB floor, as scripts/convergence_study.py), relative complex
error |H - H_exact| / |H_exact| (2-norm). Errors at exactly 6 elements per wavelength are interpolated (log-log in
elements per wavelength) per frequency.
Usage: python tests/sphere_epw.py  -> results/validation/sphere_epw.json"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from scipy.special import eval_legendre
from batertf import gpubem, numcalc, paths, sources
from sphere_check import A, analytic, hankel

CAP_RADIUS = 0.8e-3
FREQS = [40e3, 80e3, 120e3, 160e3, 240e3]
GEODESIC_N = [6, 7, 10, 12, 14, 20, 24, 28]          # geodesic frequencies (20 n^2 faces)
ICO_SUB = [3, 4, 5]                                  # trimesh icospheres: 1280, 5120, 20480 faces
N_DIR = 2000
EPW_TARGETS = [4.0, 6.0, 8.0, 10.0]
SMOOTH_WIDTH = np.deg2rad(10.0)                      # Gaussian velocity profile v = V exp(-theta^2 / (2 w^2))
OUT = paths.ROOT / "results" / "validation" / "sphere_epw.json"


def geodesic(n, radius=A):
    ico = trimesh.creation.icosahedron()
    V, F = [], []
    idx = {}
    for f in ico.faces:
        a, b, c = ico.vertices[f]
        loc = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                p = a + (b - a) * i / n + (c - a) * j / n
                p = p / np.linalg.norm(p)
                key = tuple(np.round(p, 9))
                if key not in idx:
                    idx[key] = len(V)
                    V.append(p)
                loc[i, j] = idx[key]
        for i in range(n):
            for j in range(n - i):
                F.append([loc[i, j], loc[i + 1, j], loc[i, j + 1]])
                if i + j < n - 1:
                    F.append([loc[i + 1, j], loc[i + 1, j + 1], loc[i, j + 1]])
    m = trimesh.Trimesh(np.array(V) * radius, np.array(F), process=True)
    if m.volume < 0:
        m.invert()
    return m


def fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


def metrics(h, ha, floor_db=-30.0):
    e = 20 * np.log10(abs(h) / abs(ha))
    ph = np.degrees(np.angle(h / ha))
    da = np.maximum(20 * np.log10(abs(h) / abs(h).max()), floor_db)
    db = np.maximum(20 * np.log10(abs(ha) / abs(ha).max()), floor_db)
    am, bm = da - da.mean(), db - db.mean()
    hi = db > floor_db                                   # directions where the exact field is above max - 30 dB
    return {"mag_rms_db_above_floor": float(np.sqrt((e[hi] ** 2).mean())),
            "mag_max_db_above_floor": float(abs(e[hi]).max()),
            "phase_rms_deg_above_floor": float(np.sqrt((ph[hi] ** 2).mean())),
            "phase_max_deg_above_floor": float(abs(ph[hi]).max()), "fraction_above_floor": float(hi.mean()),
            "mag_rms_db": float(np.sqrt((e ** 2).mean())), "mag_max_db": float(abs(e).max()),
            "phase_rms_deg": float(np.sqrt((ph ** 2).mean())), "phase_max_deg": float(abs(ph).max()),
            "pattern_corr": float((am * bm).mean() / np.sqrt((am ** 2).mean() * (bm ** 2).mean())),
            "pattern_rms_db": float(np.sqrt(((da - db) ** 2).mean())),
            "rel_complex_err": float(np.linalg.norm(h - ha) / np.linalg.norm(ha))}


def smooth_velocity(theta):
    return numcalc.VELOCITY * np.exp(-theta ** 2 / (2 * SMOOTH_WIDTH ** 2))


def smooth_analytic(theta, f, r=1.0, n_max=120):
    """Series solution for the axisymmetric smooth velocity profile (Legendre coefficients by Gauss quadrature)."""
    x, wx = np.polynomial.legendre.leggauss(800)
    v = smooth_velocity(np.arccos(x))
    k = 2 * np.pi * f / numcalc.C
    p = np.zeros_like(theta, complex)
    for n in range(n_max):
        vn = (2 * n + 1) / 2 * (wx * v * eval_legendre(n, x)).sum()
        p += vn * hankel(n, k * r) / hankel(n, k * A, True) * eval_legendre(n, np.cos(theta))
    return 1j * numcalc.RHO * numcalc.C * p


def solve_smooth(m, freqs, pts, device="cuda"):
    """BEM (Burton-Miller, LU) with the smooth velocity profile sampled at the face centroids (q = i w rho v),
    same operators as gpubem.solve_lu. Returns the raw far-field pressure (P, F), exp(-i w t)."""
    mesh = gpubem.Mesh(m, device)
    asm = gpubem.Assembler(mesh)
    th = np.arccos(np.clip(mesh.c[:, 0] / np.linalg.norm(mesh.c, axis=1), -1, 1))
    v = smooth_velocity(th)
    cols = np.where(v > 1e-7 * v.max())[0]
    cols_t = torch.as_tensor(cols, device=device)
    pts_t = torch.as_tensor(pts, dtype=torch.float32, device=device)
    yq, wq = mesh.quad(gpubem._R3)
    out = np.zeros((len(pts), len(freqs)), complex)
    for kf, f in enumerate(freqs):
        w = 2 * np.pi * f
        k = w / numcalc.C
        alpha = gpubem.ALPHA_SIGN * 1j / k
        A_, scale = asm.matrix(k, alpha)
        qc = torch.as_tensor(1j * w * numcalc.RHO * v[cols], dtype=torch.complex64, device=device)[:, None]
        b = asm.rhs_matrix(k, alpha, cols) @ qc
        b[cols_t] -= alpha / 2 * qc
        b *= scale[:, None]
        p = torch.linalg.solve(A_, b)
        del A_
        q = torch.zeros(mesh.N, 1, dtype=torch.complex64, device=device)
        q[cols_t] = qc
        for s0 in range(0, len(pts), 64):
            dGdn, G = gpubem._kernel(pts_t[s0:s0 + 64, None, None], None, yq[None], mesh.tn[None, :, None],
                                     wq[None], k, 0, "F")
            out[s0:s0 + 64, kf] = (dGdn.to(torch.complex64) @ p - G.to(torch.complex64) @ q).cpu().numpy()[:, 0]
        torch.cuda.empty_cache()
    return out


def interp_epw(meshes, part, keys):
    out = {}
    for k, f in enumerate(FREQS):
        rows = sorted(((lev[part][k]["epw"], lev[part][k]) for lev in meshes), key=lambda x: x[0])
        e = np.log([x[0] for x in rows])
        d = {}
        for t in EPW_TARGETS:
            if not e[0] <= np.log(t) <= e[-1]:
                continue
            d[f"{t:g}"] = {q: float(np.exp(np.interp(np.log(t), e, np.log([max(x[1][q], 1e-9) for x in rows]))))
                           for q in keys}
            d[f"{t:g}"]["pattern_corr"] = float(np.interp(np.log(t), e, [x[1]["pattern_corr"] for x in rows]))
        out[f"{f/1e3:.0f}"] = d
    return out


def main():
    pts = fibonacci(N_DIR)
    meshes = [(f"geodesic n={n}", geodesic(n)) for n in GEODESIC_N]
    meshes += [(f"icosphere sub {s}", trimesh.creation.icosphere(subdivisions=s, radius=A)) for s in ICO_SUB]
    meshes.sort(key=lambda x: len(x[1].faces))
    res = {"setup": {"radius_m": A, "cap_radius_m": CAP_RADIUS, "freqs_hz": FREQS, "n_dir": N_DIR,
                     "ka": [2 * np.pi * f / numcalc.C * A for f in FREQS], "solver": "gpubem.solve_lu (Burton-Miller)",
                     "elements_per_wavelength": "c / (f * mean edge length)"}, "meshes": []}
    for name, m in meshes:
        assert len(m.faces) <= 20480 and m.is_watertight
        cap = sources.disc_patch(m, np.array([A, 0, 0]), CAP_RADIUS)
        alpha = np.arccos(1 - m.area_faces[cap].sum() / (2 * np.pi * A ** 2))
        ax = sources.patch_info(m, cap)["centroid"]
        th = np.arccos(np.clip(pts @ (ax / np.linalg.norm(ax)), -1, 1))
        t0 = time.time()
        H = gpubem.solve_lu(m, {"cap": [cap]}, FREQS, pts, log=lambda s: None)["cap"]
        edge = m.edges_unique_length.mean()
        lev = {"mesh": name, "n_faces": int(len(m.faces)), "edge_mean_mm": edge * 1e3, "cap_faces": int(len(cap)),
               "cap_radius_over_edge": CAP_RADIUS / edge, "solve_s": time.time() - t0, "results": []}
        for k, f in enumerate(FREQS):
            ha = analytic(th, f, alpha)
            r = {"f_hz": f, "ka": 2 * np.pi * f / numcalc.C * A, "epw": numcalc.C / f / edge, **metrics(H[:, k], ha)}
            front = th <= np.pi / 2
            r["front"] = metrics(H[front, k], ha[front])
            lev["results"].append(r)
            print(f"{name:18s} {len(m.faces):6d} f {f/1e3:5.0f} kHz ka {r['ka']:5.2f} epw {r['epw']:5.1f}: "
                  f"mag {r['mag_rms_db']:.3f}/{r['mag_max_db']:.2f} dB, phase {r['phase_rms_deg']:.2f}/"
                  f"{r['phase_max_deg']:.1f} deg, corr {r['pattern_corr']:.5f}, rel {r['rel_complex_err']:.4f}",
                  flush=True)
        th0 = np.arccos(np.clip(pts[:, 0], -1, 1))
        Hs = solve_smooth(m, FREQS, pts)
        lev["smooth"] = []
        for k, f in enumerate(FREQS):
            r = {"f_hz": f, "ka": 2 * np.pi * f / numcalc.C * A, "epw": numcalc.C / f / edge,
                 **metrics(Hs[:, k], smooth_analytic(th0, f))}
            lev["smooth"].append(r)
            print(f"{'  smooth':18s} {len(m.faces):6d} f {f/1e3:5.0f} kHz epw {r['epw']:5.1f}: "
                  f"mag {r['mag_rms_db']:.3f}/{r['mag_max_db']:.2f} dB, phase {r['phase_rms_deg']:.2f} deg, "
                  f"corr {r['pattern_corr']:.5f}, rel {r['rel_complex_err']:.4f}", flush=True)
        res["meshes"].append(lev)
    keys = ["mag_rms_db", "mag_max_db", "phase_rms_deg", "pattern_rms_db", "rel_complex_err",
            "mag_rms_db_above_floor", "mag_max_db_above_floor", "phase_rms_deg_above_floor"]
    res["at_epw"] = interp_epw(res["meshes"], "results", keys)
    res["at_epw_smooth"] = interp_epw(res["meshes"], "smooth", keys)
    # the 5,120-face sphere of tests/gpubem_sphere.py (circle z = 0, GMRES) for reference
    txt = paths.ROOT / "results" / "validation" / "gpubem_sphere.txt"
    if txt.exists():
        res["gpubem_sphere_txt_sub4"] = [l.strip() for l in txt.read_text().splitlines()
                                         if re.match(r"\s+\d+\.\d kHz", l)][:5]
    OUT.write_text(json.dumps(res, indent=1))
    for part in ("at_epw", "at_epw_smooth"):
        print(f"\nInterpolated at fixed elements per wavelength (full sphere), {part}:")
        for f, d in res[part].items():
            for t, v in d.items():
                print(f"  {f:>4s} kHz, {t:>3s} el/wl: above -30 dB: mag rms {v['mag_rms_db_above_floor']:.3f} dB, "
                      f"max {v['mag_max_db_above_floor']:.2f} dB, phase rms {v['phase_rms_deg_above_floor']:.2f} deg; "
                      f"all: mag rms {v['mag_rms_db']:.3f} dB, phase rms {v['phase_rms_deg']:.2f} deg, "
                      f"corr {v['pattern_corr']:.5f}, "
                      f"rel {v['rel_complex_err']:.4f}")
    print(OUT)


if __name__ == "__main__":
    main()
