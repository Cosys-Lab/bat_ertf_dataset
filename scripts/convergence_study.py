"""Mesh convergence and angular sampling check (Technical Validation).

For a few species, the CT mesh is remeshed with the production pipeline (scripts/remesh_ct.py) at target edge
lengths of FACTORS times the production edge length, from coarse to slightly finer than production (the finest
level is limited by the GPU memory, about 68,000 faces). The nominal ears and nostrils (config landmarks) are solved
at a few frequencies up to the top of the band, on an equal-area (Fibonacci) set of directions, and every level is
compared with the finest one: correlation and rms difference of the dB patterns (each normalised to its own maximum,
clipped at -30 dB), and the relative complex error |H - H_ref| / |H_ref| (2-norm over directions; includes level and
phase).
On the production mesh, the directions of the 2.5 deg grid are solved as well, and the bilinear interpolation of the
grid (as a user of the dataset would do) is compared with the directly computed Fibonacci directions. Also on the
production mesh, the Fibonacci directions are solved at R_FAR = 20 m, to check how far the referenced transfer function
at 1 m (the dataset) is from the far-field directivity.
Usage: python scripts/convergence_study.py [key ...]   -> results/validation/convergence.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import gpubem, grid, paths, project, rotate, sources
import remesh_ct

KEYS = ["mmicr", "pdisc", "rform"]
FACTORS = [1.6, 1.25, 1.0, 0.93]       # x production target edge length; 1.0 = production settings
F_MAX_FACES = 68000
N_DIR = 4000
R_FAR = 20.0
OUT = paths.ROOT / "results" / "validation" / "convergence.json"


def fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


def make_mesh(key, L_mm):
    """Remesh the CT mesh at target edge L_mm with the production pipeline (cached in <work>/convergence)."""
    d = paths.work_dir() / "convergence"
    d.mkdir(exist_ok=True)
    fn = d / f"{key}_L{L_mm:.4f}.stl"
    if fn.exists():
        m = trimesh.load(fn, process=False)
        m.merge_vertices()      # STL stores every face with its own vertices: without merging, face_adjacency is
        return m                # empty and sources.disc_patch returns only the seed face
    src = paths.ROOT / "data" / "meshes" / "Old" / remesh_ct.SOURCES[key]
    m, clean = remesh_ct.remesh_isolated(src, L_mm)
    if not clean:
        print(f"{key} L={L_mm:.3f} mm: not clean, skipped", flush=True)
        return None
    m.apply_scale(1e-3)
    T = np.eye(4)
    T[:3, :3] = remesh_ct.CT_TO_FRAME
    m.apply_transform(T)
    if m.volume < 0:
        m.invert()
    m.export(fn)
    return m


def pattern_metrics(a, b, floor_db=-30.0):
    """a, b: complex (P, F) on equal-area directions; b is the reference."""
    da = np.maximum(20 * np.log10(abs(a) / abs(a).max(0)), floor_db)
    db = np.maximum(20 * np.log10(abs(b) / abs(b).max(0)), floor_db)
    am, bm = da - da.mean(0), db - db.mean(0)
    return {"corr": ((am * bm).mean(0) / np.sqrt((am ** 2).mean(0) * (bm ** 2).mean(0))).tolist(),
            "rms_db": np.sqrt(((da - db) ** 2).mean(0)).tolist(),
            "rel_complex_err": (np.linalg.norm(a - b, axis=0) / np.linalg.norm(b, axis=0)).tolist()}


def run(key):
    sp = paths.species()[key]
    meta = json.loads((paths.ROOT / "data" / "meshes" / "dataset" / "meshes.json").read_text())[key]
    L0 = meta["target_edge_mm"]
    fmax = sp["band_khz"][1] * 1e3
    freqs = np.array([40e3, 80e3, 120e3, fmax])
    lm = sp["landmarks"]
    fib = fibonacci(N_DIR)
    grid_pts, _ = grid.unique_points("sphere")
    res = {"species": sp["name"], "freqs_hz": freqs.tolist(), "production_edge_mm": L0, "levels": []}
    H = {}
    for fct in FACTORS:
        L = L0 * fct
        m = make_mesh(key, L)
        if m is None or len(m.faces) > F_MAX_FACES:
            continue
        patches = {"left": [sources.disc_patch(m, np.array(lm["ear_left"]), project.EAR_RADIUS)],
                   "right": [sources.disc_patch(m, np.array(lm["ear_right"]), project.EAR_RADIUS)],
                   "emitter": [sources.disc_patch(m, np.array(p), project.NOSTRIL_RADIUS) for p in lm["nostrils"]]}
        pts = np.concatenate([fib, grid_pts, R_FAR * fib]) if fct == 1.0 else fib
        t0 = time.time()
        Hl = gpubem.solve_lu(m, patches, freqs, pts, log=lambda s: None)
        e = m.edges_unique_length
        lev = {"factor": fct, "target_edge_mm": L, "n_faces": int(len(m.faces)), "edge_mean_mm": float(e.mean() * 1e3),
               "elements_per_wavelength_at_fmax": float(343.0 / fmax / e.mean()), "solve_s": time.time() - t0}
        H[fct] = {s: v[:N_DIR] for s, v in Hl.items()}
        if fct == 1.0:                                  # angular sampling: grid interpolation vs direct solution
            samp = {}
            for s, v in Hl.items():
                Hg = gpubem.to_grid(v[N_DIR:N_DIR + len(grid_pts)], "sphere")
                ref_db = 20 * np.log10(abs(v[:N_DIR]) / abs(v[:N_DIR]).max(0))
                ip_db = 20 * np.log10(rotate.SphereSampler(abs(Hg))(fib) / abs(v[:N_DIR]).max(0))
                ok = ref_db > -30
                err = np.where(ok, abs(ip_db - ref_db), np.nan)
                samp[s] = {"rms_db": np.sqrt(np.nanmean(err ** 2, 0)).tolist(),
                           "p99_db": np.nanpercentile(err, 99, 0).tolist()}
            lev["grid_interpolation_vs_direct_above_-30dB"] = samp
            lev["1m_vs_far_field"] = {s: pattern_metrics(v[:N_DIR], v[-N_DIR:]) for s, v in Hl.items()}
        res["levels"].append(lev)
        print(f"{key} x{fct}: {lev['n_faces']} faces, {lev['elements_per_wavelength_at_fmax']:.1f} el/wavelength at "
              f"{fmax/1e3:.0f} kHz, {lev['solve_s']:.0f} s", flush=True)
    ref = min(H)                                        # finest level
    for lev in res["levels"]:
        lev["vs_finest"] = {s: pattern_metrics(H[lev["factor"]][s], H[ref][s]) for s in H[ref]}
    res["reference_factor"] = ref
    return res


def main():
    keys = sys.argv[1:] or KEYS
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key in keys:
        out[key] = run(key)
        OUT.write_text(json.dumps(out, indent=1))
    for key in keys:
        for lev in out[key]["levels"]:
            v = lev["vs_finest"]
            print(key, lev["factor"], lev["n_faces"], {s: (np.round(v[s]["corr"], 3).tolist(),
                                                           np.round(v[s]["rel_complex_err"], 3).tolist()) for s in v})


if __name__ == "__main__":
    main()
