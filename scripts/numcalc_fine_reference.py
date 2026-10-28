"""Finer-mesh reference for the discretization error at production density (referee item M1.2), on the CPU.

The CT mesh of one species is remeshed with the production pipeline (convergence_study.make_mesh, i.e.
remesh_ct.remesh_isolated) at FACTOR x the production target edge length, clearly finer than production and beyond the
GPU memory limit. NumCalc (Mesh2HRTF 1.3.0, ML-FMM, CPU) solves the nominal left ear and emitter (disc patches at the
config landmarks, radii project.EAR_RADIUS / NOSTRIL_RADIUS, as in production) on the full-sphere 2.5 deg grid. The
same is done with NumCalc on the production (dataset) mesh, so the comparison
  (a) GPU production (<work>/ertf/<key>_ct.h5)        vs NumCalc fine
  (b) NumCalc production mesh                          vs NumCalc fine   (same solver: pure discretization error)
  (c) GPU production                                   vs NumCalc production mesh (solver difference, same mesh)
separates the discretization error from the solver difference. Metrics over the full sphere, solid-angle weighted
(cos el), each pattern normalised to its own maximum: correlation and rms difference of the dB patterns (-30 dB
floor), relative complex error |H - H_ref| / |H_ref| (2-norm), and rms phase difference over the directions where the
reference is above its maximum - 30 dB. The same metrics on the frontal hemisphere (|az| <= 90 deg) are also stored.

Usage: python scripts/numcalc_fine_reference.py mesh  [factor ...]     remesh (cached in <work>/convergence)
       python scripts/numcalc_fine_reference.py run   [factor] [--par N] [--freqs 80,160,120]
                                                                      write + run the NumCalc projects (resumable;
                                                                      factor 1.0 = the production dataset mesh)
       python scripts/numcalc_fine_reference.py run sub [--par N]      production mesh subdivided 1:4 (same geometry)
       python scripts/numcalc_fine_reference.py compare [factor ...]  -> results/validation/numcalc_fine.json
                                                                      (all finished levels; default 0.7 0.5)
Projects: <work>/numcalc_fine/<key>_x<factor>/"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")          # the production file is open in another process
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import grid, mesh, numcalc, paths, project, sources

KEY = "mmicr"
FACTOR = 0.5
FREQS = [160e3, 80e3, 120e3]                     # in order of priority
SOURCES = ("left", "emitter")
GRID = "sphere"
OUT = paths.ROOT / "results" / "validation" / "numcalc_fine.json"


SUB = "sub"                                      # level tag: production mesh, midpoint-subdivided once


def proj_dir(fct):
    return paths.work_dir() / "numcalc_fine" / (f"{KEY}_{SUB}" if fct == SUB else f"{KEY}_x{fct:g}")


def get_mesh(fct):
    """Production dataset mesh for fct == 1; SUB: the production mesh with every face split into 4 (midpoints, same
    geometry and same source patches, edges halved: isolates the BEM discretization error from the geometric
    differences between remeshes); else the remeshed CT mesh at fct x the production target edge."""
    sp = paths.species()[KEY]
    if fct == 1.0:
        return mesh.load(sp["stl"])
    if fct == SUB:
        import trimesh
        m = mesh.load(sp["stl"])
        v, f = trimesh.remesh.subdivide(m.vertices, m.faces)    # face i -> faces 4i .. 4i+3
        return trimesh.Trimesh(v, f, process=False)
    import convergence_study
    L0 = json.loads((paths.ROOT / "data" / "meshes" / "dataset" / "meshes.json").read_text())[KEY]["target_edge_mm"]
    return convergence_study.make_mesh(KEY, L0 * fct)


def patches_for(m, fct=None):
    if fct == SUB:                                  # children of the production patches
        p = patches_for(get_mesh(1.0))
        return {s: [(4 * np.asarray(f)[:, None] + np.arange(4)).ravel() for f in fs] for s, fs in p.items()}
    lm = paths.species()[KEY]["landmarks"]
    p = {"left": [sources.disc_patch(m, np.array(lm["ear_left"]), project.EAR_RADIUS)],
         "emitter": [sources.disc_patch(m, np.array(q), project.NOSTRIL_RADIUS) for q in lm["nostrils"]]}
    return {s: p[s] for s in SOURCES}


def mesh_stats(m):
    e = m.edges_unique_length
    return {"n_faces": int(len(m.faces)), "edge_mm_mean_min_max": [e.mean() * 1e3, e.min() * 1e3, e.max() * 1e3],
            "elements_per_wavelength": {f"{f/1e3:.0f}": float(numcalc.C / f / e.mean()) for f in sorted(FREQS)}}


def cmd_mesh(factors):
    for fct in factors:
        t0 = time.time()
        m = get_mesh(fct)
        if m is None:
            print(f"x{fct}: remeshing not clean", flush=True)
            continue
        print(f"x{fct}: {json.dumps(mesh_stats(m))} ({time.time() - t0:.0f} s)", flush=True)


def cmd_run(fct, n_par, freqs):
    d = proj_dir(fct)
    if not (d / "project.json").exists():
        m = get_mesh(fct)
        numcalc.write_project(d, m, patches_for(m, fct), freqs, grid_name=GRID)
        (d / "mesh_stats.json").write_text(json.dumps(mesh_stats(m), indent=1))
    meta = json.loads((d / "project.json").read_text())
    print(f"{d.name}: {meta['n_faces']} faces, freqs {meta['freqs']}", flush=True)
    jobs = numcalc.steps(d)
    for sd, k in jobs:                                              # outputs of killed runs are empty: redo them
        f = sd / "be.out" / f"be.{k}" / "pEvalGrid"
        if f.exists() and f.stat().st_size == 0:
            f.unlink()
    order = {f: i for i, f in enumerate(freqs)}                     # priority order instead of high-first
    jobs.sort(key=lambda j: order.get(meta["freqs"][j[1] - 1], 99))
    todo = [j for j in jobs if not numcalc.done(*j)]
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    t0 = time.time()

    def one(job):
        sd, k = job
        ts = time.time()
        r = subprocess.run([str(paths.local()["numcalc"]), "-istart", str(k), "-iend", str(k)], cwd=sd,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        ok = numcalc.done(sd, k)
        print(f"{d.name}/{sd.name} {meta['freqs'][k-1]/1e3:.0f} kHz: {'ok' if ok else 'FAILED rc=%d' % r.returncode}"
              f" ({time.time() - ts:.0f} s, elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
        return ok

    with ThreadPoolExecutor(n_par) as ex:
        n_ok = sum(ex.map(one, todo))
    print(f"{d.name}: {n_ok}/{len(todo)} steps", flush=True)


def metrics(a, b, w, floor_db=-30.0):
    """a, b complex (P,) on the grid directions, b the reference; w (P,) solid-angle weights."""
    w = w / w.sum()
    da = np.maximum(20 * np.log10(abs(a) / abs(a).max()), floor_db)
    db = np.maximum(20 * np.log10(abs(b) / abs(b).max()), floor_db)
    am, bm = da - (w * da).sum(), db - (w * db).sum()
    keep = db > floor_db
    ph = np.angle(a[keep] / b[keep])
    lev = 20 * np.log10(abs(a[keep]) / abs(b[keep]))
    return {"corr": float((w * am * bm).sum() / np.sqrt((w * am ** 2).sum() * (w * bm ** 2).sum())),
            "rms_db": float(np.sqrt((w * (da - db) ** 2).sum())),
            "rel_complex_err": float(np.sqrt((w * abs(a - b) ** 2).sum() / (w * abs(b) ** 2).sum())),
            "phase_rms_deg": float(np.degrees(np.sqrt((w[keep] * ph ** 2).sum() / w[keep].sum()))),
            "level_rms_db_above_floor": float(np.sqrt((w[keep] * lev ** 2).sum() / w[keep].sum()))}


def read_numcalc(fct):
    d = proj_dir(fct)
    if not (d / "project.json").exists():
        return None, None
    meta = json.loads((d / "project.json").read_text())
    ok = [k + 1 for k in range(len(meta["freqs"]))
          if all((d / "NumCalc" / s / "be.out" / f"be.{k + 1}" / "pEvalGrid").exists() and
                 (d / "NumCalc" / s / "be.out" / f"be.{k + 1}" / "pEvalGrid").stat().st_size > 0 for s in SOURCES)]
    if not ok:
        return None, None
    H = {}
    for s in SOURCES:                                               # read_source skips missing steps (NaN)
        f, h = _read_steps(d, s, ok)
        H[s] = h
    return f, H


def _read_steps(d, s, ok):
    """numcalc.read_source restricted to the finished steps `ok` (1-based)."""
    meta = json.loads((d / "project.json").read_text())
    _, back = grid.unique_points(meta["grid"])
    freqs = np.array(meta["freqs"])[np.array(ok) - 1]
    q = meta["velocity"] * meta["patch_area"][s]
    out = np.full(back.shape + (len(ok),), np.nan, complex)
    for j, k in enumerate(ok):
        p = numcalc.read_pressure(d / "NumCalc" / s, k)
        w = 2 * np.pi * freqs[j]
        kr = w / meta["c"] * meta["radius"]
        p_ff = -1j * w * meta["rho"] * q * np.exp(1j * kr) / (4 * np.pi * meta["radius"])
        out[..., j] = np.conj(p[back] / p_ff)
    return freqs, out


def cmd_compare(fine_factors):
    """All comparisons available on disk: every finished fine level, the production mesh with NumCalc, the GPU."""
    import h5py
    az, el = grid.GRIDS[GRID]
    W = np.clip(np.cos(np.deg2rad(el)), 0, None)[:, None] * np.ones(len(az))    # poles: weight ~0, negligible
    front = np.broadcast_to(np.abs(az)[None, :] <= 90, W.shape)
    nc = {}
    for fct in [1.0] + list(fine_factors):
        fr, H = read_numcalc(fct)
        if fr is not None:
            nc[fct] = (fr, H, json.loads((proj_dir(fct) / "mesh_stats.json").read_text()))
    freqs = sorted({float(f) for v in nc.values() for f in v[0]}, reverse=True)
    with h5py.File(paths.work_dir() / "ertf" / f"{KEY}_ct.h5", "r") as h:
        fg, done = h["freqs"][:], h["done"][:]
        G = {}
        for f in freqs:
            i = int(np.argmin(abs(fg - f)))
            if abs(fg[i] - f) < 1 and done[i]:
                G[f] = {s: h[s]["H"][:, :, i].astype(complex) for s in SOURCES}
        prod_faces = int(h.attrs["n_faces"])

    def get(fct):
        if fct == "gpu":
            return lambda f, s: G[f][s] if f in G else None
        fr, H, _ = sub if fct == SUB else nc[fct]
        return lambda f, s: H[s][..., int(np.argmin(abs(fr - f)))] if np.min(abs(fr - f)) < 1 else None

    sub = None
    fr, H = read_numcalc(SUB)
    if fr is not None:
        sub = (fr, H, json.loads((proj_dir(SUB) / "mesh_stats.json").read_text()))
    finest = min(nc) if nc else None
    res = {"species": paths.species()[KEY]["taxon"], "key": KEY, "grid": GRID,
           "fine_factor": finest if finest is not None and finest < 1 else None,
           "fine_mesh": nc[finest][2] if finest is not None and finest < 1 else None,
           "meshes": {f"x{k:g}": v[2] for k, v in nc.items()},
           "production_mesh": nc[1.0][2] if 1.0 in nc else {"n_faces": prod_faces},
           "subdivided_mesh": sub[2] if sub else None,
           "gpu_production_freqs_done_khz": [f / 1e3 for f in sorted(G)],
           "metrics": "full sphere (and frontal hemisphere), solid-angle weighted; dB patterns normalised to their "
                      "maximum, -30 dB floor; phase and level rms over directions where the reference is above -30 dB",
           "comparisons": {}}

    def cmp(name, A, B):
        out = {}
        for s in SOURCES:
            for f in freqs:
                a, b = A(f, s), B(f, s)
                if a is None or b is None or np.isnan(a).any() or np.isnan(b).any():
                    continue
                out.setdefault(s, {})[f"{f/1e3:.0f}"] = {"full": metrics(a.ravel(), b.ravel(), W.ravel()),
                                                         "front": metrics(a[front], b[front], W[front])}
        if out:
            res["comparisons"][name] = out

    for fct in sorted(k for k in nc if k < 1):
        tag = f"numcalc_fine" if fct == finest else f"numcalc_x{fct:g}"
        cmp(f"gpu_production_vs_{tag}", get("gpu"), get(fct))
        if 1.0 in nc:
            cmp(f"numcalc_production_vs_{tag}", get(1.0), get(fct))
        for c in sorted(k for k in nc if fct < k < 1):
            cmp(f"numcalc_x{c:g}_vs_{tag}", get(c), get(fct))
    if 1.0 in nc:
        cmp("gpu_production_vs_numcalc_production", get("gpu"), get(1.0))
    if sub:                                         # same geometry and patches, edges halved
        cmp("gpu_production_vs_numcalc_subdivided", get("gpu"), get(SUB))
        if 1.0 in nc:
            cmp("numcalc_production_vs_numcalc_subdivided", get(1.0), get(SUB))
        for fct in sorted(k for k in nc if k < 1):
            cmp(f"numcalc_subdivided_vs_numcalc_x{fct:g}", get(SUB), get(fct))
    OUT.write_text(json.dumps(res, indent=1))
    for name, c in res["comparisons"].items():
        for s, v in c.items():
            for f, m in sorted(v.items(), key=lambda x: float(x[0])):
                x = m["full"]
                print(f"{name:42s} {s:8s} {f:>4s} kHz  corr {x['corr']:.4f}  rms {x['rms_db']:.2f} dB  "
                      f"rel {x['rel_complex_err']:.3f}  phase {x['phase_rms_deg']:.1f} deg  "
                      f"level {x['level_rms_db_above_floor']:.2f} dB")
    print(OUT)


def main():
    args = sys.argv[1:]
    n_par = 8
    freqs = FREQS
    if "--par" in args:
        i = args.index("--par"); n_par = int(args[i + 1]); del args[i:i + 2]
    if "--freqs" in args:
        i = args.index("--freqs"); freqs = [float(x) * 1e3 for x in args[i + 1].split(",")]; del args[i:i + 2]
    cmd = args[0] if args else "compare"
    fl = [x if x == SUB else float(x) for x in args[1:]]
    if cmd == "mesh":
        cmd_mesh(fl or [FACTOR])
    elif cmd == "run":
        cmd_run(fl[0] if fl else FACTOR, n_par, freqs)
    else:
        cmd_compare(fl or [0.7, FACTOR])


if __name__ == "__main__":
    main()
