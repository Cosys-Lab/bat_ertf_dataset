"""Write Mesh2HRTF/NumCalc projects, run them in parallel, and read the results.

Project layout (same as Mesh2HRTF, so NumCalc and its tools work unchanged):
    <project>/ObjectMeshes/Reference/{Nodes,Elements}.txt
    <project>/EvaluationGrids/Hemisphere/{Nodes,Elements}.txt
    <project>/NumCalc/<source>/NC.inp          one folder per source (left, right, emitter)
    <project>/NumCalc/<source>/be.out/be.<k>/pEvalGrid   results per frequency step k (1-based)
    <project>/project.json                     faces of each source patch, frequencies, patch areas
NC.inp format: https://github.com/Any2HRTF/Mesh2HRTF/wiki/Structure_of_NC.inp (as written by mesh2input.py v1.3.0).
"""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import grid

C = 343.0
RHO = 1.1839
VELOCITY = 0.1          # normal velocity of the vibrating patches [m/s], as in Mesh2HRTF
GRID_START = 1_000_000  # node/element numbering offset of the evaluation grid


def write_project(project, m, patches, freqs, method_id=4, grid_name="hemisphere"):
    """patches: {source_name: [face index arrays...]}; a source with several arrays (nostrils) vibrates them in phase.
    Faces are reordered so every patch is a contiguous index range (NumCalc assigns boundary conditions by range)."""
    project = Path(project)
    all_patch = [np.asarray(f) for fs in patches.values() for f in fs]
    used = np.concatenate(all_patch) if all_patch else np.zeros(0, int)
    assert len(np.unique(used)) == len(used), "source patches overlap"
    rest = np.setdiff1d(np.arange(len(m.faces)), used)
    order = np.concatenate([rest] + all_patch)
    ranges, k = {}, len(rest)
    for name, fs in patches.items():
        ranges[name] = []
        for f in fs:
            ranges[name].append((k, k + len(f) - 1))
            k += len(f)
    faces = m.faces[order]

    d = project / "ObjectMeshes" / "Reference"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "Nodes.txt", "w", newline="\n") as f:
        f.write(f"{len(m.vertices)}\n")
        f.writelines(f"{i} {x:.9e} {y:.9e} {z:.9e}\n" for i, (x, y, z) in enumerate(m.vertices))
    with open(d / "Elements.txt", "w", newline="\n") as f:
        f.write(f"{len(faces)}\n")
        f.writelines(f"{i} {a} {b} {c} 0 0 0\n" for i, (a, b, c) in enumerate(faces))

    pts, _ = grid.unique_points(grid_name)
    d = project / "EvaluationGrids" / "Hemisphere"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "Nodes.txt", "w", newline="\n") as f:
        f.write(f"{len(pts)}\n")
        f.writelines(f"{GRID_START + i} {x:.9e} {y:.9e} {z:.9e}\n" for i, (x, y, z) in enumerate(pts))
    n_el = -(-len(pts) // 3)                     # meaningless triangles, as mesh2hrtf without triangulation
    with open(d / "Elements.txt", "w", newline="\n") as f:
        f.write(f"{n_el}\n")
        f.writelines(f"{GRID_START + i} {GRID_START + (3*i) % len(pts)} {GRID_START + (3*i+1) % len(pts)} "
                     f"{GRID_START + (3*i+2) % len(pts)} 2 0 1\n" for i in range(n_el))

    for name, rr in ranges.items():
        d = project / "NumCalc" / name
        d.mkdir(parents=True, exist_ok=True)
        _write_nc_inp(d / "NC.inp", name, freqs, len(faces) + n_el, len(m.vertices) + len(pts), rr, method_id)

    area = {n: float(sum(m.area_faces[f].sum() for f in fs)) for n, fs in patches.items()}
    meta = {"freqs": [float(x) for x in freqs], "ranges": ranges, "patch_area": area,
            "velocity": VELOCITY, "c": C, "rho": RHO, "radius": grid.RADIUS, "n_faces": len(faces),
            "grid": grid_name,
            "patch_faces_original": {n: [np.asarray(f).tolist() for f in fs] for n, fs in patches.items()}}
    (project / "project.json").write_text(json.dumps(meta))
    return meta


def _write_nc_inp(path, title, freqs, n_elem, n_nodes, ranges, method_id):
    L = ["##-------------------------------------------", "## Written by batertf (Mesh2HRTF 1.3.0 format)",
         "##-------------------------------------------", "Mesh2HRTF 1.3.0", "##", f"Bat ERTF {title}", "##",
         "## Controlparameter I", "0 0 0 0 7 0", "##",
         "## Controlparameter II", f"1 {len(freqs)} 0.000001 0.00e+00 1 0 0", "##",
         "## Load Frequency Curve", f"0 {len(freqs) + 1}", "0.000000 0.000000e+00 0.0"]
    L += [f"{0.000001 * (i + 1):f} {f / 10000:f}e+04 0.0" for i, f in enumerate(freqs)]
    L += ["##", "## 1. Main Parameters I", f"2 {n_elem} {n_nodes} 0 0 2 1 {method_id} 0", "##",
          "## 2. Main Parameters II", "0 0 0 0.0000e+00 0 0 0", "##",
          "## 3. Main Parameters III", "0 0 0 0", "##",
          "## 4. Main Parameters IV", f"{C} {RHO}e+00 1.0 0.0e+00 0.0 e+00 0.0e+00 0.0e+00", "##",
          "NODES", "../../ObjectMeshes/Reference/Nodes.txt", "../../EvaluationGrids/Hemisphere/Nodes.txt", "##",
          "ELEMENTS", "../../ObjectMeshes/Reference/Elements.txt", "../../EvaluationGrids/Hemisphere/Elements.txt",
          "##", "BOUNDARY"]
    L += [f"ELEM {a} TO {b} VELO {VELOCITY} -1 0.0 -1" for a, b in ranges]
    L += ["RETU", "##", "POST PROCESS", "##", "END"]
    Path(path).write_text("\n".join(L) + "\n")


def steps(project):
    """All (source_dir, step) pairs of a project, step 1-based."""
    project = Path(project)
    meta = json.loads((project / "project.json").read_text())
    return [(project / "NumCalc" / s, k + 1) for s in meta["ranges"] for k in range(len(meta["freqs"]))]


def done(src_dir, k):
    return (Path(src_dir) / "be.out" / f"be.{k}" / "pEvalGrid").exists()


def run_steps(jobs, numcalc, n_parallel, log=lambda s: print(s, flush=True), nitermax=None):
    """Run NumCalc for each (source_dir, step) not done yet, n_parallel processes at a time.
    Larger steps (higher frequencies) are started first so the tail of the run is short."""
    todo = [(d, k) for d, k in jobs if not done(d, k)]
    todo.sort(key=lambda j: -j[1])
    t0 = time.time()

    def one(job):
        d, k = job
        cmd = [str(numcalc), "-istart", str(k), "-iend", str(k)] + (["-nitermax", str(nitermax)] if nitermax else [])
        ts = time.time()
        r = subprocess.run(cmd, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        ok = done(d, k)
        log(f"{d.parent.parent.name}/{d.name} step {k}: {'ok' if ok else 'FAILED rc=%d %s' % (r.returncode, r.stderr[-200:])}"
            f" ({time.time() - ts:.0f} s, elapsed {(time.time() - t0) / 60:.1f} min)")
        return ok

    with ThreadPoolExecutor(n_parallel) as ex:
        res = list(ex.map(one, todo))
    return sum(res), len(todo)


def read_pressure(src_dir, k):
    """Complex pressure on the unique evaluation points for step k."""
    f = Path(src_dir) / "be.out" / f"be.{k}" / "pEvalGrid"
    rows = []
    with open(f) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 3 and not p[0].startswith("Mesh"):
                try:
                    rows.append((int(p[0]), float(p[1]), float(p[2])))
                except ValueError:
                    pass
    rows = np.array(rows)
    idx = rows[:, 0].astype(int) - GRID_START
    p = np.full(idx.max() + 1, np.nan, complex)
    p[idx] = rows[:, 1] + 1j * rows[:, 2]
    return p


def read_source(project, source):
    """Transfer function of one source on the [el, az, f] grid: pressure divided by the free-field pressure of a
    monopole at the origin with the same volume velocity (Mesh2HRTF referencing), so 1 means 0 dB re free field.
    NumCalc works in the exp(-i w t) convention (Mesh2HRTF reference_hrtfs.py); the result is conjugated to the
    exp(+j w t) convention of numpy's FFT, so irfft gives a causal impulse response (checked on a sphere,
    tests/sphere_check.py: the front of the sphere leads the side by about ka)."""
    project = Path(project)
    meta = json.loads((project / "project.json").read_text())
    _, back = grid.unique_points(meta.get("grid", "hemisphere"))
    freqs = np.array(meta["freqs"])
    q = meta["velocity"] * meta["patch_area"][source]
    out = np.full(back.shape + (len(freqs),), np.nan, complex)
    for k, f in enumerate(freqs):
        if not done(project / "NumCalc" / source, k + 1):
            continue
        p = read_pressure(project / "NumCalc" / source, k + 1)
        w = 2 * np.pi * f
        kr = w / meta["c"] * meta["radius"]
        p_ff = -1j * w * meta["rho"] * q * np.exp(1j * kr) / (4 * np.pi * meta["radius"])
        out[..., k] = np.conj(p[back] / p_ff)
    return freqs, out
