"""Pinna sweep for the rig video: both pinnae of one species flexed symmetrically about their base joints over a range
of angles; for every pose the deformed mesh is checked (Rig.check) and the left ear, right ear and emitter are solved
with gpubem.solve_lu at three frequencies on the full-sphere grid. Sources as in scripts/run_ertf.build, at the
deformed landmarks. Resumable: finished poses are kept in the output file.
Usage: python scripts/rig_sweep_bem.py [key] [deg_min deg_max deg_step]   (default mmicr -20 25 1.5)
Output: <work>/rig/sweep_<key>.npz (angles, freqs, H [pose, source, el, az, f], deformed vertices, validity)"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import gpubem, grid, paths, project, rig, sources

FREQS = np.array([40e3, 80e3, 120e3])
SOURCES = ("left", "right", "emitter")


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "mmicr"
    lo, hi, step = (float(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (-20.0, 25.0, 1.5)
    angles = np.round(np.arange(lo, hi + step / 2, step), 3)
    out = paths.work_dir() / "rig" / f"sweep_{key}.npz"
    r = rig.load_rig(key)
    pts, _ = grid.unique_points("sphere")
    done = dict(np.load(out, allow_pickle=True)) if out.exists() else {}
    H = done.get("H", np.zeros((len(angles), len(SOURCES), len(grid.EL), len(grid.GRIDS["sphere"][0]), len(FREQS)),
                                np.complex64))
    V = done.get("vertices", np.zeros((len(angles), len(r.rest.vertices), 3), np.float32))
    ok = done.get("valid", np.zeros(len(angles), bool))
    finished = done.get("finished", np.zeros(len(angles), bool))
    for i, a in enumerate(angles):
        if finished[i]:
            continue
        pose = {"pinna_both.base": {"flex": float(a)}}
        t0 = time.time()
        m = r.deform(pose)
        ok[i] = r.check(m, pose)["valid"]
        lm = r.deform_landmarks(pose, m)
        patches = {"left": [sources.disc_patch(m, lm["ear_left"], project.EAR_RADIUS)],
                   "right": [sources.disc_patch(m, lm["ear_right"], project.EAR_RADIUS)],
                   "emitter": [sources.disc_patch(m, np.asarray(p), project.NOSTRIL_RADIUS) for p in (lm["nostril_0"], lm["nostril_1"])]}
        Hs = gpubem.solve_lu(m, patches, FREQS, pts, log=lambda s: None)
        for k, s in enumerate(SOURCES):
            H[i, k] = gpubem.to_grid(Hs[s], "sphere").astype(np.complex64)
        V[i] = m.vertices.astype(np.float32)
        finished[i] = True
        np.savez(out, angles=angles, freqs=FREQS, sources=np.array(SOURCES), H=H, vertices=V, valid=ok,
                 finished=finished, faces=r.rest.faces)
        print(f"{key} flex {a:+.1f}: valid {ok[i]}, {time.time() - t0:.0f} s ({finished.sum()}/{len(angles)})", flush=True)


if __name__ == "__main__":
    main()
