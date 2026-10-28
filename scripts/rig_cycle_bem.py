"""Antiphase pinna cycle for the rig videos: over one period, the base joints of the two pinnae swing in opposite
phase, flex(t) = centre +- amp * sin(phi), and the tip (distal joint) bends along, distal flex = tip * sin(phi) on the
left and -tip * sin(phi) on the right. For every phase step the deformed mesh is checked and the left ear, right ear
and emitter are solved at three frequencies (as rig_sweep_bem.py). Resumable.
Usage: python scripts/rig_cycle_bem.py key [--check-only]
Output: <work>/rig/cycle_<key>.npz (phase, left/right base and tip angles, H [step, source, el, az, f], vertices)"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import gpubem, grid, paths, project, rig, sources

FREQS = np.array([40e3, 80e3, 120e3])
SOURCES = ("left", "right", "emitter")
N_STEP = 24
# centre and amplitude of the base flex, amplitude of the tip bend (deg), within the valid ranges of results/rig
CYCLE = {"mmicr": (5.0, 25.0, 30.0), "rform": (2.5, 20.0, 25.0)}


def pose_at(key, k):
    c, amp, tip = CYCLE[key]
    s = np.sin(2 * np.pi * k / N_STEP)
    lb, rb = c + amp * s, c - amp * s
    return ({"pinna_left.base": {"flex": float(lb)}, "pinna_right.base": {"flex": float(rb)},
             "pinna_left.distal": {"flex": float(tip * s)}, "pinna_right.distal": {"flex": float(-tip * s)}},
            (lb, rb, tip * s, -tip * s))


def main():
    key = sys.argv[1]
    r = rig.load_rig(key)
    if "--check-only" in sys.argv:
        bad = [k for k in range(N_STEP) if not r.check(r.deform(pose_at(key, k)[0]), pose_at(key, k)[0])["valid"]]
        print(key, "invalid steps:", bad)
        return
    out = paths.work_dir() / "rig" / f"cycle_{key}.npz"
    pts, _ = grid.unique_points("sphere")
    old = dict(np.load(out, allow_pickle=True)) if out.exists() else {}
    H = old.get("H", np.zeros((N_STEP, 3, len(grid.EL), len(grid.GRIDS["sphere"][0]), len(FREQS)), np.complex64))
    V = old.get("vertices", np.zeros((N_STEP, len(r.rest.vertices), 3), np.float32))
    ok = old.get("valid", np.zeros(N_STEP, bool))
    fin = old.get("finished", np.zeros(N_STEP, bool))
    ang = np.array([pose_at(key, k)[1] for k in range(N_STEP)])
    for k in range(N_STEP):
        if fin[k]:
            continue
        t0 = time.time()
        pose = pose_at(key, k)[0]
        m = r.deform(pose)
        ok[k] = r.check(m, pose)["valid"]
        lm = r.deform_landmarks(pose, m)
        patches = {"left": [sources.disc_patch(m, lm["ear_left"], project.EAR_RADIUS)],
                   "right": [sources.disc_patch(m, lm["ear_right"], project.EAR_RADIUS)],
                   "emitter": [sources.disc_patch(m, lm[n], project.NOSTRIL_RADIUS) for n in ("nostril_0", "nostril_1")]}
        Hs = gpubem.solve_lu(m, patches, FREQS, pts, log=lambda s: None)
        for i, s in enumerate(SOURCES):
            H[k, i] = gpubem.to_grid(Hs[s], "sphere").astype(np.complex64)
        V[k] = m.vertices.astype(np.float32)
        fin[k] = True
        np.savez(out, phase_deg=np.arange(N_STEP) * 360 / N_STEP, angles=ang, freqs=FREQS, sources=np.array(SOURCES),
                 H=H, vertices=V, valid=ok, finished=fin, faces=r.rest.faces, cycle=np.array(CYCLE[key]))
        print(f"{key} step {k}: L/R base {ang[k, 0]:+.1f}/{ang[k, 1]:+.1f}, valid {ok[k]}, {time.time() - t0:.0f} s "
              f"({fin.sum()}/{N_STEP})", flush=True)


if __name__ == "__main__":
    main()
