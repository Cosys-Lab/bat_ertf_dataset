"""Production ERTF run of one species on the GPU solver.

The mesh and landmarks are rotated into the reporting frame given by `rotation` (yaw, pitch, roll in degrees,
rotate.matrix convention: u_mesh = R u_frame) in config/species.toml [species.<key>.frames.<frame>], the source
patches are built from the landmarks, and the transfer functions of all sources are computed on the full-sphere grid,
in chunks of frequencies appended to an HDF5 file (resumable).
Usage: python scripts/run_ertf.py key frame [fmin_kHz fmax_kHz df_kHz] [--stride N]
(--stride N: only every N-th frequency of the grid, for coarse-to-fine passes; later runs fill in the rest)
Output: <work>/ertf/<key>_<frame>.h5"""
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import gpubem, grid, mesh, paths, piston, project, rotate, sources

CHUNK = 8


def frame_matrix(fr):
    """R (u_mesh = R u_frame) from a frame entry: rotation = [yaw, pitch, roll], or symmetry = [yaw, roll] plus a
    pitch about the symmetry frame's Y axis (keeps the midsagittal plane at Y = 0)."""
    if "matrix" in fr:
        return np.array(fr["matrix"], float)
    if "rotation" in fr:
        return rotate.matrix(*fr["rotation"])
    return rotate.matrix(fr["symmetry"][0], 0, fr["symmetry"][1]) @ rotate.matrix(0, fr["pitch"], 0)


def build(key, frame):
    sp = paths.species()[key]
    fr = sp["frames"][frame]
    R = frame_matrix(fr)
    m = mesh.load(sp["stl"])
    m.vertices = m.vertices @ R                      # v_frame = R^T v_mesh
    lm = {k: (np.asarray(v) @ R).tolist() for k, v in sp["landmarks"].items()}
    patches = {"left": [sources.disc_patch(m, np.array(lm["ear_left"]), project.EAR_RADIUS)],
               "right": [sources.disc_patch(m, np.array(lm["ear_right"]), project.EAR_RADIUS)]}
    if sp["emitter"] == "nasal":
        patches["emitter"] = [sources.disc_patch(m, np.array(p), project.NOSTRIL_RADIUS) for p in lm["nostrils"]]
    for name, alts in ensemble(key, sp["landmarks"]).items():   # placement ensemble (CT frame, docs/placement_rule.md)
        for i, a in enumerate(alts):
            a = np.asarray(a) @ R
            r = project.NOSTRIL_RADIUS if name == "emitter" else project.EAR_RADIUS
            patches[f"{name}_alt{i + 1}"] = [sources.disc_patch(m, p, r) for p in (a if a.ndim == 2 else [a])]
    return sp, R, m, lm, patches


def ensemble(key, lm, n_max=4, min_sep=0.5e-3):
    """Alternative ear and nostril positions (CT frame) from results/placement/<key>_rule.json: the rule's nominal
    and next-best distinct positions, without those within min_sep of the configured landmark."""
    f = paths.RESULTS / "placement" / f"{key}_rule.json"
    if not f.exists():
        return {}
    r = json.loads(f.read_text())
    out = {}
    for name, k, nominal in (("left", "left", lm["ear_left"]), ("right", "right", lm["ear_right"]),
                             ("emitter", "nostrils", lm.get("nostrils"))):
        if k not in r or nominal is None:
            continue
        cands = [r[k]["nominal"]] + r[k]["alternatives"]
        keep = [c for c in cands if np.linalg.norm(np.mean(np.atleast_2d(c), 0) - np.mean(np.atleast_2d(nominal), 0)) >= min_sep]
        out[name] = keep[:n_max]
    return out


def write_piston(h, sp, m, lm, freqs):
    """Analytical piston emitter of an oral species (piston.py), radius from config, axis = mouth normal."""
    centre = np.array(lm["mouth"])
    axis = piston.mouth_axis(m, centre)
    if "symmetry" in sp["frames"][h.attrs["frame"]]:     # symmetry-based frame: Y = 0 is the midsagittal plane
        axis[1] = 0
        axis /= np.linalg.norm(axis)
    u = grid.unit_vectors(*grid.GRIDS["sphere"])
    g = h.require_group("emitter")
    if "H" in g:
        del g["H"]
    g["H"] = piston.transfer(u, freqs, centre, axis, sp["piston_radius_m"]).astype(np.complex64)
    g.attrs.update(model="analytical piston with obliquity factor (batertf/piston.py)",
                   radius_m=sp["piston_radius_m"], centre_m=centre, axis=axis,
                   calibration=sp.get("piston_calibration", ""))
    for i, (a, note) in enumerate(zip(sp.get("piston_alternatives_m", []), sp.get("piston_alternatives_note", []))):
        g = h.require_group(f"emitter_alt{i + 1}")    # alternative radii (wide beam), same centre and axis
        if "H" in g:
            del g["H"]
        g["H"] = piston.transfer(u, freqs, centre, axis, a).astype(np.complex64)
        g.attrs.update(model="analytical piston with obliquity factor (batertf/piston.py)", radius_m=a, centre_m=centre,
                       axis=axis, calibration=note)


def main():
    stride = 1
    if "--stride" in sys.argv:                          # coarse-to-fine passes: only every stride-th frequency
        i = sys.argv.index("--stride")
        stride = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    key, frame = sys.argv[1], sys.argv[2]
    sp, R, m, lm, patches = build(key, frame)
    band = [float(x) * 1e3 for x in sys.argv[3:6]] if len(sys.argv) > 5 else [x * 1e3 for x in sp["band_khz"]]
    freqs = np.round(np.arange(band[0], band[1] + band[2] / 2, band[2]), 3)
    pts, back = grid.unique_points("sphere")
    az, el = grid.GRIDS["sphere"]
    out = paths.work_dir() / "ertf"
    out.mkdir(exist_ok=True)
    fn = out / f"{key}_{frame}.h5"
    with h5py.File(fn, "a") as h:
        if "freqs" not in h:
            h["freqs"] = freqs
            h["azimuth"], h["elevation"] = az, el
            for s, fs in patches.items():
                g = h.create_group(s)
                g.create_dataset("H", shape=(len(el), len(az), len(freqs)), dtype=np.complex64)
                g["patch_faces"] = np.concatenate(fs)
                g.attrs["patch_area_m2"] = float(sum(m.area_faces[f].sum() for f in fs))
                g.attrs["n_patches"] = len(fs)
            h["done"] = np.zeros(len(freqs), bool)
            h.attrs.update(species=sp["name"], key=key, frame=frame, R=R,
                           landmarks_frame_m=str(lm), radius_m=grid.RADIUS, n_faces=len(m.faces),
                           convention="exp(+j w t); H = p / p_monopole(same volume velocity, origin), 1 m",
                           git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                              cwd=paths.ROOT).stdout.strip())
        assert np.allclose(h["freqs"][:], freqs), "frequency grid differs from the existing file"
        if sp["emitter"] == "oral":
            write_piston(h, sp, m, lm, freqs)
        todo = np.where(~h["done"][:])[0]
    todo = todo[todo % stride == 0]
    t0 = time.time()
    for c in range(0, len(todo), CHUNK):            # the file is open only while a chunk is written, so readers
        idx = todo[c:c + CHUNK]                     # (figures, export) can open it between chunks
        H = gpubem.solve_lu(m, patches, freqs[idx], pts, log=lambda s: None)
        with h5py.File(fn, "a") as h:
            for s in patches:
                Hs = gpubem.to_grid(H[s], "sphere").astype(np.complex64)
                for j, k in enumerate(idx):
                    h[s]["H"][:, :, k] = Hs[..., j]
            d = h["done"][:]
            d[idx] = True
            h["done"][:] = d
        print(f"{key}/{frame}: {d.sum()}/{len(freqs)} frequencies ({freqs[idx[-1]]/1e3:.1f} kHz), "
              f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
