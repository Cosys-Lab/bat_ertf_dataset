"""Acoustic reporting frame of a species (oral species: the mouth normal projected into the symmetry plane): midsagittal plane from mirror symmetry (yaw, roll), and the pitch about the
symmetry frame's Y axis that puts the emitter axis at 0 deg elevation.
Emitter axis: oral species, the mouth normal (piston.mouth_axis); nasal species, the power-weighted mean direction of the
band-averaged BEM emission pattern over directions within 6 dB of its maximum (N_FREQ frequencies across the band).
Usage: python scripts/acoustic_frame.py key   -> prints the TOML block for [species.<key>.frames.acoustic]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import gpubem, grid, mesh, paths, piston, project, rotate, sources, symmetry

N_FREQ = 6


def emitter_axis(key, m, sp):
    lm = sp["landmarks"]
    if sp["emitter"] == "oral":
        return piston.mouth_axis(m, np.array(lm["mouth"]))
    band = np.array(sp["band_khz"][:2]) * 1e3
    freqs = np.linspace(band[0], band[1], N_FREQ + 2)[1:-1]
    patches = {"emitter": [sources.disc_patch(m, np.array(p), project.NOSTRIL_RADIUS) for p in lm["nostrils"]]}
    pts, back = grid.unique_points("sphere")
    H = gpubem.to_grid(gpubem.solve(m, patches, freqs, pts, log=lambda s: None)["emitter"], "sphere")
    P = (np.abs(H) ** 2 / (np.abs(H) ** 2).max(axis=(0, 1), keepdims=True)).mean(-1)
    u = grid.unit_vectors(*grid.GRIDS["sphere"])
    w = np.where(P > P.max() / 4, P, 0) * np.cos(np.deg2rad(grid.EL))[:, None]
    a = (u * w[..., None]).sum((0, 1))
    return a / np.linalg.norm(a)


def main():
    key = sys.argv[1]
    sp = paths.species()[key]
    m = mesh.load(sp["stl"])
    (yaw, roll, y0), _ = symmetry.fit(m)
    S = rotate.matrix(yaw, 0, roll)
    a = S.T @ emitter_axis(key, m, sp)                  # emitter axis in the symmetry frame
    az, el = rotate.to_az_el(a)
    if sp["emitter"] == "oral":                          # the piston aims along the midline: drop the lateral part
        el = np.degrees(np.arctan2(a[2], a[0]))
    print(f"# {key}: emitter axis in the symmetry frame az {az:.1f} el {el:.1f} deg; symmetry plane y0 {y0:.2f} mm")
    print(f"[species.{key}.frames.acoustic]   # scripts/acoustic_frame.py: symmetry yaw/roll, pitch puts the emitter axis at el 0")
    print(f"symmetry = [{yaw:.2f}, {roll:.2f}]\npitch = {-el:.2f}")


if __name__ == "__main__":
    main()
