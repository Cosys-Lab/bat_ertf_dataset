"""Pose a head and compute its transfer functions with the BEM (needs the full code from GitHub and an NVIDIA GPU,
see examples/POSING.md). The pinnae and the noseleaf are posed with the rig of the species folder, the posed mesh is
checked, the landmarks move with the surface, and both ears and the emitter are solved on the full-sphere grid of the
dataset. The result is saved in the layout of ertf_ct.h5 and compared with the rest pose of the dataset.
Usage: python 05_pose_and_simulate.py [species_folder] [--flex 15] [--abduct 0] [--yaw 0] [--tip 0]
                                      [--noseleaf 0] [--freqs 40000,80000,120000] [--save posed.h5]"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import numpy as np

from _setup import args, bd

try:                                                   # the solver and the rig are part of the full repository
    from batertf import gpubem, grid, piston, project, rig, sources
except ImportError:
    sys.exit("This example needs the full code (GitHub): batertf/gpubem.py, rig.py, ... See examples/POSING.md.")

a = args(__doc__, flex=15.0, abduct=0.0, yaw=0.0, tip=0.0, noseleaf=0.0, freqs="40000,80000,120000",
         save="posed.h5")
folder = Path(a.species)
freqs = np.array([float(x) for x in a.freqs.split(",")])

# 1. The pose: both pinnae (mirror-symmetric) about their base joints, the tips, and the noseleaf
r = rig.load_rig_folder(folder)
pose = {"pinna_both.base": {"flex": a.flex, "abduct": a.abduct, "yaw": a.yaw}}
if a.tip:
    pose["pinna_both.distal"] = {"flex": a.tip}
if a.noseleaf and "noseleaf.base" in r.bones:
    pose["noseleaf.base"] = {"flex": a.noseleaf}

# 2. Posed mesh, validity check, landmarks on the posed surface
m = r.deform(pose)
check = r.check(m, pose)
print("pose", pose, "valid:", check["valid"])
if not check["valid"]:
    sys.exit(f"invalid pose: {check} (see the valid joint ranges in POSING.md)")
lm = r.deform_landmarks(pose, m)

# 3. Sources as in the dataset: disc patches at the ears and nostrils; the oral emitters are analytical pistons
patches = {"left": [sources.disc_patch(m, lm["ear_left"], project.EAR_RADIUS)],
           "right": [sources.disc_patch(m, lm["ear_right"], project.EAR_RADIUS)]}
if "nostril_0" in lm:
    patches["emitter"] = [sources.disc_patch(m, lm[k], project.NOSTRIL_RADIUS) for k in ("nostril_0", "nostril_1")]

# 4. BEM on the full-sphere grid of the dataset (one LU decomposition per frequency)
pts, _ = grid.unique_points("sphere")
H = gpubem.solve_lu(m, patches, freqs, pts)
H = {s: gpubem.to_grid(v, "sphere") for s, v in H.items()}
if "emitter" not in H:                                 # oral species: the piston of the dataset (pinna poses do
    with h5py.File(folder / "ertf_ct.h5", "r") as h:    # not move the mouth), same centre, axis and radius
        g = h["emitter"].attrs
        centre, axis, radius = np.asarray(g["centre_m"]), np.asarray(g["axis"]), float(g["radius_m"])
    H["emitter"] = piston.transfer(grid.unit_vectors(*grid.GRIDS["sphere"]), freqs, centre, axis, radius)

# 5. Save in the layout of ertf_ct.h5
az, el = grid.GRIDS["sphere"]
with h5py.File(a.save, "w") as h:
    h["freqs"], h["azimuth"], h["elevation"] = freqs, az, el
    for s, v in H.items():
        h.create_group(s)["H"] = v.astype(np.complex64)
    h["vertices"] = m.vertices.astype(np.float32)          # faces as mesh.stl
    h.attrs["pose"] = json.dumps(pose)
    h.attrs["check"] = json.dumps(check, default=str)
print("saved", a.save)

# 6. Compare with the rest pose of the dataset (pattern correlation of the dB maps, -30 dB floor)
d = bd.load(folder)
w = np.cos(np.deg2rad(el))[:, None] * np.ones(len(az))
for s in ("left", "right", "emitter"):
    for j, f in enumerate(freqs):
        k = int(np.argmin(np.abs(d.freqs - f)))
        x = [np.maximum(20 * np.log10(np.abs(y) / np.abs(y).max()), -30) for y in (H[s][..., j], d.H[s][..., k])]
        x = [y - (w * y).sum() / w.sum() for y in x]
        c = (w * x[0] * x[1]).sum() / np.sqrt((w * x[0] ** 2).sum() * (w * x[1] ** 2).sum())
        print(f"{s:8s} {f / 1e3:5.0f} kHz: pattern correlation with the rest pose {c:.3f}")
