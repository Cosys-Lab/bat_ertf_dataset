"""Build the NumCalc project of one species: mesh, source patches from the landmarks, frequencies."""
import numpy as np

from . import mesh, numcalc, paths, sources

EAR_RADIUS = 0.4e-3       # disc patch at the ear canal entrance [m] (as in the placement calibration)
NOSTRIL_RADIUS = 0.3e-3   # disc patch per nostril [m] (as in the placement calibration)


def patches_from_landmarks(m, lm, mouth_radius=None):
    p = {"left": [sources.disc_patch(m, np.array(lm["ear_left"]), EAR_RADIUS)],
         "right": [sources.disc_patch(m, np.array(lm["ear_right"]), EAR_RADIUS)]}
    if "nostrils" in lm:
        p["emitter"] = [sources.disc_patch(m, np.array(n), NOSTRIL_RADIUS) for n in lm["nostrils"]]
    elif "mouth" in lm:
        p["emitter"] = [sources.disc_patch(m, np.array(lm["mouth"]), mouth_radius)]
    return p


def build(key, freqs, landmarks=None, name=None, mouth_radius=None, grid_name="hemisphere", extra=None):
    sp = paths.species()[key]
    m = mesh.load(sp["stl"])
    lm = landmarks or sp["landmarks"]
    proj = paths.work_dir() / "runs" / (name or key)
    patches = patches_from_landmarks(m, lm, mouth_radius)
    patches.update(extra or {})
    meta = numcalc.write_project(proj, m, patches, freqs, grid_name=grid_name)
    return proj, meta
