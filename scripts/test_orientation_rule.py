"""Does a reproducible orientation rule match the reference frames? For pdisc and mmicr compare the pattern
correlations at (a) the freely fitted yaw/pitch/roll (fit.json) and (b) yaw/roll from the mirror-symmetry plane with
only the pitch fitted. Also reports the emitter beam axis (energy-weighted, frontal hemisphere) in the reference.
Usage: python scripts/test_orientation_rule.py"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import grid, mesh, paths, reference, rotate, symmetry
from fit_calibration import load, corr


def scores(H, area, fit, ref, ang):
    R = rotate.matrix(*ang)
    rs = lambda h: rotate.hemisphere_from_sphere(h.real, R) + 1j * rotate.hemisphere_from_sphere(h.imag, R)
    a, b = fit["nostrils"]["pair"]
    e = rs((H[a] * area[a] + H[b] * area[b]))
    return [corr(abs(rs(H[fit["ears"]["left"]["cand"]])), ref["left"]), corr(abs(rs(H[fit["ears"]["right"]["cand"]])), ref["right"]),
            corr(abs(e), ref["emitter"])]


def beam_axis(mag):
    u = grid.unit_vectors()
    w = (mag ** 2).sum(-1) * np.cos(np.deg2rad(grid.EL))[:, None]
    d = (u * w[..., None]).sum((0, 1))
    return rotate.to_az_el(d / np.linalg.norm(d))


for key in ["pdisc", "mmicr"]:
    proj, meta, cand, f, H, ref = load(key)
    fit = json.loads((proj / "fit.json").read_text())
    (yaw, roll, _), _ = symmetry.fit(mesh.load(paths.species()[key]["stl"]))
    area = meta["patch_area"]
    free = scores(H, area, fit, ref, fit["angles"])
    best = max(((np.mean(scores(H, area, fit, ref, (yaw, p, roll))), p) for p in np.arange(-70, -20, 2.5)))
    rule = scores(H, area, fit, ref, (yaw, best[1], roll))
    print(f"{key}: free fit {np.round(fit['angles'], 1)} corr L/R/E {np.round(free, 3)}")
    print(f"{key}: symmetry yaw {yaw:.1f} roll {roll:.1f}, best pitch {best[1]:.1f}: corr L/R/E {np.round(rule, 3)}")
    print(f"{key}: reference emitter beam axis az/el {np.round(beam_axis(ref['emitter']), 1)} deg")
