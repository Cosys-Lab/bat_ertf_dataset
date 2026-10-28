"""Technical Validation: sensitivity to the emitter and receiver assumptions, and consistency checks of the conventions.

1. Piston emitters (E. fuscus, M. daubentonii): radius scaled by RADIUS_SCALE and the axis tilted by +-TILT_DEG in
   elevation; directivity index, -6 dB half-angle and pattern correlation with the nominal piston.
2. Placement ensemble (all species, from the production files): every alternative ear / nostril pair against the
   nominal one, per frequency: correlation of the dB patterns (full sphere, solid-angle weighted, -30 dB floor),
   difference in directivity index, angle between the peak directions.
3. Conventions: (a) group delays fitted from the phase slope over 20-40 kHz against geometric delays u . p / c:
   for the emitter, its own delay over the main lobe (p: nostril midpoint or mouth); for the ears, the interaural
   difference (p: left minus right ear landmark; directions where both ears are within 20 dB of their maximum), since the pinna adds a direction-dependent delay of its own. This
   checks in one go the time convention exp(+j w t), the referencing to a monopole at the origin, the azimuth/elevation
   convention and the registration of mesh, landmarks and directions. (b) Left/right mirror symmetry: |H_left(az, el)|
   against |H_right(-az, el)|.
Usage: python scripts/sensitivity_checks.py   -> results/validation/sensitivity.json"""
import json
import os
import sys
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import grid, mesh, paths, piston, rotate

ORDER = ["pdisc", "mmicr", "mmacr", "rrouxi", "rform", "efusc", "mdaub"]
AZ, EL = grid.GRIDS["sphere"]
U = grid.unit_vectors(AZ, EL)
W = np.cos(np.deg2rad(EL))[:, None] * np.ones(len(AZ))[None, :]
W = W / W.sum()
RADIUS_SCALE = [0.8, 0.9, 1.0, 1.1, 1.2]
TILT_DEG = 10.0
F_PISTON = np.arange(20e3, 160.01e3, 2e3)
OUT = paths.ROOT / "results" / "validation" / "sensitivity.json"


def pattern_db(mag, floor=-30.0):
    return np.maximum(20 * np.log10(mag / mag.max((0, 1), keepdims=True)), floor)


def corr(a, b):
    """Solid-angle-weighted correlation of two dB patterns [el, az, f] (per frequency)."""
    w = W[..., None]
    am, bm = a - (w * a).sum((0, 1)), b - (w * b).sum((0, 1))
    return (w * am * bm).sum((0, 1)) / np.sqrt((w * am ** 2).sum((0, 1)) * (w * bm ** 2).sum((0, 1)))


def di(mag):
    P = mag ** 2
    return 10 * np.log10(P.max((0, 1)) / (P * W[..., None]).sum((0, 1)))


def half_angle(mag, axis):
    """-6 dB half-angle (deg) around the beam axis: largest angle from the axis still within -6 dB, median over the
    azimuth around the axis (approximated by the angular radius of the -6 dB region of equal solid angle)."""
    db = 20 * np.log10(mag / mag.max((0, 1), keepdims=True))
    omega = ((db > -6) * W[..., None]).sum((0, 1)) * 4 * np.pi
    return np.degrees(np.arccos(1 - omega / (2 * np.pi)))


def peak_dir(mag):
    flat = mag.reshape(-1, mag.shape[-1])
    return U.reshape(-1, 3)[np.argmax(flat, 0)]


def pistons(sp):
    out = {}
    for key in ("efusc", "mdaub"):
        s = sp[key]
        m = mesh.load(s["stl"])
        c = np.array(s["landmarks"].get("mouth", s.get("mouth")))
        ax = piston.mouth_axis(m, c)
        a0 = s["piston_radius_m"]
        ref = abs(piston.transfer(U, F_PISTON, c, ax, a0))
        r = {"f_khz": (F_PISTON / 1e3).tolist(), "radius_m": a0, "axis": ax.tolist(), "radius": {}, "tilt": {}}
        for sc in RADIUS_SCALE:
            H = abs(piston.transfer(U, F_PISTON, c, ax, a0 * sc))
            r["radius"][str(sc)] = {"di_db": di(H).tolist(), "half_angle_6db_deg": half_angle(H, ax).tolist(),
                                    "corr_vs_nominal": corr(pattern_db(H), pattern_db(ref)).tolist()}
        for t in (-TILT_DEG, TILT_DEG):
            R = rotate.matrix(0, t, 0)
            H = abs(piston.transfer(U, F_PISTON, c, R @ ax, a0))
            r["tilt"][str(t)] = {"corr_vs_nominal": corr(pattern_db(H), pattern_db(ref)).tolist()}
        out[key] = r
    return out


def landmark(lm, s):
    if s.startswith("left"):
        return np.array(lm["ear_left"])
    if s.startswith("right"):
        return np.array(lm["ear_right"])
    return np.mean(np.atleast_2d(lm["nostrils"] if "nostrils" in lm else lm["mouth"]), 0)


def per_species(key, sp):
    with h5py.File(paths.work_dir() / "ertf" / f"{key}_ct.h5", "r") as h:
        done = h["done"][:]
        f = h["freqs"][:]
        fd = f[done]
        names = [n for n in h if isinstance(h[n], h5py.Group) and "H" in h[n]]
        H = {n: h[n]["H"][..., done] for n in names}
        Hlow = {n: h[n]["H"][..., :] for n in ("left", "right", "emitter")}
        dense = done.copy()
    r = {"f_khz": (fd / 1e3).tolist(), "ensemble": {}, "delay": {}, "symmetry": {}}
    for base in ("left", "right", "emitter"):
        if base == "emitter" and sp[key]["emitter"] == "oral":
            continue
        nom = abs(H[base])
        for n in sorted(x for x in H if x.startswith(base + "_alt")):
            alt = abs(H[n])
            ang = np.degrees(np.arccos(np.clip((peak_dir(nom) * peak_dir(alt)).sum(1), -1, 1)))
            r["ensemble"][n] = {"corr": corr(pattern_db(alt), pattern_db(nom)).tolist(),
                                "d_di_db": (di(alt) - di(nom)).tolist(), "peak_angle_deg": ang.tolist()}
    # group delays over 20-40 kHz from the phase slope (all computed frequencies in that band)
    lm = sp[key]["landmarks"]
    sel = np.where(dense & (f <= 40e3))[0]
    w = 2 * np.pi * f[sel]
    tau = {}
    for s in ("left", "right", "emitter"):
        ph = np.unwrap(np.angle(Hlow[s][..., sel]), axis=-1)
        tau[s] = ((ph - ph.mean(-1, keepdims=True)) * (w - w.mean())).sum(-1) / ((w - w.mean()) ** 2).sum()
    # emitter: fitted delay against the geometric delay of its landmark, over the main lobe (within 10 dB of the max)
    lit = 20 * np.log10(abs(Hlow["emitter"][..., sel]).mean(-1))
    lobe = lit > lit.max() - 10
    geo = (U @ landmark(lm, "emitter")) / 343.0
    r["delay"]["emitter"] = {"corr_fitted_vs_geometric": float(np.corrcoef(tau["emitter"][lobe], geo[lobe])[0, 1]),
                             "rms_diff_us": float(np.sqrt(((tau["emitter"] - geo)[lobe] ** 2).mean()) * 1e6),
                             "geometric_range_us": float(np.ptp(geo[lobe]) * 1e6), "n_freqs": int(len(sel))}
    # ears: interaural group delay difference against the geometric one of the two ear landmarks, over the directions
    # where both ears are within 20 dB of their maximum (no shadowed ear, where the phase slope is not a delay),
    # solid-angle weighted; the sign checks the azimuth convention, the slope is > 1 because of diffraction
    itd = tau["left"] - tau["right"]
    geo = (U @ (landmark(lm, "left") - landmark(lm, "right"))) / 343.0
    lvl = {s: 20 * np.log10(abs(Hlow[s][..., sel]).mean(-1)) for s in ("left", "right")}
    lit2 = (lvl["left"] > lvl["left"].max() - 20) & (lvl["right"] > lvl["right"].max() - 20)
    w2 = W * lit2
    wm = lambda a: a - (w2 * a).sum() / w2.sum()
    c = (w2 * wm(itd) * wm(geo)).sum() / np.sqrt((w2 * wm(itd) ** 2).sum() * (w2 * wm(geo) ** 2).sum())
    r["delay"]["interaural"] = {"corr_fitted_vs_geometric": float(c),
                                "slope": float(np.polyfit(geo[lit2], itd[lit2], 1, w=W[lit2])[0]),
                                "solid_angle_fraction": float(w2.sum() / W.sum()),
                                "geometric_range_us": float(np.ptp(geo[lit2]) * 1e6),
                                "fitted_range_us": float(np.ptp(itd[lit2]) * 1e6), "n_freqs": int(len(sel))}
    mirror = np.concatenate([[0], np.arange(len(AZ) - 1, 0, -1)])        # index of -az on the -180..177.5 grid
    a = pattern_db(abs(H["left"]))
    b = pattern_db(abs(H["right"][:, mirror]))
    r["symmetry"] = {"corr_left_vs_mirrored_right": corr(a, b).tolist()}
    return r


def main():
    sp = paths.species()
    res = {"pistons": pistons(sp), "species": {}}
    for key in ORDER:
        res["species"][key] = per_species(key, sp)
        d = res["species"][key]
        print(key, {s: round(v["corr_fitted_vs_geometric"], 3) for s, v in d["delay"].items()},
              "sym", round(float(np.median(d["symmetry"]["corr_left_vs_mirrored_right"])), 3),
              "ens", {n: round(float(np.median(v["corr"])), 3) for n, v in d["ensemble"].items()}, flush=True)
    for key, r in res["pistons"].items():
        print(key, {k: (round(v["di_db"][-1] - r["radius"]["1.0"]["di_db"][-1], 2),
                        round(float(np.median(v["corr_vs_nominal"])), 3)) for k, v in r["radius"].items()})
    OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
