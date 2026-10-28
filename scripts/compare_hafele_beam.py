"""Compare our analytical mouth piston of M. daubentonii (batertf/piston.py, radius from config/species.toml) with the
in-flight emission beam of M. daubentonii of Häfele et al. 2026 (PNAS, doi:10.1073/pnas.2605701123; Zenodo
doi:10.5281/zenodo.20927789, file BEAM_INFLIGHT_daubentons_1kHz_steps.mat, CC BY-NC-ND 4.0: compare and cite only,
the beam and maps derived from it are never put into our dataset). Independent of our piston calibration data
(Surlykke et al. 2009, field recordings). Conventions: docs/vesterholm_validation.md (section Häfele).

Beam: `daub` (181 el x 361 az x 122), el = -90:1:90, az = -180:1:180 (the duplicate +180 column is dropped, as in
their figure_04.m), page 0 broadband, pages 1..121 = 30..150 kHz; dB re max, floored at -40 dB; frame: flight
direction (not our CT frame). The piston is axisymmetric, so it is aligned with the beam's peak direction per
frequency; azimuth sign / frame handedness then do not matter.
Per frequency: -6 dB half-angle (solid-angle-equivalent, as scripts/sensitivity_checks.half_angle), directivity
index, pattern correlation and rms dB (each normalized to its max, -30 dB floor, solid-angle weighted), and the
best-fit piston radius (a) matching the half-angle, (b) maximizing the pattern correlation.
Usage: python -I scripts/compare_hafele_beam.py [beam.mat]
  default beam file: <work>/../external/hafele2026/download/BEAM_INFLIGHT_daubentons_1kHz_steps.mat
  -> results/validation/hafele_beam.json, paper/figures/hafele_beam.pdf/.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths, piston, plotting, reference

EL = np.arange(-90, 90.5, 1.0)
AZ = np.arange(-180, 180, 1.0)
FREQS = np.arange(30e3, 150.5e3, 1e3)
FLOOR = -30.0
RADII = np.linspace(1e-3, 15e-3, 281)
MAP_KHZ = (40, 80, 120)


def unit(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)


U = unit(*np.meshgrid(AZ, EL))                                       # [el, az, 3]
W = np.cos(np.radians(EL))[:, None] * np.ones(len(AZ))
W /= W.sum()


def half_angle(db):
    """Angular radius (deg) of the cap with the solid angle of the -6 dB region."""
    omega = ((db > -6) * W).sum() * 4 * np.pi
    return float(np.degrees(np.arccos(1 - omega / (2 * np.pi))))


def di(db):
    p = 10 ** (db / 10)
    return float(10 * np.log10(p.max() / (W * p).sum()))


def piston_db(axis, f, a):
    cos_t = np.clip(U @ axis, -1, 1)
    x = 2 * np.pi * f / piston.C * a * np.sqrt(1 - cos_t ** 2)
    m = np.abs(piston.directivity(x, cos_t))
    return 20 * np.log10(np.maximum(m / m.max(), 1e-12))


def piston_half_angle(f, a):
    """Exact -6 dB half-angle of the piston (largest angle within -6 dB), for the radius fit."""
    t = np.radians(np.linspace(0, 180, 3601))
    m = np.abs(piston.directivity(2 * np.pi * f / piston.C * a * np.sin(t), np.cos(t)))
    return float(np.degrees(t[np.argmax(m < 0.5)])) if (m < 0.5).any() else 180.0


def corr_rms(db_a, db_b):
    r = reference.metrics_points(10 ** (db_a.reshape(-1, 1) / 20), 10 ** (db_b.reshape(-1, 1) / 20), W.ravel(),
                                 U.reshape(-1, 3), FLOOR)
    return float(r["corr"][0]), float(r["rms_db"][0])


def main():
    fn = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        paths.work_dir().parent / "external" / "hafele2026" / "download" / "BEAM_INFLIGHT_daubentons_1kHz_steps.mat"
    B = sio.loadmat(fn)["daub"][:, :-1, :]                              # drop the duplicate +180 column
    assert B.shape == (181, 360, 122)
    sp = paths.species()["mdaub"]
    a0 = sp["piston_radius_m"]
    out = {"source": {"paper": "Häfele et al. 2026, PNAS, doi:10.1073/pnas.2605701123",
                      "data": "doi:10.5281/zenodo.20927789, BEAM_INFLIGHT_daubentons_1kHz_steps.mat",
                      "licence": "CC BY-NC-ND 4.0 (comparison only; data not redistributed)"},
           "piston_radius_m": a0, "piston_calibration": sp.get("piston_calibration", ""), "floor_db": FLOOR,
           "f_khz": (FREQS / 1e3).tolist(), "beam": {k: [] for k in ("peak_az_deg", "peak_el_deg", "half_angle_deg",
                                                                     "di_db", "min_db")},
           "piston": {k: [] for k in ("half_angle_deg", "di_db", "corr", "rms_db")},
           "fit": {k: [] for k in ("radius_half_angle_mm", "radius_corr_mm", "corr_at_best")}}
    maps = {}
    for k, f in enumerate(FREQS):
        b = B[..., k + 1] - B[..., k + 1].max()
        i, j = np.unravel_index(np.argmax(b), b.shape)
        ax = U[i, j]
        p = piston_db(ax, f, a0)
        c, r = corr_rms(p, b)
        ha = half_angle(b)
        out["beam"]["peak_az_deg"].append(float(AZ[j]))
        out["beam"]["peak_el_deg"].append(float(EL[i]))
        out["beam"]["half_angle_deg"].append(ha)
        out["beam"]["di_db"].append(di(b))
        out["beam"]["min_db"].append(float(b.min()))
        out["piston"]["half_angle_deg"].append(half_angle(p))
        out["piston"]["di_db"].append(di(p))
        out["piston"]["corr"].append(c)
        out["piston"]["rms_db"].append(r)
        # radius whose solid-angle-equivalent half-angle equals the beam's (monotonic in a)
        g = lambda a: half_angle(piston_db(ax, f, a)) - ha
        lo, hi = RADII[0], RADII[-1]
        out["fit"]["radius_half_angle_mm"].append(float(brentq(g, lo, hi, xtol=1e-5) * 1e3)
                                                  if g(lo) > 0 > g(hi) else float("nan"))
        cs = [corr_rms(piston_db(ax, f, a), b)[0] for a in RADII[::4]]
        best = RADII[::4][int(np.argmax(cs))]
        out["fit"]["radius_corr_mm"].append(float(best * 1e3))
        out["fit"]["corr_at_best"].append(float(max(cs)))
        if round(f / 1e3) in MAP_KHZ:
            maps[round(f / 1e3)] = (b, p, ax)
    f_khz = FREQS / 1e3
    sel = lambda lo, hi: (f_khz >= lo) & (f_khz <= hi)
    s = {}
    for name, (lo, hi) in {"30-150": (30, 150), "45-75 (calibration band)": (45, 75), "30-100": (30, 100)}.items():
        m = sel(lo, hi)
        s[name] = {"half_angle_diff_deg_mean": float(np.mean(np.array(out["piston"]["half_angle_deg"])[m]
                                                           - np.array(out["beam"]["half_angle_deg"])[m])),
                   "di_diff_db_mean": float(np.mean(np.array(out["piston"]["di_db"])[m]
                                                     - np.array(out["beam"]["di_db"])[m])),
                   "corr_median": float(np.median(np.array(out["piston"]["corr"])[m])),
                   "rms_db_median": float(np.median(np.array(out["piston"]["rms_db"])[m])),
                   "radius_half_angle_mm_median": float(np.nanmedian(np.array(out["fit"]["radius_half_angle_mm"])[m])),
                   "radius_corr_mm_median": float(np.median(np.array(out["fit"]["radius_corr_mm"])[m]))}
    out["summary"] = s
    (paths.RESULTS / "validation" / "hafele_beam.json").write_text(json.dumps(out, indent=1))
    figure(out, maps, a0)
    for k, v in s.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})


def figure(out, maps, a0):
    """Curves only: the beam is CC BY-NC-ND, so no maps derived from it are shown (maps stay unused).
    a: -6 dB half-angle against frequency; b: piston radius fitted to the in-flight half-angle.
    Redraw from the saved results (results/validation/hafele_beam.json) without the beam file: load the JSON as `o`
    and call figure(o, None, o["piston_radius_m"])."""
    sys.path.insert(0, str(paths.ROOT / "paper" / "figures"))
    import style
    style.apply()
    f = np.array(out["f_khz"])
    fig, axs = plt.subplots(1, 2, figsize=(0.85 * style.WIDTH, 1.9), gridspec_kw=dict(wspace=0.3))
    lab_b, lab_p = "In flight (Häfele et al. 2026)", f"Piston, $a$ = {a0*1e3:.2f} mm"
    axs[0].plot(f, out["beam"]["half_angle_deg"], color="tab:red", lw=1.0, label=lab_b)
    axs[0].plot(f, out["piston"]["half_angle_deg"], color="tab:blue", lw=1.0, label=lab_p)
    axs[0].set(xlabel="Frequency (kHz)", ylabel="\u22126 dB half-angle (deg)", ylim=(0, None))
    axs[1].plot(f, np.array(out["fit"]["radius_half_angle_mm"]), color="k", lw=1.0, label="Fitted to the half-angle")
    axs[1].axhline(a0 * 1e3, color="tab:blue", ls="--", lw=0.8, label="Calibrated (field data)")
    axs[1].set(xlabel="Frequency (kHz)", ylabel="Piston radius (mm)", ylim=(0, None))
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    for ax, t in zip(axs, "de"):                      # panels d, e of the measurement figure (a-c: Vesterholm)
        ax.grid(alpha=0.3)
        ax.axvspan(100, f.max(), color="0.9", zorder=-1)
        ax.set_xlim(f.min(), f.max())
        style.letter(ax, t, dx=-26)
    style.save(fig, paths.ROOT / "paper" / "figures" / "hafele_beam", png_dpi=250)


if __name__ == "__main__":
    main()
