"""Calibrate the mouth piston radius of the oral emitters against published horizontal half-amplitude angles
(docs/mouth_emitter_calibration.md).

A disc patch ("piston") of radius r at the mouth landmark, on the rigid head, for r in RADII (each one right-hand side
of one GPU solve). At each target frequency the beam axis a is the direction of maximum |H|; the horizontal arc is
u(t) = cos(t) a + sin(t) b, with b the symmetry-plane normal made orthogonal to a. The half-amplitude angle is the mean
of the left and right angles where the arc drops 6 dB below the axis. The calibrated radius minimizes the squared
error in degrees over the target frequencies.
Usage: python scripts/calibrate_piston.py key [target_set]   -> results/calibration/piston_<key>_<set>.{json,png}"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import gpubem, grid, mesh, paths, rotate, sources, symmetry

RADII = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) * 1e-3
TARGETS = {   # (frequency Hz, half-amplitude angle deg); docs/mouth_emitter_calibration.md
    ("mdaub", "field"): [(45e3, 25), (50e3, 22), (55e3, 20), (60e3, 18), (65e3, 17), (70e3, 16), (75e3, 14)],
    ("mdaub", "lab"): [(55e3, 40)],
    ("efusc", "field"): [(35e3, 30)]}


def half_amplitude_angle(Hf, a, b):
    t = np.deg2rad(np.arange(-90, 90.01, 0.25))
    u = np.cos(t)[:, None] * a + np.sin(t)[:, None] * b
    d = 20 * np.log10(np.abs(rotate.SphereSampler(Hf)(u)) / np.abs(rotate.SphereSampler(Hf)(a[None]))[0])
    i0 = len(t) // 2
    out = []
    for side in (1, -1):
        j = i0
        while 0 < j < len(t) - 1 and d[j] > -6:
            j += side
        k = j - side                                # last point above -6 dB
        out.append(np.degrees(abs(t[k] + (t[j] - t[k]) * (d[k] + 6) / (d[k] - d[j]))))
    return np.mean(out), out


def main():
    key = sys.argv[1]
    tset = sys.argv[2] if len(sys.argv) > 2 else "field"
    sp = paths.species()[key]
    m = mesh.load(sp["stl"])
    mouth = np.array(sp["landmarks"]["mouth"])
    (yaw, roll, _), _ = symmetry.fit(m)
    nrm = rotate.matrix(yaw, 0, roll) @ np.array([0.0, 1.0, 0.0])
    patches = {f"r{r*1e3:.2f}": [sources.disc_patch(m, mouth, r)] for r in RADII}
    freqs = np.array([f for f, _ in TARGETS[(key, tset)]])
    target = np.array([h for _, h in TARGETS[(key, tset)]], float)
    pts, back = grid.unique_points("sphere")
    H = gpubem.solve(m, patches, freqs, pts, log=lambda s: print(s, flush=True))
    u = grid.unit_vectors(*grid.GRIDS["sphere"])
    haa = np.zeros((len(RADII), len(freqs)))
    axes = np.zeros((len(RADII), len(freqs), 3))
    for i, n in enumerate(patches):
        Hg = gpubem.to_grid(H[n], "sphere")
        for k in range(len(freqs)):
            a = u.reshape(-1, 3)[np.argmax(np.abs(Hg[..., k]))]
            b = nrm - (nrm @ a) * a
            b /= np.linalg.norm(b)
            haa[i, k], _ = half_amplitude_angle(Hg[..., k], a, b)
            axes[i, k] = a
    rr = np.linspace(RADII[0], RADII[-1], 701)
    model = np.array([np.interp(rr, RADII, haa[:, k]) for k in range(len(freqs))])      # [freq, radius]
    err = ((model - target[:, None]) ** 2).mean(0)
    r_best = rr[np.argmin(err)]
    i_near = np.argmin(abs(RADII - r_best))
    res = {"key": key, "target_set": tset, "freqs_hz": freqs.tolist(), "target_deg": target.tolist(),
           "radii_m": RADII.tolist(), "haa_deg": haa.tolist(), "radius_m": float(r_best),
           "rms_error_deg": float(np.sqrt(err.min())),
           "haa_at_radius_deg": [float(np.interp(r_best, RADII, haa[:, k])) for k in range(len(freqs))],
           "beam_axis_az_el_deg": np.round(np.array(rotate.to_az_el(axes[i_near])).T, 1).tolist(),
           "mouth_m": mouth.tolist(), "symmetry_yaw_roll_deg": [float(yaw), float(roll)]}
    out = paths.RESULTS / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"piston_{key}_{tset}.json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in range(len(freqs)):
        ax[0].plot(RADII * 1e3, haa[:, k], "o-", label=f"{freqs[k]/1e3:.0f} kHz")
        ax[0].axhline(target[k], color=ax[0].lines[-1].get_color(), ls=":", lw=0.8)
    ax[0].axvline(r_best * 1e3, color="k", ls="--")
    ax[0].set_xlabel("Piston Radius (mm)"); ax[0].set_ylabel("Half-Amplitude Angle (deg)"); ax[0].legend(fontsize=7)
    ax[1].plot(freqs / 1e3, target, "ko", label="Published")
    ax[1].plot(freqs / 1e3, res["haa_at_radius_deg"], "r^-", label=f"BEM, r = {r_best*1e3:.2f} mm")
    ax[1].set_xlabel("Frequency (kHz)"); ax[1].set_ylabel("Half-Amplitude Angle (deg)"); ax[1].legend()
    fig.suptitle(f"{sp['name']}: Mouth Piston Calibration ({tset})")
    fig.tight_layout()
    fig.savefig(out / f"piston_{key}_{tset}.png", dpi=120)
    print(json.dumps({k: res[k] for k in ("radius_m", "rms_error_deg", "haa_at_radius_deg", "beam_axis_az_el_deg")}))


if __name__ == "__main__":
    main()
