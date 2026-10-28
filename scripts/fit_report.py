"""Per-frequency metrics and maps of the calibrated pilot (fit.json) against the reference.
Usage: python scripts/fit_report.py key  -> <work>/compare/fit_<key>.png and printed table"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import grid, paths, reference, rotate
from fit_calibration import load


def calibrated(key):
    """Calibrated simulation of the pilot on the hemisphere grid of the reference frame: {source: complex [el, az, f]}."""
    proj, meta, cand, f, H, ref = load(key)
    fit = json.loads((proj / "fit.json").read_text())
    R = rotate.matrix(*fit["angles"])
    rs = lambda h: rotate.hemisphere_from_sphere(h.real, R) + 1j * rotate.hemisphere_from_sphere(h.imag, R)
    a, b = fit["nostrils"]["pair"]
    area = meta["patch_area"]
    sim = {"left": rs(H[fit["ears"]["left"]["cand"]]), "right": rs(H[fit["ears"]["right"]["cand"]]),
           "emitter": rs((H[a] * area[a] + H[b] * area[b]) / (area[a] + area[b]))}
    return f, sim, ref


def main():
    key = sys.argv[1]
    f, sim, ref = calibrated(key)
    for s in sim:
        mt = reference.metrics(abs(sim[s]), ref[s])
        for k in range(len(f)):
            print(f"{s:8s} {f[k]/1e3:6.1f} kHz  rms {mt['rms_db'][k]:5.2f} dB  corr {mt['corr'][k]:.3f}  peak {mt['peak_deg'][k]:5.1f} deg")
    n = len(f)
    fig, axs = plt.subplots(6, n, figsize=(3 * n, 15), squeeze=False)
    ext = [grid.AZ[-1], grid.AZ[0], grid.EL[0], grid.EL[-1]]
    for r, s in enumerate(sim):
        for k in range(n):
            for rr, (X, lab) in enumerate([(abs(sim[s][..., k]), "sim"), (ref[s][..., k], "ref")]):
                ax = axs[2 * r + rr, k]
                d = 20 * np.log10(X / X.max())
                ax.imshow(d[:, ::-1], origin="lower", extent=ext, vmin=-30, vmax=0, cmap="inferno")
                ax.contour(grid.AZ[::-1], grid.EL, d[:, ::-1], levels=[-12, -6, -3], colors="w", linewidths=0.5)
                ax.set_title(f"{s} {lab} {f[k]/1e3:.0f} kHz", fontsize=8)
                ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(paths.work_dir() / "compare" / f"fit_{key}.png", dpi=70)


if __name__ == "__main__":
    main()
