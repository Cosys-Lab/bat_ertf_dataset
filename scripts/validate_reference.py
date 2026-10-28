"""Technical validation against the reference ERTFs over the full reference band: production run in the reference frame
(<work>/ertf/<key>_reference.h5) vs the reference file, on the frontal hemisphere.
Metrics per frequency and source (reference.metrics): pattern correlation, rms dB difference, peak-direction error;
plus maps at a few frequencies.
Usage: python scripts/validate_reference.py key  -> results/validation/reference_<key>.{png,txt}, reference_<key>_maps.png"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import grid, paths, reference

SOURCES = {"left": "Left Ear", "right": "Right Ear", "emitter": "Emitter"}


def hemisphere(Hs):
    """Frontal hemisphere [el, az(-90..90)] of a full-sphere array."""
    az = grid.GRIDS["sphere"][0]
    j = [int(np.argmin(abs(az - a))) for a in grid.AZ]
    return Hs[:, j]


def main():
    key = sys.argv[1]
    ref = reference.load(key)
    with h5py.File(paths.work_dir() / "ertf" / f"{key}_reference.h5") as h:
        f = h["freqs"][:]
        done = h["done"][:]
        sim = {s: hemisphere(h[s]["H"][:])[..., done] for s in SOURCES}
    f = f[done]
    ki = np.array([int(np.argmin(abs(ref["freqs"] - x))) for x in f])
    ok = np.isclose(ref["freqs"][ki], f)
    f, ki = f[ok], ki[ok]
    mt = {s: reference.metrics(abs(sim[s][..., ok]), ref[s][..., ki]) for s in SOURCES}
    lines = [f"{paths.species()[key]['name']}: {len(f)} frequencies {f[0]/1e3:.1f}-{f[-1]/1e3:.1f} kHz, frontal hemisphere, "
             "patterns normalized per frequency and clipped at -30 dB",
             "source    corr median [p10, p90]    rms dB median [p10, p90]    peak error deg median [p10, p90]"]
    for s in SOURCES:
        q = lambda x: f"{np.median(x):6.3f} [{np.percentile(x, 10):6.3f}, {np.percentile(x, 90):6.3f}]"
        lines.append(f"{s:8s}  {q(mt[s]['corr'])}   {q(mt[s]['rms_db'])}   {q(mt[s]['peak_deg'])}")
    out = paths.RESULTS / "validation"
    (out / f"reference_{key}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (m, lab) in zip(axs, [("corr", "Pattern Correlation"), ("rms_db", "RMS Difference (dB)"),
                                  ("peak_deg", "Peak Direction Error (deg)")]):
        for s, name in SOURCES.items():
            ax.plot(f / 1e3, mt[s][m], label=name)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axs[0].legend()
    fig.suptitle(f"{paths.species()[key]['name']}: Simulated vs Reference ERTFs")
    fig.tight_layout()
    fig.savefig(out / f"reference_{key}.png", dpi=120)

    sel = np.linspace(0, len(f) - 1, 5).round().astype(int)
    fig, axs = plt.subplots(6, len(sel), figsize=(3 * len(sel), 15), squeeze=False)
    ext = [grid.AZ[-1], grid.AZ[0], grid.EL[0], grid.EL[-1]]
    for r, s in enumerate(SOURCES):
        for c, k in enumerate(sel):
            for rr, (X, lab) in enumerate([(abs(sim[s][..., ok][..., k]), "BEM"), (ref[s][..., ki[k]], "Reference")]):
                ax = axs[2 * r + rr, c]
                d = 20 * np.log10(X / X.max())
                ax.imshow(d[:, ::-1], origin="lower", extent=ext, vmin=-30, vmax=0, cmap="inferno")
                ax.contour(grid.AZ[::-1], grid.EL, d[:, ::-1], levels=[-12, -6, -3], colors="w", linewidths=0.5)
                ax.set_title(f"{SOURCES[s]} {lab} {f[k]/1e3:.1f} kHz", fontsize=8)
                ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out / f"reference_{key}_maps.png", dpi=80)


if __name__ == "__main__":
    main()
