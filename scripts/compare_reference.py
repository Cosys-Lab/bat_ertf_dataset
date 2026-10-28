"""Compare a simulated project with the reference ERTFs of the same species at the simulated frequencies.
Usage: python scripts/compare_reference.py key project_name [n_freqs_to_plot]
Output: <work>/compare/<project_name>.png and printed per-frequency metrics."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import grid, numcalc, paths, reference


def main():
    key, name = sys.argv[1], sys.argv[2]
    proj = paths.work_dir() / "runs" / name
    ref = reference.load(key)
    rows = []
    figs = {}
    for src in ["left", "right", "emitter"]:
        f, H = numcalc.read_source(proj, src)
        ok = ~np.isnan(H).any((0, 1))
        f, H = f[ok], H[..., ok]
        ki = [int(np.argmin(abs(ref["freqs"] - x))) for x in f]
        assert np.allclose(ref["freqs"][ki], f), "simulated frequencies not on the reference grid"
        R = ref[src][..., ki]
        mt = reference.metrics(abs(H), R)
        for k in range(len(f)):
            rows.append(f"{src:8s} {f[k]/1e3:6.1f} kHz  rms {mt['rms_db'][k]:5.2f} dB  corr {mt['corr'][k]:.3f}  "
                        f"peak {mt['peak_deg'][k]:5.1f} deg")
        figs[src] = (f, H, R)
    print("\n".join(rows))
    n = min(int(sys.argv[3]) if len(sys.argv) > 3 else 5, len(figs["left"][0]))
    fig, axs = plt.subplots(6, n, figsize=(3 * n, 15), squeeze=False)
    ext = [grid.AZ[-1], grid.AZ[0], grid.EL[0], grid.EL[-1]]       # azimuth +90 (left) on the left of the image
    for r, src in enumerate(["left", "right", "emitter"]):
        f, H, R = figs[src]
        sel = np.linspace(0, len(f) - 1, n).round().astype(int)
        for c, k in enumerate(sel):
            for rr, (X, lab) in enumerate([(abs(H[..., k]), "sim"), (R[..., k], "ref")]):
                ax = axs[2 * r + rr, c]
                d = 20 * np.log10(X / X.max())
                ax.imshow(d[:, ::-1], origin="lower", extent=ext, vmin=-30, vmax=0, cmap="inferno")
                ax.contour(grid.AZ[::-1], grid.EL, d[:, ::-1], levels=[-12, -6, -3], colors="w", linewidths=0.5)
                ax.set_title(f"{src} {lab} {f[k]/1e3:.0f} kHz", fontsize=8)
                ax.tick_params(labelsize=6)
    fig.tight_layout()
    out = paths.work_dir() / "compare"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=70)
    print(out / f"{name}.png")


if __name__ == "__main__":
    main()
