"""Full (two-way) ERTF, reference model (De Mey et al. 2008 for pdisc, Vanderelst et al. 2010 for mmicr) vs this work,
in the reference frame.
ERTF(psi, f) = |H_emitter(psi, f)| * |H_ear(psi, f)| (spatial sensitivity, Vanderelst et al. 2010), shown per frequency
and broadband: sum over f of ERTF^2 weighted by the call spectrum^2 stored in the reference file. Each map in dB re
its own maximum. Without a call spectrum in the reference file the broadband map weights all frequencies equally.
Usage: python scripts/plot_ertf_compare.py [key]  -> results/validation/ertf_full_<key>.png"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import grid, paths, reference
from validate_reference import hemisphere

REF_LABEL = {"pdisc": "Reference (De Mey et al. 2008)", "mmicr": "Reference (Vanderelst et al. 2010)"}


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "pdisc"
    ref = reference.load(key)
    raw = sio.loadmat(paths.REF_DIR / paths.species()[key]["reference"], squeeze_me=True, struct_as_record=False)["bat_hrtf"]
    call = getattr(raw, "call_spec", None)
    with h5py.File(paths.work_dir() / "ertf" / f"{key}_reference.h5") as h:
        f = h["freqs"][:]
        sim = {s: np.abs(hemisphere(h[s]["H"][:])) for s in ("left", "right", "emitter")}
    ki = np.array([int(np.argmin(abs(ref["freqs"] - x))) for x in f])
    old = {s: ref[s][..., ki] for s in sim}
    w = (call[ki] / call.max()) ** 2 if call is not None else np.ones(len(f))
    FREQS = np.round(np.linspace(f[0], f[-1], 7)[1:-1] / 5e3) * 5e3            # five frequencies inside the band
    rows = []
    for name, d in ((REF_LABEL[key], old), ("This Work (GPU BEM)", sim)):
        for ear in ("left", "right"):
            e = d["emitter"] * d[ear]
            per_f = [e[..., int(np.argmin(abs(f - x)))] for x in FREQS]
            bb = np.sqrt((e ** 2 * w).sum(-1))
            rows.append((f"{name}\n{ear.capitalize()} Ear", per_f + [bb]))
    cols = [f"{x/1e3:.0f} kHz" for x in FREQS] + ["Broadband (Call-Weighted)" if call is not None else "Broadband (Flat)"]
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(2.9 * len(cols), 2.9 * len(rows)), squeeze=False)
    ext = [grid.AZ[-1], grid.AZ[0], grid.EL[0], grid.EL[-1]]
    for r, (lab, maps) in enumerate(rows):
        for c, X in enumerate(maps):
            ax = axs[r, c]
            d = 20 * np.log10(X / X.max())
            im = ax.imshow(d[:, ::-1], origin="lower", extent=ext, vmin=-30, vmax=0, cmap="inferno")
            ax.contour(grid.AZ[::-1], grid.EL, d[:, ::-1], levels=[-12, -6, -3], colors="w", linewidths=0.6)
            ax.set_xticks([90, 45, 0, -45, -90]); ax.set_yticks([-90, -45, 0, 45, 90]); ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(lab + "\nElevation (deg)", fontsize=8)
            if r == len(rows) - 1:
                ax.set_xlabel("Azimuth (deg)", fontsize=8)
    # agreement of the full ERTF per column
    txt = []
    for ear_i, ear in enumerate(("left", "right")):
        a, b = rows[ear_i][1], rows[2 + ear_i][1]
        cc = [reference.metrics(b[k][..., None], a[k][..., None])["corr"][0] for k in range(len(cols))]
        txt.append(f"{ear} ear pattern correlation: " + ", ".join(f"{c}: {v:.2f}" for c, v in zip(cols, cc)))
    fig.colorbar(im, ax=axs, shrink=0.5, label="Two-Way Sensitivity (dB re Max)")
    fig.suptitle(f"{paths.species()[key]['name']}: Full ERTF (Emitter x Ear), Reference vs This Work\n" + "\n".join(txt),
                 fontsize=10)
    out = paths.RESULTS / "validation" / f"ertf_full_{key}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("\n".join(txt))
    print(out)


if __name__ == "__main__":
    main()
