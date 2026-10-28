"""Full (two-way) ERTF comparison from a placement study: best-matching ears and nostril pair on the CT mesh, in the
fitted reference frame, at the study's frequencies, against the reference model. Same layout and metric as
plot_ertf_compare.py (|emitter| x |ear| per frequency, dB re max; broadband weighted by the call spectrum if present).
Usage: python scripts/plot_ertf_placement.py key  -> results/validation/ertf_full_<key>_ct.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import grid, paths, plotting, reference, rotate
from plot_ertf_compare import REF_LABEL


def main():
    key = sys.argv[1]
    d = np.load(paths.work_dir() / "placement" / f"{key}.npz")
    names, H, area, f = [str(n) for n in d["names"]], d["H"], d["area"], d["freqs"]
    res = json.loads((paths.RESULTS / "placement" / f"{key}.json").read_text())
    r = res["refit"]
    R = np.array(res.get("start_matrix", np.eye(3))) @ rotate.matrix(*r["angles"])
    rs = lambda h: np.abs(rotate.hemisphere_from_sphere(h.real, R) + 1j * rotate.hemisphere_from_sphere(h.imag, R))
    a, b, _ = r["emitter_best_pair"]
    ia, ib = names.index(a), names.index(b)
    sim = {"left": rs(H[names.index(r["best_left"])]), "right": rs(H[names.index(r["best_right"])]),
           "emitter": rs(H[ia] * area[ia] + H[ib] * area[ib])}
    ref = reference.load(key)
    ki = [int(np.argmin(abs(ref["freqs"] - x))) for x in f]
    old = {s: ref[s][..., ki] for s in sim}
    raw = sio.loadmat(paths.REF_DIR / paths.species()[key]["reference"], squeeze_me=True, struct_as_record=False)["bat_hrtf"]
    call = getattr(raw, "call_spec", None)
    w = (call[ki] / call.max()) ** 2 if call is not None else np.ones(len(f))
    sel = np.linspace(0, len(f) - 1, 5).round().astype(int)
    cols = [f"{f[k]/1e3:.1f} kHz" for k in sel] + ["Broadband (Call-Weighted)" if call is not None else "Broadband (Flat)"]
    rows = []
    for name, dd in ((REF_LABEL[key], old), ("This Work (CT Mesh, Calibrated Placement)", sim)):
        for ear in ("left", "right"):
            e = dd["emitter"] * dd[ear]
            rows.append((f"{name.replace(' (CT Mesh', chr(10) + '(CT Mesh')}\n{ear.capitalize()} Ear", [e[..., k] for k in sel] + [np.sqrt((e ** 2 * w).sum(-1))]))
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(2.9 * len(cols), 2.9 * len(rows)), squeeze=False)
    for i, (lab, maps) in enumerate(rows):
        for c, X in enumerate(maps):
            ax = axs[i, c]
            dB = 20 * np.log10(X / X.max())
            im = plotting.laea_map(ax, grid.AZ, grid.EL, dB)
            if i == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8)
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
                for sp_ in ax.spines.values():
                    sp_.set_visible(False)
    txt = []
    for j, ear in enumerate(("left", "right")):
        A, B = rows[j][1], rows[2 + j][1]
        cc = [reference.metrics(B[k][..., None], A[k][..., None])["corr"][0] for k in range(len(cols))]
        txt.append(f"{ear} ear full-ERTF pattern correlation: " + ", ".join(f"{c}: {v:.2f}" for c, v in zip(cols, cc)))
    one = {s: reference.metrics(sim[s], old[s])["corr"] for s in sim}
    txt.append("one-way, mean over all frequencies: " + ", ".join(f"{s} {v.mean():.2f}" for s, v in one.items()))
    fig.colorbar(im, ax=axs, shrink=0.5, label="Two-Way Sensitivity (dB re Max)")
    fig.text(0.5, 0.06, "LAEA projection of the frontal hemisphere, centred straight ahead; bat's left on the left; "
             "graticule every 30 deg in azimuth and elevation", ha="center", fontsize=9, color="0.3")
    fig.suptitle(f"{paths.species()[key]['name']}: Full ERTF, Reference vs This Work (CT mesh), LAEA projection\n" + "\n".join(txt), fontsize=10)
    out = paths.RESULTS / "validation" / f"ertf_full_{key}_ct.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("\n".join(txt)); print(out)


if __name__ == "__main__":
    main()
