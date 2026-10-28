"""Compact full-ERTF comparison of the two reference species side by side, in Lambert azimuthal equal-area projection.
For each species (placement study: best-matching ears and nostril pair on the CT mesh, fitted reference frame):
rows = reference left, reference right, this work left, this work right; columns = five frequencies + broadband.
|emitter| x |ear| in dB re each map's maximum, -3/-6/-12 dB contours, graticule every 30 deg.
Panels a, b: the two species (Fig. 4 of the paper; the key to rows, columns and numbers is in its caption).
Usage: python scripts/plot_ertf_pair.py [keyA keyB]  -> paper/figures/ertf_full_pair.{png,pdf}"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import grid, paths, plotting, reference, rotate
sys.path.insert(0, str(paths.ROOT / "paper" / "figures"))
import style

REF = {"pdisc": "De Mey et al. 2008", "mmicr": "Vanderelst et al. 2010"}


def species_maps(key):
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
    cols = [f"{f[k]/1e3:.0f} kHz" for k in sel] + ["Broadband"]
    rows, corr = [], {}
    for lab, dd in (("Ref.", old), ("BEM", sim)):
        for ear in ("left", "right"):
            e = dd["emitter"] * dd[ear]
            rows.append((f"{lab} {ear[0].upper()}", [e[..., k] for k in sel] + [np.sqrt((e ** 2 * w).sum(-1))]))
    for j, ear in enumerate(("left", "right")):
        A, B = rows[j][1], rows[2 + j][1]
        corr[ear] = [reference.metrics(B[k][..., None], A[k][..., None])["corr"][0] for k in range(len(cols))]
    return cols, rows, corr, call is not None


def main():
    keys = sys.argv[1:3] if len(sys.argv) > 2 else ["pdisc", "mmicr"]
    data = {k: species_maps(k) for k in keys}
    style.apply()
    nc, nr, gap, bgap = 6, 4, 0.25, 0.12                # bgap: before the broadband column
    W, lm, rm, top, rowgap = style.WIDTH, 0.16, 0.5, 0.42, 0.12          # in
    size = (W - lm - rm) / (2 * nc + gap + 2 * bgap)
    H = top + nr * (size + rowgap)
    fig = plt.figure(figsize=(W, H))
    block = [1] * (nc - 1) + [bgap, 1]
    slot = list(range(nc - 1)) + [nc]                  # gridspec column of each map column within a block
    gs = fig.add_gridspec(nr, 2 * len(block) + 1, width_ratios=block + [gap] + block, left=lm / W, right=1 - rm / W,
                          top=1 - top / H, bottom=0.0, wspace=0.0, hspace=rowgap / size)
    for b, key in enumerate(keys):
        cols, rows, corr, has_call = data[key]
        off = 0 if b == 0 else len(block) + 1
        for i, (lab, maps) in enumerate(rows):
            for c, X in enumerate(maps):
                ax = fig.add_subplot(gs[i, off + slot[c]])
                im = plotting.laea_map(ax, grid.AZ, grid.EL, 20 * np.log10(X / X.max()))
                if i == 0:
                    ax.set_title(cols[c], pad=1)
                if c == 0:
                    ax.text(-1.5, 0, lab, rotation=90, ha="right", va="center")
                if i >= 2:                              # correlation of this work with the reference, per map
                    ear = "left" if lab.endswith("L") else "right"
                    ax.text(0, -1.47, f"{corr[ear][c]:.2f}", ha="center", va="top", color="0.25")
        a0 = gs[0, off].get_position(fig)
        a1 = gs[0, off + len(block) - 1].get_position(fig)
        fig.text((a0.x0 + a1.x1) / 2, a0.y1 + 0.2 / H, style.italic(paths.species()[key]["taxon"]) +
                 f" (reference: {REF[key]})", ha="center", va="bottom")
        style.letter((a0.x0, a0.y1 + 0.2 / H), "ab"[b], dx=-10, dy=0, fig=fig)
    cax = fig.add_axes([1 - (rm - 0.1) / W, 0.18, 0.07 / W, 0.55])
    fig.colorbar(im, cax=cax, label="Two-way sensitivity (dB re max)")
    out = paths.ROOT / "paper" / "figures" / "ertf_full_pair"
    style.save(fig, out, png_dpi=250)
    for k, (cols, rows, corr, _) in data.items():
        for ear in corr:
            print(k, ear, " ".join(f"{c}: {v:.2f}" for c, v in zip(cols, corr[ear])))
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
