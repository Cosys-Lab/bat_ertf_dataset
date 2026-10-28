"""Cross-check of the production solver (gpubem.solve_lu) against NumCalc on a final dataset mesh (CT mesh, CT frame),
nominal sources of the production run, five frequencies over the band, frontal hemisphere grid.
NumCalc runs on the CPU (N_PAR processes) and can run while the GPU is busy; the GPU part runs when called with --gpu.
Usage: python tests/gpubem_vs_numcalc_ct.py key [--gpu]  -> <work>/gpubem/ct_<key>/, results/validation/gpubem_vs_numcalc_ct_<key>.txt"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from batertf import gpubem, grid, numcalc, paths, reference
import run_ertf

FREQS = [40e3, 70e3, 100e3, 130e3, 160e3]
N_PAR = 14


def main():
    key = sys.argv[1]
    sp, R, m, lm, patches = run_ertf.build(key, "ct")
    patches = {s: patches[s] for s in ("left", "right", "emitter") if s in patches}
    freqs = [f for f in FREQS if f <= sp["band_khz"][1] * 1e3]
    proj = paths.work_dir() / "gpubem" / f"ct_{key}"
    if not (proj / "project.json").exists():
        numcalc.write_project(proj, m, patches, freqs, grid_name="hemisphere")
    if "--gpu" not in sys.argv:
        ok, n = numcalc.run_steps(numcalc.steps(proj), paths.local()["numcalc"], N_PAR)
        print(f"NumCalc {ok}/{n} steps")
        return
    pts, _ = grid.unique_points("hemisphere")
    G = gpubem.solve_lu(m, patches, freqs, pts)
    lines = [f"{sp['name']} (dataset mesh, {len(m.faces)} faces): GPU BEM (LU) vs NumCalc, frontal hemisphere",
             "source  f[kHz]  rms dB  max dB  phase rms [deg]  corr(dB)"]
    for s in patches:
        f, Hn = numcalc.read_source(proj, s)
        Hg = gpubem.to_grid(G[s], "hemisphere")
        for k in range(len(f)):
            a, b = np.abs(Hg[..., k]), np.abs(Hn[..., k])
            keep = b > b.max() * 10 ** (-40 / 20)
            d = 20 * np.log10(a[keep] / b[keep])
            ph = np.angle(Hg[..., k][keep] / Hn[..., k][keep])
            c = reference.metrics(a[..., None], b[..., None])["corr"][0]
            lines.append(f"{s:8s} {f[k]/1e3:6.1f}  {np.sqrt((d**2).mean()):6.3f}  {abs(d).max():6.2f}  "
                         f"{np.degrees(np.sqrt((ph**2).mean())):8.2f}  {c:.4f}")
    (paths.RESULTS / "validation" / f"gpubem_vs_numcalc_ct_{key}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
