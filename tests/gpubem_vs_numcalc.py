"""GPU BEM (batertf/gpubem.py) against NumCalc on a real bat head: same mesh, patches and evaluation points as an
existing NumCalc run (default: the P. discolor pilot, sources left/right/emitter, 30-90 kHz).
Metrics per source and frequency, on the [el, az] grid (directions with H_numcalc above -40 dB re its max):
rms / max dB difference, magnitude-weighted rms phase difference, Pearson correlation of the dB patterns.
Only frequencies with NumCalc results are solved (runs may still be in progress; the run folder is only read).
Usage: python tests/gpubem_vs_numcalc.py [run_dir] [stl] [--timing]
       -> results/validation/gpubem_vs_numcalc_<run>.{txt,png}, arrays and timing in <work>/gpubem/<run>/
       --timing: also time one frequency with 1, 2, 4, ... sources (cost per additional source)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import gpubem, grid, mesh, numcalc, paths

RUN = Path("/data/BatERTF/work/runs/pilot_pdisc")
STL = "Phyllostomus discolor.stl"


def compare(Hg, Hn, floor_db=-40):
    dg, dn = 20 * np.log10(abs(Hg)), 20 * np.log10(abs(Hn))
    sel = dn > dn.max() + floor_db
    d = (dg - dn)[sel]
    ph = np.degrees(np.angle(Hg / Hn))[sel]
    w = abs(Hn)[sel]
    return {"rms_db": np.sqrt((d ** 2).mean()), "max_db": abs(d).max(), "p99_db": np.percentile(abs(d), 99),
            "phase_rms_deg": np.sqrt((w * ph ** 2).sum() / w.sum()), "corr": np.corrcoef(dg[sel], dn[sel])[0, 1],
            "mean_db": d.mean()}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run = Path(args[0]) if args else RUN
    stl = args[1] if len(args) > 1 else STL
    meta = json.loads((run / "project.json").read_text())
    gname = meta.get("grid", "hemisphere")
    m = mesh.load(stl)
    assert len(m.faces) == meta["n_faces"], "mesh differs from the NumCalc run"
    patches = {n: [np.array(f) for f in fs] for n, fs in meta["patch_faces_original"].items()}
    pts, _ = grid.unique_points(gname)
    freqs = [f for k, f in enumerate(meta["freqs"])
             if any(numcalc.done(run / "NumCalc" / n, k + 1) for n in patches)]
    kidx = [meta["freqs"].index(f) for f in freqs]
    out_dir = paths.work_dir() / "gpubem" / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    torch.cuda.reset_peak_memory_stats()
    H = gpubem.solve(m, patches, freqs, pts, stats=stats)
    (out_dir / "timing.json").write_text(json.dumps(stats, indent=1))
    np.savez(out_dir / "gpubem.npz", freqs=freqs, **{n: gpubem.to_grid(h, gname) for n, h in H.items()})

    lines = [f"{stl}: {len(m.faces)} faces, NumCalc run {run}, grid {gname}",
             "source  f[kHz]  rms dB  p99 dB  max dB  mean dB  phase rms [deg]  corr(dB)"]
    plot = list(patches)[:3] if len(patches) <= 3 else [list(patches)[i] for i in (0, len(patches) // 4, -1)]
    fig, axs = plt.subplots(len(plot), len(freqs), figsize=(3.2 * len(freqs) + 1, 3 * len(plot)), squeeze=False)
    i0 = np.argmin(abs(grid.EL))
    az = grid.GRIDS[gname][0]
    summary = []
    for name in patches:
        try:
            _, Hn = numcalc.read_source(run, name)
        except (IndexError, ValueError):                       # a result file still being written by NumCalc
            lines.append(f"{name:7s} skipped (NumCalc result incomplete)")
            continue
        Hn = Hn[..., kidx]
        Hg = gpubem.to_grid(H[name], gname)
        for k in range(len(freqs)):
            if np.isnan(Hn[..., k]).any():
                continue
            r = compare(Hg[..., k], Hn[..., k])
            summary.append((freqs[k], r))
            lines.append(f"{name:7s} {freqs[k]/1e3:6.1f}  {r['rms_db']:6.2f}  {r['p99_db']:6.2f}  {r['max_db']:6.2f}  "
                         f"{r['mean_db']:+6.2f}  {r['phase_rms_deg']:15.1f}  {r['corr']:8.4f}")
            if name not in plot:
                continue
            ax = axs[plot.index(name), k]
            ax.plot(az, 20 * np.log10(abs(Hn[i0, :, k])), "k-", label="NumCalc")
            ax.plot(az, 20 * np.log10(abs(Hg[i0, :, k])), "r--", label="GPU BEM")
            ax.set_title(f"{name}, {freqs[k]/1e3:.0f} kHz (el = 0)", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("Azimuth (deg)")
    for ax in axs[:, 0]:
        ax.set_ylabel("dB re free field")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    v = paths.RESULTS / "validation"
    fig.savefig(v / f"gpubem_vs_numcalc_{run.name}.png", dpi=110)
    lines.append("summary over sources: median / max of rms dB, max of max dB, max phase rms, min corr")
    for f in freqs:
        rr = [r for ff, r in summary if ff == f]
        if rr:
            g = lambda key: np.array([r[key] for r in rr])
            lines.append(f"  {f/1e3:6.1f} kHz ({len(rr)} sources): rms {np.median(g('rms_db')):.2f} / {g('rms_db').max():.2f} dB,"
                         f" max {g('max_db').max():.2f} dB, phase rms {g('phase_rms_deg').max():.1f} deg,"
                         f" corr {g('corr').min():.4f}")
    pf = stats["per_freq"]
    lines.append(f"timing ({len(patches)} sources per solve): setup {stats['setup_s']:.1f} s, near pairs {stats['near_pairs']}")
    lines += [f"  {d['f']/1e3:6.1f} kHz: assembly {d['assembly_s']:.1f} s, GMRES {d['solve_s']:.1f} s ({d['iters']} it, "
              f"res {d['residual']:.1e}), field {d['field_s']:.1f} s, peak GPU {d['peak_gb']:.1f} GB" for d in pf]
    if "--timing" in sys.argv:
        names = list(patches)
        n_src = [1, 2, 4, 8, 16, 32, len(names)] if len(names) > 3 else [1, 2, len(names)]
        lines.append(f"cost per additional source at {freqs[len(freqs)//2]/1e3:.0f} kHz (GMRES, batched right-hand sides):")
        for ns in sorted(set(n for n in n_src if n <= len(names))):
            st = {}
            gpubem.solve(m, {n: patches[n] for n in names[:ns]}, [freqs[len(freqs) // 2]], pts, stats=st,
                         log=lambda s: None)
            d = st["per_freq"][0]
            lines.append(f"  {ns:3d} sources: assembly {d['assembly_s']:.1f} s, GMRES {d['solve_s']:.1f} s "
                         f"({d['iters']} it), field {d['field_s']:.1f} s")
    txt = "\n".join(lines) + "\n"
    (v / f"gpubem_vs_numcalc_{run.name}.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
